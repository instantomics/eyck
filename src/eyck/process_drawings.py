"""Hand-drawn continuous processes on cell embeddings.

A process is drawn as an ordered list of parts. A part's area is the cells
inside any of its include regions (every cell when it has none) and inside none
of its exclude regions; regions may lie on any embedding. Each cell belongs to
the first part, in list order, whose area holds it, and takes that part's
status:

- ``value``: the cell takes the position of its orthogonal projection onto the
  nearest curve of the part, each curve mapping its arc length linearly from
  ``value_start`` (where drawing began) to ``value_end``. A value part without
  curves marks a subset: its cells take the value 1.
- ``unknown``: the process is active in the cell but its value is not known.
- ``absent``: the process does not exist in the cell, as for every cell that no
  part holds.

Cells outside a view that embeds only a subpopulation lie in no region of that
view and cannot project onto its curves.

Drawings are stored per annotation set as GeoJSON: an autosaved draft and a
reviewed copy, both beneath the project output, mirroring Bruegel's draft and
reviewed vector annotations.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .discovery import DiscoveryError, canonical_sha256, validate_safe_id
from .models import (
    ProcessDescriptor,
    ProcessDocument,
    ProcessExportResponse,
    ProcessFeature,
    ProcessIndex,
    ProcessPart,
    ProcessReviewRequest,
    ProcessSetProgress,
    ProcessSetSummary,
    ProcessValues,
)
from .project import EyckProject, ProjectError, RevisionConflict

_CHUNK = 4096


def project_onto_polyline(
    points: np.ndarray, line: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return each point's distance to a polyline and its arc-length fraction."""
    starts = line[:-1]
    deltas = line[1:] - starts
    lengths = np.sqrt((deltas**2).sum(axis=1))
    keep = lengths > 0
    starts, deltas, lengths = starts[keep], deltas[keep], lengths[keep]
    if len(lengths) == 0:
        raise ValueError("curve has no length")
    offsets = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    total = lengths.sum()
    distance = np.empty(len(points))
    fraction = np.empty(len(points))
    for begin in range(0, len(points), _CHUNK):
        chunk = points[begin : begin + _CHUNK, None, :]
        t = np.clip(((chunk - starts) * deltas).sum(axis=2) / lengths**2, 0.0, 1.0)
        squared = ((chunk - (starts + t[:, :, None] * deltas)) ** 2).sum(axis=2)
        nearest = squared.argmin(axis=1)
        rows = np.arange(len(nearest))
        distance[begin : begin + _CHUNK] = np.sqrt(squared[rows, nearest])
        fraction[begin : begin + _CHUNK] = (
            offsets[nearest] + t[rows, nearest] * lengths[nearest]
        ) / total
    return distance, fraction


def points_in_polygon(points: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """Even-odd rule containment of points in a polygon; inner rings are holes."""
    inside = np.zeros(len(points), dtype=bool)
    x, y = points[:, 0], points[:, 1]
    for ring in rings:
        following = np.roll(ring, -1, axis=0)
        for (x1, y1), (x2, y2) in zip(ring, following, strict=True):
            if y1 == y2:
                continue
            crosses = (y1 > y) != (y2 > y)
            inside ^= crosses & (x < x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    return inside


def process_values(
    parts: list[ProcessPart],
    features: list[ProcessFeature],
    coordinates: dict[str, np.ndarray],
    n_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Values (NaN unless the status is value) and statuses of one process."""
    values = np.full(n_obs, np.nan)
    statuses = np.full(n_obs, "absent", dtype=object)
    decided = np.zeros(n_obs, dtype=bool)
    for part in parts:
        members = [item for item in features if item.properties.part_id == part.id]
        if not members:
            continue
        area = _area(members, coordinates, n_obs) & ~decided
        curves = [item for item in members if item.properties.role == "curve"]
        if part.status == "value" and curves:
            embeddings = {item.properties.embedding_id for item in curves}
            if len(embeddings) > 1:
                raise ValueError(
                    f"curves of part {part.id} span several embeddings: {sorted(embeddings)}"
                )
            curve_points = coordinates[embeddings.pop()]
            area &= np.isfinite(curve_points).all(axis=1)
            values[area] = _nearest_curve_values(curve_points[area], curves)
        elif part.status == "value":
            values[area] = 1.0
        statuses[area] = part.status
        decided |= area
    return values, statuses


def _area(
    features: list[ProcessFeature], coordinates: dict[str, np.ndarray], n_obs: int
) -> np.ndarray:
    regions = [item for item in features if item.properties.role == "region"]
    included = [
        r for r in regions if (r.properties.region_mode or "include") == "include"
    ]
    area = np.zeros(n_obs, dtype=bool) if included else np.ones(n_obs, dtype=bool)
    for region in included:
        area |= points_in_polygon(
            coordinates[region.properties.embedding_id], _rings(region)
        )
    for region in regions:
        if region.properties.region_mode == "exclude":
            area &= ~points_in_polygon(
                coordinates[region.properties.embedding_id], _rings(region)
            )
    return area


def _nearest_curve_values(
    points: np.ndarray, curves: list[ProcessFeature]
) -> np.ndarray:
    best = np.full(len(points), np.inf)
    selected = np.full(len(points), np.nan)
    for curve in curves:
        line = np.asarray(curve.geometry.coordinates, dtype=float)
        distance, fraction = project_onto_polyline(points, line)
        start = curve.properties.value_start
        end = curve.properties.value_end
        start = 0.0 if start is None else start
        end = 1.0 if end is None else end
        closer = distance < best
        best[closer] = distance[closer]
        selected[closer] = start + (end - start) * fraction[closer]
    return selected


def _rings(feature: ProcessFeature) -> list[np.ndarray]:
    return [np.asarray(ring, dtype=float) for ring in feature.geometry.coordinates]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _semantic(feature: ProcessFeature) -> dict[str, Any]:
    properties = feature.properties
    return {
        "geometry": feature.geometry.model_dump(mode="json"),
        "process_id": properties.process_id,
        "part_id": properties.part_id,
        "role": properties.role,
        "embedding_id": properties.embedding_id,
        "value_start": properties.value_start,
        "value_end": properties.value_end,
        "region_mode": properties.region_mode,
    }


class ProcessAnnotations:
    """Draft and reviewed process drawings of one Eyck project."""

    def __init__(self, project: EyckProject):
        self.project = project

    @property
    def root(self) -> Path:
        return self.project.spec.output_path / "processes"

    def _set_dir(self, set_id: str) -> Path:
        self._validate_id(set_id, "annotation set")
        return self.root / set_id

    @staticmethod
    def _validate_id(value: str, kind: str) -> None:
        try:
            validate_safe_id(value, kind)
        except DiscoveryError as exc:
            raise ProjectError(str(exc)) from exc

    def coordinates(self, embedding_id: str) -> np.ndarray:
        return self.project.embedding_coordinates(embedding_id)

    def _set_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def _exported_revision(self, set_id: str) -> str | None:
        report = self._set_dir(set_id) / "export.json"
        if not report.exists():
            return None
        return json.loads(report.read_text(encoding="utf-8"))["revision"]

    def progress(self) -> list[ProcessSetProgress]:
        """Per annotation set: processes drawn, fully reviewed, and whether the export is current."""
        progress = []
        for set_id in self._set_ids():
            document = self.document(set_id)
            statuses: dict[str, set[str]] = {}
            for feature in document.features:
                statuses.setdefault(feature.properties.process_id, set()).add(
                    feature.properties.review_status
                )
            progress.append(
                ProcessSetProgress(
                    id=set_id,
                    drawn_processes=len(statuses),
                    reviewed_processes=sum(
                        found == {"reviewed"} for found in statuses.values()
                    ),
                    exported=self._exported_revision(set_id) == document.revision,
                )
            )
        return progress

    def index(self) -> ProcessIndex:
        spec = self.project.spec
        summaries = []
        for set_id in self._set_ids():
            document = self.document(set_id)
            exported = self._exported_revision(set_id)
            summaries.append(
                ProcessSetSummary(
                    id=set_id,
                    origin=document.origin,
                    revision=document.revision,
                    feature_count=len(document.features),
                    exported_revision=exported,
                )
            )
        return ProcessIndex(
            processes=[
                ProcessDescriptor(id=p.id, title=p.title, description=p.description)
                for p in spec.processes
            ],
            annotation_sets=summaries,
            embeddings=self.project.views(),
            metadata_columns=list(spec.metadata_columns),
            overlay_columns=list(spec.overlay_columns),
        )

    def _document(
        self,
        set_id: str,
        origin: str,
        parts: list[ProcessPart],
        features: list[ProcessFeature],
    ) -> ProcessDocument:
        revision = canonical_sha256(
            {
                "annotation_set": set_id,
                "parts": [item.model_dump(mode="json") for item in parts],
                "features": [item.model_dump(mode="json") for item in features],
            }
        )
        return ProcessDocument(
            project_id=self.project.spec.id,
            annotation_set=set_id,
            revision=revision,
            origin=origin,
            parts=parts,
            features=features,
        )

    def _read(self, path: Path, set_id: str, origin: str) -> ProcessDocument:
        self.project._assert_generated_path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("schema_version") != 2:
                raise ValueError("expected schema version 2")
            parts = [ProcessPart.model_validate(item) for item in raw["parts"]]
            features = [ProcessFeature.model_validate(item) for item in raw["features"]]
        except (OSError, KeyError, ValueError) as exc:
            raise ProjectError(f"invalid process drawing {path}: {exc}") from exc
        return self._document(set_id, origin, parts, features)

    def document(self, set_id: str) -> ProcessDocument:
        directory = self._set_dir(set_id)
        for name, origin in (
            ("draft.geojson", "draft"),
            ("reviewed.geojson", "reviewed"),
        ):
            if (directory / name).exists():
                return self._read(directory / name, set_id, origin)
        return self._document(set_id, "empty", [], [])

    def _validate(
        self, set_id: str, parts: list[ProcessPart], features: list[ProcessFeature]
    ) -> None:
        embeddings = {view.id for view in self.project.views()}
        by_id: dict[str, ProcessPart] = {}
        for part in parts:
            self._validate_id(part.id, "part")
            self._validate_id(part.process_id, "process")
            if part.id in by_id:
                raise ProjectError(f"duplicate part ID: {part.id}")
            by_id[part.id] = part
        seen: set[str] = set()
        for feature in features:
            properties = feature.properties
            self._validate_id(feature.id, "feature")
            self._validate_id(properties.process_id, "process")
            if feature.id in seen:
                raise ProjectError(f"duplicate feature ID: {feature.id}")
            seen.add(feature.id)
            if properties.annotation_set != set_id:
                raise ProjectError(
                    f"feature {feature.id} belongs to another annotation set"
                )
            part = by_id.get(properties.part_id)
            if part is None or part.process_id != properties.process_id:
                raise ProjectError(
                    f"feature {feature.id} belongs to no part of its process"
                )
            if properties.role == "curve" and part.status != "value":
                raise ProjectError(f"part {part.id} has no values, so no curves")
            if properties.embedding_id not in embeddings:
                raise ProjectError(f"unknown embedding: {properties.embedding_id}")
            geometry = feature.geometry
            expected = "LineString" if properties.role == "curve" else "Polygon"
            if geometry.type != expected:
                raise ProjectError(f"a {properties.role} must be a {expected}")
            lines = (
                [geometry.coordinates]
                if geometry.type == "LineString"
                else geometry.coordinates
            )
            if not lines:
                raise ProjectError(f"feature {feature.id} has no rings")
            for vertices in lines:
                array = np.asarray(vertices, dtype=float)
                if (
                    array.ndim != 2
                    or array.shape[1] != 2
                    or not np.isfinite(array).all()
                ):
                    raise ProjectError(f"feature {feature.id} has invalid coordinates")
                if len(array) < (2 if expected == "LineString" else 4):
                    raise ProjectError(f"feature {feature.id} has too few vertices")
            if properties.role == "curve" and properties.region_mode is not None:
                raise ProjectError("curves have no region mode")
            if properties.role == "region" and (
                properties.value_start is not None or properties.value_end is not None
            ):
                raise ProjectError("regions have no values")
        by_part: dict[str, set[str]] = {}
        for feature in features:
            if feature.properties.role == "curve":
                by_part.setdefault(feature.properties.part_id, set()).add(
                    feature.properties.embedding_id
                )
        for part_id, used in by_part.items():
            if len(used) > 1:
                raise ProjectError(
                    f"curves of part {part_id} span several embeddings: {sorted(used)}"
                )

    def put_draft(
        self,
        set_id: str,
        expected_revision: str,
        parts: list[ProcessPart],
        features: list[ProcessFeature],
    ) -> ProcessDocument:
        directory = self._set_dir(set_id)
        with self.project.writer_lock():
            current = self.document(set_id)
            if current.revision != expected_revision:
                raise RevisionConflict("process drawing revision is stale")
            self._validate(set_id, parts, features)
            previous = {item.id: item for item in current.features}
            now = _now()
            stamped = []
            for feature in features:
                before = previous.get(feature.id)
                properties = feature.properties
                if before is not None and _semantic(before) == _semantic(feature):
                    properties = before.properties
                else:
                    properties = properties.model_copy(
                        update={
                            "created_at": before.properties.created_at
                            if before
                            else now,
                            "updated_at": now,
                            "review_status": "draft",
                            "reviewed_at": None,
                            "reviewed_by": None,
                        }
                    )
                stamped.append(feature.model_copy(update={"properties": properties}))
            document = self._document(set_id, "draft", parts, stamped)
            self._write_geojson(directory / "draft.geojson", document)
            return document

    def review(
        self, set_id: str, expected_revision: str, request: ProcessReviewRequest
    ) -> ProcessExportResponse:
        directory = self._set_dir(set_id)
        with self.project.writer_lock():
            current = self.document(set_id)
            if current.revision != expected_revision:
                raise RevisionConflict("process drawing revision is stale")
            targets = set(request.process_ids) if request.process_ids else None
            now = _now()
            features = [
                item.model_copy(
                    update={
                        "properties": item.properties.model_copy(
                            update={
                                "review_status": "reviewed",
                                "reviewed_at": now,
                                "reviewed_by": request.reviewer,
                            }
                        )
                    }
                )
                if (targets is None or item.properties.process_id in targets)
                and item.properties.review_status != "reviewed"
                else item
                for item in current.features
            ]
            draft = self._document(set_id, "draft", current.parts, features)
            self._write_geojson(directory / "draft.geojson", draft)
            reviewed_features = [
                f for f in features if f.properties.review_status == "reviewed"
            ]
            reviewed_parts = {f.properties.part_id for f in reviewed_features}
            reviewed = self._document(
                set_id,
                "reviewed",
                [p for p in current.parts if p.id in reviewed_parts],
                reviewed_features,
            )
            self._write_geojson(directory / "reviewed.geojson", reviewed)
            return self._export_locked(draft)

    def export(self, set_id: str, expected_revision: str) -> ProcessExportResponse:
        with self.project.writer_lock():
            current = self.document(set_id)
            if current.revision != expected_revision:
                raise RevisionConflict("process drawing revision is stale")
            return self._export_locked(current)

    def preview(
        self,
        process_id: str,
        parts: list[ProcessPart],
        features: list[ProcessFeature],
    ) -> ProcessValues:
        try:
            values, statuses = self._values(process_id, parts, features)
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc
        finite = np.isfinite(values)
        return ProcessValues(
            process_id=process_id,
            values=[
                float(v) if ok else None for v, ok in zip(values, finite, strict=True)
            ],
            statuses=statuses.tolist(),
            assigned=int(finite.sum()),
            unknown=int((statuses == "unknown").sum()),
        )

    def _values(
        self,
        process_id: str,
        parts: list[ProcessPart],
        features: list[ProcessFeature],
    ) -> tuple[np.ndarray, np.ndarray]:
        parts = [p for p in parts if p.process_id == process_id]
        features = [f for f in features if f.properties.process_id == process_id]
        needed = {f.properties.embedding_id for f in features}
        coordinates = {embedding: self.coordinates(embedding) for embedding in needed}
        return process_values(parts, features, coordinates, self.project.n_obs)

    def _export_locked(self, document: ProcessDocument) -> ProcessExportResponse:
        set_id = document.annotation_set
        directory = self._set_dir(set_id)
        process_ids = sorted({f.properties.process_id for f in document.features})
        columns: dict[str, pa.Array] = {
            "observation_id": pa.array(self.project.observation_ids, pa.string())
        }
        summary = {}
        for process_id in process_ids:
            selected = [
                f for f in document.features if f.properties.process_id == process_id
            ]
            try:
                values, statuses = self._values(
                    process_id, document.parts, document.features
                )
            except ValueError as exc:
                raise ProjectError(f"process {process_id}: {exc}") from exc
            columns[process_id] = pa.array(
                values, pa.float64(), mask=~np.isfinite(values)
            )
            columns[f"{process_id}.status"] = pa.array(
                statuses.tolist(), pa.string()
            ).dictionary_encode()
            review = {f.properties.review_status for f in selected}
            summary[process_id] = {
                "observations": {
                    status: int((statuses == status).sum())
                    for status in ("value", "unknown", "absent")
                },
                "parts": sum(p.process_id == process_id for p in document.parts),
                "curves": sum(f.properties.role == "curve" for f in selected),
                "regions": sum(f.properties.role == "region" for f in selected),
                "review_status": "reviewed" if review == {"reviewed"} else "draft",
                "annotators": sorted({f.properties.annotator for f in selected}),
            }
        values_path = directory / "values.parquet"
        self.project._mkdir_output(directory)
        descriptor, temporary = self.project._output_mkstemp(
            prefix=".values.", suffix=".parquet", dir=directory
        )
        os.close(descriptor)
        try:
            pq.write_table(pa.table(columns), temporary, compression="zstd")
            os.replace(temporary, values_path)
            self.project._chmod_output_file(values_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        report = {
            "schema_version": 1,
            "project_id": self.project.spec.id,
            "annotation_set": set_id,
            "revision": document.revision,
            "exported_at": _now(),
            "source_sha256": self.project.identities.source_sha256,
            "observation_index_sha256": self.project.identities.observation_index_sha256,
            "observations": self.project.n_obs,
            "semantics": (
                "values.parquet has one row per project observation and, per process, "
                "a value column and a '<process>.status' column; the status is the "
                "first part holding the cell (value, unknown or absent, and absent for "
                "cells in no part), and the value, set only for status value, is the "
                "arc-length position of the cell's projection onto the nearest curve "
                "of its part, or 1 in a part without curves"
            ),
            "processes": summary,
        }
        self.project._atomic_output_json(directory / "export.json", report)
        return ProcessExportResponse(
            annotation_set=set_id,
            revision=document.revision,
            values_path=str(values_path),
            report=report,
        )

    def _write_geojson(self, path: Path, document: ProcessDocument) -> None:
        payload = document.model_dump(mode="json")
        payload.pop("origin")
        self.project._atomic_output_json(path, payload)


__all__ = [
    "ProcessAnnotations",
    "points_in_polygon",
    "process_values",
    "project_onto_polyline",
]
