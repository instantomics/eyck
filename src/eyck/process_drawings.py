"""Hand-drawn continuous processes on cell embeddings.

A process is drawn as one or more directed curves in the coordinates of one
embedding. Each curve maps its arc length linearly from ``value_start`` (where
drawing began) to ``value_end``. Optional regions, possibly on other embeddings,
restrict which cells receive a value: a cell is in scope when it lies inside any
include region (or when there are none) and inside no exclude region. Every cell
in scope takes the value of its orthogonal projection onto the nearest curve;
all other cells are not applicable (NaN). A process drawn with include regions
but no curve marks a subset of cells: every cell in scope takes the value 1.
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


def points_in_polygon(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd rule containment of points in one closed ring."""
    inside = np.zeros(len(points), dtype=bool)
    x, y = points[:, 0], points[:, 1]
    following = np.roll(ring, -1, axis=0)
    for (x1, y1), (x2, y2) in zip(ring, following, strict=True):
        if y1 == y2:
            continue
        crosses = (y1 > y) != (y2 > y)
        inside ^= crosses & (x < x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    return inside


def process_values(
    features: list[ProcessFeature], coordinates: dict[str, np.ndarray], n_obs: int
) -> np.ndarray:
    """Project cells onto the curves of one process; NaN outside its scope."""
    curves = [item for item in features if item.properties.role == "curve"]
    regions = [item for item in features if item.properties.role == "region"]
    values = np.full(n_obs, np.nan)
    if not curves and not regions:
        return values
    curve_embeddings = {item.properties.embedding_id for item in curves}
    if len(curve_embeddings) > 1:
        raise ValueError(
            "all curves of one process must be drawn on the same embedding"
        )
    included = [
        r for r in regions if (r.properties.region_mode or "include") == "include"
    ]
    excluded = [r for r in regions if r.properties.region_mode == "exclude"]
    scope = np.zeros(n_obs, dtype=bool) if included else np.ones(n_obs, dtype=bool)
    for region in included:
        scope |= points_in_polygon(
            coordinates[region.properties.embedding_id], _ring(region)
        )
    for region in excluded:
        scope &= ~points_in_polygon(
            coordinates[region.properties.embedding_id], _ring(region)
        )
    if not curves:
        values[scope] = 1.0 if included else np.nan
        return values
    curve_points = coordinates[curve_embeddings.pop()]
    scope &= np.isfinite(curve_points).all(axis=1)
    points = curve_points[scope]
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
    values[scope] = selected
    return values


def _ring(feature: ProcessFeature) -> np.ndarray:
    return np.asarray(feature.geometry.coordinates[0], dtype=float)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _semantic(feature: ProcessFeature) -> dict[str, Any]:
    properties = feature.properties
    return {
        "geometry": feature.geometry.model_dump(mode="json"),
        "process_id": properties.process_id,
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
                    reviewed_processes=sum(found == {"reviewed"} for found in statuses.values()),
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
        self, set_id: str, origin: str, features: list[ProcessFeature]
    ) -> ProcessDocument:
        revision = canonical_sha256(
            {
                "annotation_set": set_id,
                "features": [item.model_dump(mode="json") for item in features],
            }
        )
        return ProcessDocument(
            project_id=self.project.spec.id,
            annotation_set=set_id,
            revision=revision,
            origin=origin,
            features=features,
        )

    def _read(self, path: Path, set_id: str, origin: str) -> ProcessDocument:
        self.project._assert_generated_path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            features = [ProcessFeature.model_validate(item) for item in raw["features"]]
        except (OSError, KeyError, ValueError) as exc:
            raise ProjectError(f"invalid process drawing {path}: {exc}") from exc
        return self._document(set_id, origin, features)

    def document(self, set_id: str) -> ProcessDocument:
        directory = self._set_dir(set_id)
        for name, origin in (
            ("draft.geojson", "draft"),
            ("reviewed.geojson", "reviewed"),
        ):
            if (directory / name).exists():
                return self._read(directory / name, set_id, origin)
        return self._document(set_id, "empty", [])

    def _validate(self, set_id: str, features: list[ProcessFeature]) -> None:
        embeddings = {view.id for view in self.project.views()}
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
            if properties.embedding_id not in embeddings:
                raise ProjectError(f"unknown embedding: {properties.embedding_id}")
            geometry = feature.geometry
            expected = "LineString" if properties.role == "curve" else "Polygon"
            if geometry.type != expected:
                raise ProjectError(f"a {properties.role} must be a {expected}")
            vertices = (
                geometry.coordinates
                if geometry.type == "LineString"
                else geometry.coordinates[0]
                if len(geometry.coordinates) == 1
                else None
            )
            if vertices is None:
                raise ProjectError("regions must be single-ring polygons")
            array = np.asarray(vertices, dtype=float)
            if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
                raise ProjectError(f"feature {feature.id} has invalid coordinates")
            if len(array) < (2 if expected == "LineString" else 4):
                raise ProjectError(f"feature {feature.id} has too few vertices")
            if properties.role == "curve" and properties.region_mode is not None:
                raise ProjectError("curves have no region mode")
            if properties.role == "region" and (
                properties.value_start is not None or properties.value_end is not None
            ):
                raise ProjectError("regions have no values")
        by_process: dict[str, set[str]] = {}
        for feature in features:
            if feature.properties.role == "curve":
                by_process.setdefault(feature.properties.process_id, set()).add(
                    feature.properties.embedding_id
                )
        for process_id, used in by_process.items():
            if len(used) > 1:
                raise ProjectError(
                    f"curves of process {process_id} span several embeddings: {sorted(used)}"
                )

    def put_draft(
        self, set_id: str, expected_revision: str, features: list[ProcessFeature]
    ) -> ProcessDocument:
        directory = self._set_dir(set_id)
        with self.project.writer_lock():
            current = self.document(set_id)
            if current.revision != expected_revision:
                raise RevisionConflict("process drawing revision is stale")
            self._validate(set_id, features)
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
            document = self._document(set_id, "draft", stamped)
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
            draft = self._document(set_id, "draft", features)
            self._write_geojson(directory / "draft.geojson", draft)
            reviewed = self._document(
                set_id,
                "reviewed",
                [f for f in features if f.properties.review_status == "reviewed"],
            )
            self._write_geojson(directory / "reviewed.geojson", reviewed)
            return self._export_locked(draft)

    def export(self, set_id: str, expected_revision: str) -> ProcessExportResponse:
        with self.project.writer_lock():
            current = self.document(set_id)
            if current.revision != expected_revision:
                raise RevisionConflict("process drawing revision is stale")
            return self._export_locked(current)

    def preview(self, process_id: str, features: list[ProcessFeature]) -> ProcessValues:
        selected = [f for f in features if f.properties.process_id == process_id]
        try:
            values = self._values(selected)
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc
        finite = np.isfinite(values)
        return ProcessValues(
            process_id=process_id,
            values=[
                float(v) if ok else None for v, ok in zip(values, finite, strict=True)
            ],
            assigned=int(finite.sum()),
        )

    def _values(self, features: list[ProcessFeature]) -> np.ndarray:
        needed = {f.properties.embedding_id for f in features}
        coordinates = {embedding: self.coordinates(embedding) for embedding in needed}
        return process_values(features, coordinates, self.project.n_obs)

    def _export_locked(self, document: ProcessDocument) -> ProcessExportResponse:
        set_id = document.annotation_set
        directory = self._set_dir(set_id)
        process_ids = sorted(
            {
                f.properties.process_id
                for f in document.features
                if f.properties.role == "curve"
            }
        )
        columns: dict[str, pa.Array] = {
            "observation_id": pa.array(self.project.observation_ids, pa.string())
        }
        summary = {}
        for process_id in process_ids:
            selected = [
                f for f in document.features if f.properties.process_id == process_id
            ]
            try:
                values = self._values(selected)
            except ValueError as exc:
                raise ProjectError(f"process {process_id}: {exc}") from exc
            columns[process_id] = pa.array(
                values, pa.float64(), mask=~np.isfinite(values)
            )
            statuses = {f.properties.review_status for f in selected}
            summary[process_id] = {
                "assigned_observations": int(np.isfinite(values).sum()),
                "curves": sum(f.properties.role == "curve" for f in selected),
                "regions": sum(f.properties.role == "region" for f in selected),
                "curve_embedding": next(
                    f.properties.embedding_id
                    for f in selected
                    if f.properties.role == "curve"
                ),
                "review_status": "reviewed" if statuses == {"reviewed"} else "draft",
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
                "values.parquet has one row per project observation and one column per "
                "process; a value is the arc-length position of the cell's projection "
                "onto the nearest drawn curve, and null marks cells outside the process "
                "regions"
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
