"""Immutable H5AD queries and transactional annotation persistence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock
from pydantic import ValidationError
from scipy import sparse

from .discovery import DiscoveryError, ProjectSpec, canonical_sha256, validate_safe_id
from .models import (
    AnnotationPoints,
    DecisionRow,
    ExportResponse,
    FeatureHit,
    FeatureSearch,
    FeatureValues,
    IdentitySet,
    MembershipDocument,
    NamedInput,
    PointPayload,
    ProjectDetail,
    ProjectSummary,
    ProjectUrls,
)


class ProjectError(ValueError):
    pass


class RevisionConflict(ProjectError):
    pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _revision(project_id: str, rows: list[DecisionRow]) -> str:
    logical = {
        "schema_version": 1,
        "project_id": project_id,
        "rows": [row.model_dump(mode="json") for row in rows],
    }
    return canonical_sha256(logical)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class EyckProject:
    def __init__(self, spec: ProjectSpec):
        self.spec = spec
        self._thread_lock = threading.Lock()
        self._validate_and_identify()

    @contextmanager
    def _adata(self) -> Iterator[ad.AnnData]:
        data = ad.read_h5ad(self.spec.h5ad_path, backed="r")
        try:
            yield data
        finally:
            data.file.close()

    @property
    def draft_path(self) -> Path:
        return self.spec.output_path / "drafts" / "memberships.json"

    @property
    def lock_path(self) -> Path:
        return self.spec.output_path.parent / f".{self.spec.output_path.name}.eyck.lock"

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        self._assert_output_confined()
        with self._thread_lock:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(self.lock_path, timeout=30):
                self._assert_output_confined()
                yield

    def _assert_output_confined(self) -> None:
        resolved = self.spec.output_path.resolve(strict=False)
        if resolved == self.spec.source_root or self.spec.source_root not in resolved.parents:
            raise ProjectError("project output path no longer resolves within the source project")

    def _assert_generated_path(self, path: Path) -> None:
        output = self.spec.output_path.resolve(strict=False)
        resolved = path.resolve(strict=False)
        if resolved != output and output not in resolved.parents:
            raise ProjectError(f"generated path no longer resolves within project output: {path}")

    def _validate_and_identify(self) -> None:
        source_hash = _file_sha256(self.spec.h5ad_path)
        if self.spec.source_digest and source_hash != self.spec.source_digest.removeprefix("sha256:"):
            raise DiscoveryError(f"source digest mismatch for project {self.spec.id}")
        with self._adata() as data:
            observations = data.obs_names.astype(str).tolist()
            features = data.var_names.astype(str).tolist()
            if len(observations) != len(set(observations)):
                raise DiscoveryError(f"project {self.spec.id} observation IDs are not unique")
            if len(features) != len(set(features)):
                raise DiscoveryError(f"project {self.spec.id} feature IDs are not unique")
            missing_metadata = set(self.spec.metadata_columns) - set(data.obs.columns)
            if missing_metadata:
                raise DiscoveryError(f"missing metadata columns: {sorted(missing_metadata)}")
            if self.spec.expression_layer != "X" and self.spec.expression_layer not in data.layers:
                raise DiscoveryError(f"missing expression layer: {self.spec.expression_layer}")
            dimensions: dict[str, int] = {}
            for item in self.spec.embeddings + self.spec.modalities:
                if item.obsm not in data.obsm:
                    raise DiscoveryError(f"missing obsm input: {item.obsm}")
                shape = data.obsm[item.obsm].shape
                if len(shape) != 2 or shape[0] != data.n_obs:
                    raise DiscoveryError(f"invalid obsm input shape for {item.obsm}: {shape}")
                dimensions[item.id] = int(shape[1])
            for embedding in self.spec.embeddings:
                if dimensions[embedding.id] < 2:
                    raise DiscoveryError(f"embedding {embedding.id} needs at least two dimensions")
            self.observation_ids = observations
            self.feature_ids = features
            self.feature_symbols = (
                [str(value) for value in data.var["symbol"]]
                if "symbol" in data.var.columns
                else [None] * data.n_vars
            )
            self.n_obs = data.n_obs
            self.n_vars = data.n_vars
            self.input_dimensions = dimensions
        labels_raw = self.spec.labels.model_dump(mode="json")
        label_semantics = {
            "schema_version": 1,
            "labels": [
                {"id": label.id, "parent_ids": sorted(label.parent_ids)}
                for label in sorted(self.spec.labels.labels, key=lambda item: item.id)
            ],
        }
        self.identities = IdentitySet(
            source_sha256=source_hash,
            observation_index_sha256=_ordered_hash(observations),
            feature_index_sha256=_ordered_hash(features),
            manifest_sha256=self.spec.manifest_sha256,
            labels_revision=canonical_sha256(labels_raw),
            labels_semantic_sha256=canonical_sha256(label_semantics),
        )

    def summary(self) -> ProjectSummary:
        return ProjectSummary(
            id=self.spec.id,
            title=self.spec.title,
            description=self.spec.description,
            dataset_id=self.spec.dataset_id,
            observation_count=self.n_obs,
            feature_count=self.n_vars,
            url=f"/api/projects/{self.spec.id}",
        )

    def detail(self, csrf_token: str) -> ProjectDetail:
        base = f"/api/projects/{self.spec.id}"
        return ProjectDetail(
            id=self.spec.id,
            title=self.spec.title,
            description=self.spec.description,
            dataset_id=self.spec.dataset_id,
            source_owner=self.spec.source_owner,
            observation_count=self.n_obs,
            feature_count=self.n_vars,
            expression_layer=self.spec.expression_layer,
            metadata_columns=list(self.spec.metadata_columns),
            embeddings=[NamedInput(id=x.id, obsm=x.obsm, dimensions=self.input_dimensions[x.id]) for x in self.spec.embeddings],
            modalities=[NamedInput(id=x.id, obsm=x.obsm, dimensions=self.input_dimensions[x.id]) for x in self.spec.modalities],
            labels=self.spec.labels,
            identities=self.identities,
            csrf_token=csrf_token,
            urls=ProjectUrls(
                points=f"{base}/points",
                feature_search=f"{base}/features/search",
                feature_values=f"{base}/features/values",
                memberships=f"{base}/memberships",
                export=f"{base}/export",
                restart="/api/restart",
            ),
        )

    def points(self, embedding_id: str) -> PointPayload:
        matches = [item for item in self.spec.embeddings if item.id == embedding_id]
        if not matches:
            raise ProjectError(f"unknown embedding: {embedding_id}")
        with self._adata() as data:
            coordinates = np.asarray(data.obsm[matches[0].obsm])
            metadata = {
                column: [_json_scalar(value) for value in data.obs[column].tolist()]
                for column in self.spec.metadata_columns
            }
        return PointPayload(
            embedding_id=embedding_id,
            observation_ids=self.observation_ids,
            x=coordinates[:, 0].astype(float).tolist(),
            y=coordinates[:, 1].astype(float).tolist(),
            metadata=metadata,
        )

    def annotation_points(self, embedding_id: str) -> AnnotationPoints:
        matches = [item for item in self.spec.embeddings if item.id == embedding_id]
        if not matches:
            raise ProjectError(f"unknown embedding: {embedding_id}")
        with self._adata() as data:
            coordinates = np.asarray(data.obsm[matches[0].obsm])
            metadata = {
                column: [_json_scalar(value) for value in data.obs[column].tolist()]
                for column in self.spec.metadata_columns
            }
            modalities: dict[str, list[Any]] = {}
            for modality in self.spec.modalities:
                values = np.asarray(data.obsm[modality.obsm])
                if values.shape[1] == 1:
                    modalities[modality.id] = [_json_scalar(value) for value in values[:, 0]]
                else:
                    for dimension in range(values.shape[1]):
                        modalities[f"{modality.id}:{dimension}"] = [
                            _json_scalar(value) for value in values[:, dimension]
                        ]
        return AnnotationPoints(
            observation_ids=self.observation_ids,
            coordinates=[(float(x), float(y)) for x, y in coordinates[:, :2]],
            metadata=metadata,
            modalities=modalities,
        )

    def feature_search(self, query: str, limit: int) -> FeatureSearch:
        query_folded = query.casefold()
        hits = [
            FeatureHit(id=feature_id, index=index)
            for index, feature_id in enumerate(self.feature_ids)
            if query_folded in feature_id.casefold()
            or (
                self.feature_symbols[index] is not None
                and query_folded in self.feature_symbols[index].casefold()
            )
        ][:limit]
        return FeatureSearch(query=query, features=hits)

    def feature_values(self, feature_id: str) -> FeatureValues:
        try:
            index = self.feature_ids.index(feature_id)
        except ValueError as exc:
            raise ProjectError(f"unknown feature: {feature_id}") from exc
        with self._adata() as data:
            matrix = data.X if self.spec.expression_layer == "X" else data.layers[self.spec.expression_layer]
            selected = matrix[:, index]
            if sparse.issparse(selected):
                values = selected.toarray().reshape(-1)
            else:
                values = np.asarray(selected).reshape(-1)
        result = [None if not math.isfinite(float(value)) else float(value) for value in values]
        return FeatureValues(feature_id=feature_id, observation_ids=self.observation_ids, values=result)

    def feature_values_by_index(self, feature_index: int) -> list[float | None]:
        if feature_index < 0 or feature_index >= self.n_vars:
            raise ProjectError(f"unknown feature index: {feature_index}")
        return self.feature_values(self.feature_ids[feature_index]).values

    def _binary_state(self, value: Any, context: str) -> str:
        if pd.isna(value):
            return "unreviewed"
        if value is True or value == 1:
            return "present"
        if value is False or value == 0:
            return "absent"
        raise ProjectError(f"{context} is not binary: {value!r}")

    def _initial_rows(self) -> list[DecisionRow]:
        path = self.spec.initial_memberships_path
        if path is None:
            return []
        self._assert_generated_path(path)
        frame = pd.read_parquet(path)
        if frame.index.name == "droplet_id" and "droplet_id" not in frame.columns:
            frame = frame.reset_index()
        rows: list[DecisionRow] = []
        if {"droplet_id", "cell_type", "value"}.issubset(frame.columns):
            if "entity_id" in frame.columns:
                entities = frame["entity_id"]
            elif "index" in frame.columns:
                entities = frame["index"]
            elif frame.index.name in {"index", "entity_id"} or not isinstance(frame.index, pd.RangeIndex):
                entities = pd.Series(frame.index, index=frame.index)
            else:
                entities = pd.Series("0", index=frame.index)
            for position, (_, item) in enumerate(frame.iterrows()):
                rows.append(
                    DecisionRow(
                        support_id="generated_initial",
                        observation_id=str(item["droplet_id"]),
                        entity_id=str(entities.iloc[position]),
                        label_id=str(item["cell_type"]),
                        state=self._binary_state(item["value"], f"row {position}"),
                        decision_view_id="generated_initial",
                        provenance="generated_initial",
                    )
                )
        else:
            if "droplet_id" in frame.columns:
                observation_ids = frame.pop("droplet_id").astype(str).tolist()
            elif not isinstance(frame.index, pd.RangeIndex):
                observation_ids = frame.index.astype(str).tolist()
            else:
                raise ProjectError("wide initial memberships need a droplet_id column or named index")
            if not len(frame.columns):
                raise ProjectError("wide initial memberships have no label columns")
            for row_index, observation_id in enumerate(observation_ids):
                for label_id in frame.columns:
                    rows.append(
                        DecisionRow(
                            support_id="generated_initial",
                            observation_id=observation_id,
                            entity_id="0",
                            label_id=str(label_id),
                            state=self._binary_state(frame.iloc[row_index][label_id], f"{observation_id}/{label_id}"),
                            decision_view_id="generated_initial",
                            provenance="generated_initial",
                        )
                    )
        return rows

    def _json_document(self, path: Path, source: str) -> MembershipDocument:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            rows = [DecisionRow.model_validate(item) for item in raw["rows"]]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
            raise ProjectError(f"invalid membership document {path}: {exc}") from exc
        if raw.get("schema_version") != 1 or raw.get("project_id") != self.spec.id:
            raise ProjectError(f"membership document does not belong to project {self.spec.id}")
        self.validate_rows(rows)
        return MembershipDocument(
            project_id=self.spec.id,
            revision=_revision(self.spec.id, rows),
            source=source,
            rows=rows,
        )

    def current_memberships(self) -> MembershipDocument:
        if self.spec.reviewed_memberships_path is not None:
            return self._json_document(self.spec.reviewed_memberships_path, "reviewed")
        if self.draft_path.exists():
            self._assert_generated_path(self.draft_path)
            return self._json_document(self.draft_path, "draft")
        rows = self._initial_rows()
        self.validate_rows(rows)
        source = "generated_initial" if self.spec.initial_memberships_path else "empty"
        return MembershipDocument(
            project_id=self.spec.id,
            revision=_revision(self.spec.id, rows),
            source=source,
            rows=rows,
        )

    def validate_rows(self, rows: list[DecisionRow]) -> None:
        observations = set(self.observation_ids)
        labels = {label.id: label for label in self.spec.labels.labels}
        decisions: dict[tuple[str, str, str, str], DecisionRow] = {}
        for row in rows:
            try:
                validate_safe_id(row.support_id, "support")
                validate_safe_id(row.decision_view_id, "view")
            except DiscoveryError as exc:
                raise ProjectError(str(exc)) from exc
            if not row.entity_id:
                raise ProjectError("entity_id must not be empty")
            if row.observation_id not in observations:
                raise ProjectError(f"unknown observation ID: {row.observation_id}")
            if row.label_id not in labels:
                raise ProjectError(f"unknown label ID: {row.label_id}")
            key = (row.support_id, row.observation_id, row.entity_id, row.label_id)
            if key in decisions:
                raise ProjectError(f"duplicate membership decision: {key}")
            decisions[key] = row
        for key, row in decisions.items():
            if row.state != "present":
                continue
            for parent_id in labels[row.label_id].parent_ids:
                parent = decisions.get((*key[:3], parent_id))
                if parent is not None and parent.state == "absent":
                    raise ProjectError(f"present label {row.label_id} has absent parent {parent_id}")
        for key, row in decisions.items():
            parents = labels[row.label_id].parent_ids
            if len(parents) < 2 or row.state == "unreviewed":
                continue
            parent_states = [decisions.get((*key[:3], parent)) for parent in parents]
            if any(parent is None for parent in parent_states):
                continue
            states = [parent.state for parent in parent_states if parent is not None]
            derived = "absent" if "absent" in states else "present" if all(state == "present" for state in states) else "unreviewed"
            if row.state != derived:
                raise ProjectError(f"multiple-parent label {row.label_id} contradicts its parents")

    def put_memberships(self, expected_revision: str, rows: list[DecisionRow]) -> MembershipDocument:
        with self.writer_lock():
            current = self.current_memberships()
            if current.revision != expected_revision:
                raise RevisionConflict("membership revision is stale")
            if current.source == "reviewed":
                raise ProjectError("reviewed membership state is read-only")
            self.validate_rows(rows)
            revision = _revision(self.spec.id, rows)
            stored = {
                "schema_version": 1,
                "project_id": self.spec.id,
                "revision": revision,
                "rows": [row.model_dump(mode="json") for row in rows],
            }
            self._assert_generated_path(self.draft_path)
            _atomic_json(self.draft_path, stored)
            return MembershipDocument(
                project_id=self.spec.id,
                revision=revision,
                source="draft",
                rows=rows,
            )

    def _materialize(self, rows: list[DecisionRow]) -> list[DecisionRow]:
        labels = {label.id: label for label in self.spec.labels.labels}
        materialized = {(r.support_id, r.observation_id, r.entity_id, r.label_id): r for r in rows}
        changed = True
        while changed:
            changed = False
            snapshot = list(materialized.items())
            for key, row in snapshot:
                if row.state != "present":
                    continue
                for parent_id in labels[row.label_id].parent_ids:
                    parent_key = (*key[:3], parent_id)
                    existing = materialized.get(parent_key)
                    if existing is not None:
                        if existing.state == "absent":
                            raise ProjectError(f"present label {row.label_id} has absent parent {parent_id}")
                        continue
                    materialized[parent_key] = DecisionRow(
                        support_id=row.support_id,
                        observation_id=row.observation_id,
                        entity_id=row.entity_id,
                        label_id=parent_id,
                        state="present",
                        decision_view_id=row.decision_view_id,
                        provenance="derived_ancestor",
                    )
                    changed = True
            for label in labels.values():
                if len(label.parent_ids) < 2:
                    continue
                domains = {key[:3] for key in materialized}
                for domain in domains:
                    parent_rows = [materialized.get((*domain, parent)) for parent in label.parent_ids]
                    states = [item.state if item else "unreviewed" for item in parent_rows]
                    state = "absent" if "absent" in states else "present" if all(x == "present" for x in states) else "unreviewed"
                    key = (*domain, label.id)
                    existing = materialized.get(key)
                    if existing is not None and existing.state != "unreviewed" and existing.state != state:
                        raise ProjectError(f"multiple-parent label {label.id} contradicts its parents")
                    if state != "unreviewed" and existing is None:
                        source = next(item for item in parent_rows if item is not None)
                        materialized[key] = DecisionRow(
                            support_id=domain[0],
                            observation_id=domain[1],
                            entity_id=domain[2],
                            label_id=label.id,
                            state=state,
                            decision_view_id=source.decision_view_id,
                            provenance="derived_intersection",
                        )
                        changed = True
        return sorted(materialized.values(), key=lambda row: (row.support_id, row.observation_id, row.entity_id, row.label_id))

    def export(self, expected_revision: str) -> ExportResponse:
        with self.writer_lock():
            current = self.current_memberships()
            if current.revision != expected_revision:
                raise RevisionConflict("membership revision is stale")
            self.validate_rows(current.rows)
            materialized = [row for row in self._materialize(current.rows) if row.state != "unreviewed"]
            membership_records = [
                {
                    "support_id": row.support_id,
                    "observation_id": row.observation_id,
                    "entity_id": row.entity_id,
                    "label_id": row.label_id,
                    "state": row.state,
                    "provenance": row.provenance,
                    "decision_view_id": row.decision_view_id,
                }
                for row in materialized
            ]
            support_map: dict[tuple[str, str, str], str] = {}
            for row in materialized:
                support_map[(row.support_id, row.observation_id, row.label_id)] = row.decision_view_id
            support_records = [
                {
                    "support_id": key[0],
                    "observation_id": key[1],
                    "label_id": key[2],
                    "selection_id": key[0],
                    "accepted_view_id": view,
                }
                for key, view in sorted(support_map.items())
            ]
            self._write_export(current.revision, membership_records, support_records)
            return ExportResponse(
                revision=current.revision,
                report_url=f"/api/v1/annotations/{self.spec.id}/export/report",
                membership_rows=len(membership_records),
                support_rows=len(support_records),
            )

    def _write_export(self, revision: str, memberships: list[dict[str, Any]], support: list[dict[str, Any]]) -> None:
        output = self.spec.output_path
        output.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".export-", dir=output))
        membership_path = stage / "memberships.parquet"
        support_path = stage / "support.parquet"
        report_path = stage / "export_report.json"
        membership_schema = pa.schema([
            ("support_id", pa.string()), ("observation_id", pa.string()), ("entity_id", pa.string()),
            ("label_id", pa.string()), ("state", pa.string()), ("provenance", pa.string()),
            ("decision_view_id", pa.string()),
        ])
        support_schema = pa.schema([
            ("support_id", pa.string()), ("observation_id", pa.string()), ("label_id", pa.string()),
            ("selection_id", pa.string()), ("accepted_view_id", pa.string()),
        ])
        try:
            pq.write_table(pa.Table.from_pylist(memberships, schema=membership_schema), membership_path, compression="zstd")
            pq.write_table(pa.Table.from_pylist(support, schema=support_schema), support_path, compression="zstd")
            report = {
                "schema_version": 1,
                "project_id": self.spec.id,
                "project_state_sha256": canonical_sha256({
                    "manifest": self.identities.manifest_sha256,
                    "labels": self.identities.labels_semantic_sha256,
                    "memberships": revision,
                }),
                "membership_revision": revision,
                "identities": self.identities.model_dump(mode="json"),
                "outputs": {
                    "memberships.parquet": _file_sha256(membership_path),
                    "support.parquet": _file_sha256(support_path),
                },
                "counts": {
                    "membership_rows": len(memberships),
                    "support_rows": len(support),
                    "observations": len({row["observation_id"] for row in memberships}),
                    "entities": len({(row["observation_id"], row["entity_id"]) for row in memberships}),
                    "labels": len({row["label_id"] for row in memberships}),
                    "supports": len({row["support_id"] for row in memberships}),
                    "present": sum(row["state"] == "present" for row in memberships),
                    "absent": sum(row["state"] == "absent" for row in memberships),
                },
                "validation": {"valid": True, "errors": []},
            }
            _atomic_json(report_path, report)
            targets = [output / "memberships.parquet", output / "support.parquet", output / "export_report.json"]
            staged = [membership_path, support_path, report_path]
            backups: list[tuple[Path, Path]] = []
            promoted: list[Path] = []
            try:
                for target in targets:
                    if target.exists():
                        backup = stage / f"previous-{target.name}"
                        os.replace(target, backup)
                        backups.append((target, backup))
                for source, target in zip(staged, targets, strict=True):
                    os.replace(source, target)
                    promoted.append(target)
            except Exception:
                for target in promoted:
                    target.unlink(missing_ok=True)
                for target, backup in backups:
                    os.replace(backup, target)
                raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def export_report(self) -> dict[str, Any]:
        path = self.spec.output_path / "export_report.json"
        self._assert_generated_path(path)
        if not path.exists():
            raise ProjectError("project has not been exported")
        return json.loads(path.read_text(encoding="utf-8"))
