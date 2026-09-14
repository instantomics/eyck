"""Immutable H5AD queries and transactional annotation persistence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock
from pydantic import ValidationError
from scipy import sparse

from .discovery import (
    DiscoveryError,
    ProjectSpec,
    canonical_sha256,
    load_labels,
    validate_label_document,
    validate_safe_id,
)
from .models import (
    AnnotationPoints,
    BooleanSelectionDefinition,
    ClusteringDescriptor,
    ClusteringImport,
    ClusteringResult,
    ClusterSelectionDefinition,
    DecisionRow,
    DeletionConfirm,
    DeletionImpact,
    DeletionRequest,
    EmbeddingDescriptor,
    ExportResponse,
    ExternalClusteringImport,
    FeatureHit,
    FeatureSearch,
    FeatureValues,
    HierarchySummary,
    IdentitySet,
    LabelDocument,
    LabelImpact,
    LabelMutation,
    LabelState,
    LocalAnalysisCreate,
    MarkerClusterSummary,
    MarkerCutoffDefinition,
    MarkerProgramCreate,
    MarkerProgramDescriptor,
    MarkerProgramResult,
    MembershipDocument,
    NamedInput,
    ObservationSetDefinition,
    PointPayload,
    ProjectDetail,
    ProjectSummary,
    ProjectUrls,
    SavedSelection,
    SelectionCreate,
    WorkspaceDocument,
    WorkspaceObjectRef,
    Zoom,
    ZoomCreate,
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


def _document_revision(value: dict[str, Any]) -> str:
    logical = dict(value)
    logical.pop("revision", None)
    return canonical_sha256(logical)


OUTPUT_FILE_MODE = 0o660
OUTPUT_DIRECTORY_MODE = 0o2770


def _atomic_json(path: Path, value: Any, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    if mode is None:
        try:
            existing = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            mode = 0o600
        else:
            if not stat.S_ISREG(existing.st_mode):
                raise ProjectError(f"atomic JSON target is not a regular file: {path}")
            mode = stat.S_IMODE(existing.st_mode)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class EyckProject:
    def __init__(self, spec: ProjectSpec):
        self.spec = spec
        self._thread_lock = threading.Lock()
        self._matrix_lock = threading.Lock()
        self._matrix_cache: dict[str, tuple[Any, pd.DataFrame]] = {}
        self._marker_source: str | None = None
        self._validate_and_identify()
        self._recover_if_needed()

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
        return self.spec.output_path / ".eyck.lock"

    @property
    def transaction_path(self) -> Path:
        return self.spec.output_path / ".transaction.json"

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        self._assert_output_confined()
        with self._thread_lock:
            self._prepare_output_file(self.lock_path)
            with FileLock(self.lock_path, timeout=30):
                self._chmod_output_file(self.lock_path)
                self._assert_output_confined()
                yield

    def _assert_output_confined(self) -> None:
        resolved = self.spec.output_path.resolve(strict=False)
        if (
            resolved == self.spec.output_root
            or self.spec.output_root not in resolved.parents
        ):
            raise ProjectError(
                "project output path no longer resolves within the source project"
            )
        current = self.spec.output_root
        for part in self.spec.output_path.relative_to(self.spec.output_root).parts:
            try:
                info = current.lstat()
            except FileNotFoundError:
                break
            if stat.S_ISLNK(info.st_mode):
                raise ProjectError(f"project output path contains a symlink: {current}")
            current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode):
            raise ProjectError(f"project output path contains a symlink: {current}")

    def _assert_generated_path(self, path: Path) -> None:
        try:
            relative = path.relative_to(self.spec.output_path)
        except ValueError as exc:
            raise ProjectError(
                f"generated path is outside project output: {path}"
            ) from exc
        if ".." in relative.parts:
            raise ProjectError(f"generated path escapes project output: {path}")
        self._assert_output_confined()
        output = self.spec.output_path.resolve(strict=False)
        resolved = path.resolve(strict=False)
        if resolved != output and output not in resolved.parents:
            raise ProjectError(
                f"generated path no longer resolves within project output: {path}"
            )
        current = self.spec.output_path
        for part in relative.parts:
            try:
                info = current.lstat()
            except FileNotFoundError:
                break
            if stat.S_ISLNK(info.st_mode):
                raise ProjectError(f"generated path contains a symlink: {current}")
            current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode):
            raise ProjectError(f"generated path contains a symlink: {current}")

    def _chmod_output_directory(self, path: Path) -> None:
        self._assert_generated_path(path)
        descriptor = os.open(
            path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            current_mode = os.fstat(descriptor).st_mode
            if not stat.S_ISDIR(current_mode):
                raise ProjectError(f"generated directory is not a directory: {path}")
            if stat.S_IMODE(current_mode) != OUTPUT_DIRECTORY_MODE:
                os.fchmod(descriptor, OUTPUT_DIRECTORY_MODE)
        finally:
            os.close(descriptor)

    def _mkdir_output(self, path: Path) -> None:
        self._assert_generated_path(path)
        output = self.spec.output_path
        output.mkdir(parents=True, exist_ok=True)
        self._chmod_output_directory(output)
        current = output
        for part in path.relative_to(output).parts:
            current /= part
            current.mkdir(exist_ok=True)
            self._chmod_output_directory(current)

    def _chmod_output_file(self, path: Path) -> None:
        self._assert_generated_path(path)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            current_mode = os.fstat(descriptor).st_mode
            if not stat.S_ISREG(current_mode):
                raise ProjectError(f"generated file is not a regular file: {path}")
            if stat.S_IMODE(current_mode) != OUTPUT_FILE_MODE:
                os.fchmod(descriptor, OUTPUT_FILE_MODE)
        finally:
            os.close(descriptor)

    def _prepare_output_file(self, path: Path) -> None:
        self._mkdir_output(path.parent)
        self._assert_generated_path(path)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            OUTPUT_FILE_MODE,
        )
        try:
            current_mode = os.fstat(descriptor).st_mode
            if not stat.S_ISREG(current_mode):
                raise ProjectError(f"generated file is not a regular file: {path}")
            if stat.S_IMODE(current_mode) != OUTPUT_FILE_MODE:
                os.fchmod(descriptor, OUTPUT_FILE_MODE)
        finally:
            os.close(descriptor)

    def _output_mkstemp(
        self, *, prefix: str, suffix: str = "", dir: Path
    ) -> tuple[int, str]:
        self._mkdir_output(dir)
        descriptor, temporary = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=dir)
        os.fchmod(descriptor, OUTPUT_FILE_MODE)
        return descriptor, temporary

    def _atomic_output_json(self, path: Path, value: Any) -> None:
        self._mkdir_output(path.parent)
        self._assert_generated_path(path)
        _atomic_json(path, value, mode=OUTPUT_FILE_MODE)

    def _assert_labels_confined(self) -> None:
        resolved = self.spec.labels_path.resolve(strict=False)
        if (
            resolved == self.spec.labels_root
            or self.spec.labels_root not in resolved.parents
        ):
            raise ProjectError(
                "labels path no longer resolves within the source project"
            )

    def _transaction_targets(self) -> dict[str, Path]:
        return {
            "labels": self.spec.labels_path,
            "workspace": self.workspace_path,
            "memberships": self.draft_path,
        }

    def _recover_transaction_locked(self) -> None:
        if not self.transaction_path.exists():
            return
        self._assert_generated_path(self.transaction_path)
        try:
            journal = json.loads(self.transaction_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectError(f"invalid transaction journal: {exc}") from exc
        updates = journal.get("updates")
        if (
            journal.get("schema_version") != 1
            or journal.get("project_id") != self.spec.id
            or not isinstance(updates, dict)
            or not updates
            or set(updates) - set(self._transaction_targets())
            or journal.get("generation") != canonical_sha256(updates)
        ):
            raise ProjectError("invalid transaction journal")
        targets = self._transaction_targets()
        for name in ("labels", "workspace", "memberships"):
            if name not in updates:
                continue
            if not isinstance(updates[name], dict):
                raise ProjectError(f"invalid transaction payload for {name}")
            if name == "labels":
                self._assert_labels_confined()
            else:
                self._assert_generated_path(targets[name])
            if name == "labels":
                _atomic_json(targets[name], updates[name])
            else:
                self._atomic_output_json(targets[name], updates[name])
        self.transaction_path.unlink()
        directory = os.open(self.transaction_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _recover_if_needed(self) -> None:
        if not self.transaction_path.exists():
            return
        with self.writer_lock():
            self._recover_transaction_locked()

    def _commit_transaction(self, updates: dict[str, dict[str, Any]]) -> None:
        self._assert_generated_path(self.transaction_path)
        journal = {
            "schema_version": 1,
            "project_id": self.spec.id,
            "generation": canonical_sha256(updates),
            "updates": updates,
        }
        self._atomic_output_json(self.transaction_path, journal)
        self._recover_transaction_locked()

    def _validate_and_identify(self) -> None:
        source_hash = _file_sha256(self.spec.h5ad_path)
        if (
            self.spec.source_digest
            and source_hash != self.spec.source_digest.removeprefix("sha256:")
        ):
            raise DiscoveryError(f"source digest mismatch for project {self.spec.id}")
        with self._adata() as data:
            observations = data.obs_names.astype(str).tolist()
            features = data.var_names.astype(str).tolist()
            if len(observations) != len(set(observations)):
                raise DiscoveryError(
                    f"project {self.spec.id} observation IDs are not unique"
                )
            if len(features) != len(set(features)):
                raise DiscoveryError(
                    f"project {self.spec.id} feature IDs are not unique"
                )
            missing_metadata = set(self.spec.metadata_columns) - set(data.obs.columns)
            if missing_metadata:
                raise DiscoveryError(
                    f"missing metadata columns: {sorted(missing_metadata)}"
                )
            if (
                self.spec.expression_layer != "X"
                and self.spec.expression_layer not in data.layers
            ):
                raise DiscoveryError(
                    f"missing expression layer: {self.spec.expression_layer}"
                )
            dimensions: dict[str, int] = {}
            for item in self.spec.embeddings + self.spec.modalities:
                if item.obsm not in data.obsm:
                    raise DiscoveryError(f"missing obsm input: {item.obsm}")
                shape = data.obsm[item.obsm].shape
                if len(shape) != 2 or shape[0] != data.n_obs:
                    raise DiscoveryError(
                        f"invalid obsm input shape for {item.obsm}: {shape}"
                    )
                dimensions[item.id] = int(shape[1])
            for embedding in self.spec.embeddings:
                if dimensions[embedding.id] < 2:
                    raise DiscoveryError(
                        f"embedding {embedding.id} needs at least two dimensions"
                    )
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
        self._source_sha256 = source_hash
        self._observation_index_sha256 = _ordered_hash(observations)
        self._feature_index_sha256 = _ordered_hash(features)

    def _marker_source_name(self) -> str:
        with self._matrix_lock:
            if self._marker_source is None:
                with self._adata() as data:
                    self._marker_source = (
                        "raw"
                        if self.spec.expression_layer == "X" and data.raw is not None
                        else "X"
                        if self.spec.expression_layer == "X"
                        else f"layer:{self.spec.expression_layer}"
                    )
            return self._marker_source

    def _materialize_matrix(self, source: str) -> tuple[Any, pd.DataFrame]:
        # Source data is immutable, so one full representation can safely serve every zoom.
        with self._matrix_lock:
            cached = self._matrix_cache.get(source)
            if cached is not None:
                return cached
            with self._adata() as data:
                if source == "raw":
                    if data.raw is None:
                        raise ProjectError("H5AD raw expression is unavailable")
                    backed_matrix = data.raw.X
                    var = data.raw.var.copy()
                elif source == "X":
                    backed_matrix = data.X
                    var = data.var.copy()
                elif source.startswith("layer:"):
                    layer = source.removeprefix("layer:")
                    if layer not in data.layers:
                        raise ProjectError(
                            f"H5AD expression layer is unavailable: {layer}"
                        )
                    backed_matrix = data.layers[layer]
                    var = data.var.copy()
                else:
                    raise ProjectError(f"unknown expression source: {source}")
                materialized = (
                    backed_matrix.to_memory()
                    if hasattr(backed_matrix, "to_memory")
                    else backed_matrix[:]
                )
                if sparse.issparse(materialized):
                    materialized = materialized.copy()
                else:
                    materialized = np.asarray(materialized).copy()
            cached = (materialized, var)
            self._matrix_cache[source] = cached
            return cached

    def _local_adata(self, source: str, positions: list[int]) -> ad.AnnData:
        matrix, var = self._materialize_matrix(source)
        subset = matrix[positions, :]
        if sparse.issparse(subset):
            subset = subset.copy()
        else:
            subset = np.asarray(subset).copy()
        with self._adata() as data:
            obs = data.obs.iloc[positions].copy()
        return ad.AnnData(X=subset, obs=obs, var=var.copy())

    @property
    def identities(self) -> IdentitySet:
        labels = self.current_labels()
        labels_raw = labels.model_dump(mode="json")
        label_semantics = {
            "schema_version": 1,
            "labels": [
                {"id": label.id, "parent_ids": sorted(label.parent_ids)}
                for label in sorted(labels.labels, key=lambda item: item.id)
            ],
        }
        return IdentitySet(
            source_sha256=self._source_sha256,
            observation_index_sha256=self._observation_index_sha256,
            feature_index_sha256=self._feature_index_sha256,
            manifest_sha256=self.spec.manifest_sha256,
            labels_revision=canonical_sha256(labels_raw),
            labels_semantic_sha256=canonical_sha256(label_semantics),
        )

    def current_labels(self) -> LabelDocument:
        self._recover_if_needed()
        self._assert_labels_confined()
        try:
            document = load_labels(self.spec.labels_path)
        except DiscoveryError as exc:
            raise ProjectError(str(exc)) from exc
        if self.transaction_path.exists():
            self._recover_if_needed()
            return self.current_labels()
        return document

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
            embeddings=[
                NamedInput(id=x.id, obsm=x.obsm, dimensions=self.input_dimensions[x.id])
                for x in self.spec.embeddings
            ],
            modalities=[
                NamedInput(id=x.id, obsm=x.obsm, dimensions=self.input_dimensions[x.id])
                for x in self.spec.modalities
            ],
            labels=self.current_labels(),
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
                    modalities[modality.id] = [
                        _json_scalar(value) for value in values[:, 0]
                    ]
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
        source = (
            "X"
            if self.spec.expression_layer == "X"
            else f"layer:{self.spec.expression_layer}"
        )
        matrix, _ = self._materialize_matrix(source)
        selected = matrix[:, index]
        if sparse.issparse(selected):
            values = selected.toarray().reshape(-1)
        else:
            values = np.asarray(selected).reshape(-1)
        result = [
            None if not math.isfinite(float(value)) else float(value)
            for value in values
        ]
        return FeatureValues(
            feature_id=feature_id, observation_ids=self.observation_ids, values=result
        )

    def feature_values_by_index(self, feature_index: int) -> list[float | None]:
        if feature_index < 0 or feature_index >= self.n_vars:
            raise ProjectError(f"unknown feature index: {feature_index}")
        return self.feature_values(self.feature_ids[feature_index]).values

    @property
    def workspace_path(self) -> Path:
        return self.spec.output_path / "workspace.json"

    def _validate_object_id(self, value: str, kind: str) -> None:
        try:
            validate_safe_id(value, kind)
        except DiscoveryError as exc:
            raise ProjectError(str(exc)) from exc

    def _initial_workspace(self) -> WorkspaceDocument:
        root_hash = _ordered_hash(self.observation_ids)
        embeddings = [
            EmbeddingDescriptor(
                id=item.id,
                name=item.id,
                zoom_id="root",
                identity=canonical_sha256(
                    {
                        "source": self.identities.source_sha256,
                        "obsm": item.obsm,
                        "observations": root_hash,
                    }
                ),
                implementation="h5ad.obsm",
                parameters={
                    "obsm": item.obsm,
                    "dimensions": self.input_dimensions[item.id],
                },
                provenance={"source_h5ad_sha256": self.identities.source_sha256},
            )
            for item in self.spec.embeddings
        ]
        root = Zoom(
            id="root",
            name="All loaded observations",
            parent_id=None,
            recipe=None,
            observation_ids=list(self.observation_ids),
            observation_sha256=root_hash,
            observation_count=self.n_obs,
            fraction_of_root=1.0,
            fraction_of_parent=1.0,
            initial_embedding_id=embeddings[0].id,
        )
        raw = {
            "schema_version": 1,
            "project_id": self.spec.id,
            "revision": "",
            "source_identity": self.identities.source_sha256,
            "root_zoom_id": "root",
            "zooms": [root.model_dump(mode="json")],
            "selections": [],
            "clusterings": [],
            "embeddings": [item.model_dump(mode="json") for item in embeddings],
            "marker_programs": [],
        }
        raw["revision"] = _document_revision(raw)
        return WorkspaceDocument.model_validate(raw)

    def _read_workspace(self) -> WorkspaceDocument:
        self._assert_generated_path(self.workspace_path)
        try:
            raw = json.loads(self.workspace_path.read_text(encoding="utf-8"))
            document = WorkspaceDocument.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ProjectError(
                f"invalid workspace document {self.workspace_path}: {exc}"
            ) from exc
        if document.project_id != self.spec.id:
            raise ProjectError("workspace document belongs to another project")
        if document.source_identity != self.identities.source_sha256:
            raise ProjectError("workspace source identity does not match the H5AD")
        if document.revision != _document_revision(document.model_dump(mode="json")):
            raise ProjectError("workspace revision does not match its contents")
        self._validate_workspace(document)
        return document

    def current_workspace(self) -> WorkspaceDocument:
        self._recover_if_needed()
        if not self.workspace_path.exists():
            with self.writer_lock():
                if not self.workspace_path.exists():
                    document = self._initial_workspace()
                    self._assert_generated_path(self.workspace_path)
                    self._atomic_output_json(
                        self.workspace_path, document.model_dump(mode="json")
                    )
        document = self._read_workspace()
        if self.transaction_path.exists():
            self._recover_if_needed()
            return self.current_workspace()
        return document

    def _validate_workspace(self, document: WorkspaceDocument) -> None:
        zooms = {item.id: item for item in document.zooms}
        if len(zooms) != len(document.zooms) or set(zooms) != {
            item.id for item in document.zooms
        }:
            raise ProjectError("workspace contains duplicate zoom IDs")
        root = zooms.get(document.root_zoom_id)
        if root is None or root.parent_id is not None or root.recipe is not None:
            raise ProjectError("workspace must contain one parentless root zoom")
        if root.observation_ids != self.observation_ids:
            raise ProjectError(
                "root zoom must contain every loaded observation in source order"
            )
        known_observations = set(self.observation_ids)
        for zoom in document.zooms:
            self._validate_object_id(zoom.id, "zoom")
            if len(zoom.observation_ids) != len(set(zoom.observation_ids)):
                raise ProjectError(f"zoom {zoom.id} contains duplicate observations")
            if not set(zoom.observation_ids) <= known_observations:
                raise ProjectError(f"zoom {zoom.id} contains unknown observations")
            if zoom.observation_count != len(zoom.observation_ids):
                raise ProjectError(f"zoom {zoom.id} observation count is invalid")
            if zoom.observation_sha256 != _ordered_hash(zoom.observation_ids):
                raise ProjectError(f"zoom {zoom.id} observation identity is invalid")
            if zoom.parent_id is not None:
                parent = zooms.get(zoom.parent_id)
                if parent is None:
                    raise ProjectError(f"zoom {zoom.id} has an unknown parent")
                if not set(zoom.observation_ids) < set(parent.observation_ids):
                    raise ProjectError(
                        f"zoom {zoom.id} is not a strict subset of its parent"
                    )
        for collection, kind in (
            (document.selections, "selection"),
            (document.clusterings, "clustering"),
            (document.embeddings, "embedding"),
            (document.marker_programs, "marker program"),
        ):
            ids = [item.id for item in collection]
            if len(ids) != len(set(ids)):
                raise ProjectError(f"workspace contains duplicate {kind} IDs")
            for item in collection:
                self._validate_object_id(item.id, kind)
                if item.zoom_id not in zooms:
                    raise ProjectError(f"{kind} {item.id} has an unknown zoom")
        embedding_ids = {item.id for item in document.embeddings}
        selection_ids = {item.id for item in document.selections}
        clustering_ids = {item.id for item in document.clusterings}
        marker_program_ids = {item.id for item in document.marker_programs}
        for zoom in document.zooms:
            if zoom.initial_embedding_id not in embedding_ids:
                raise ProjectError(f"zoom {zoom.id} has an unknown initial embedding")
            if (
                zoom.recipe is not None
                and not set(zoom.recipe.selection_ids) <= selection_ids
            ):
                raise ProjectError(f"zoom {zoom.id} recipe has unknown selections")
        for selection in document.selections:
            population = set(zooms[selection.zoom_id].observation_ids)
            if not set(selection.observation_ids) <= population:
                raise ProjectError(
                    f"selection {selection.id} escapes its zoom population"
                )
            if selection.observation_count != len(selection.observation_ids):
                raise ProjectError(
                    f"selection {selection.id} observation count is invalid"
                )
            if selection.observation_sha256 != _ordered_hash(selection.observation_ids):
                raise ProjectError(
                    f"selection {selection.id} observation identity is invalid"
                )
            definition = selection.definition
            if (
                isinstance(definition, BooleanSelectionDefinition)
                and not set(definition.selection_ids) <= selection_ids
            ):
                raise ProjectError(
                    f"selection {selection.id} has unknown selection inputs"
                )
            if isinstance(definition, ClusterSelectionDefinition) and (
                definition.clustering_id not in clustering_ids
            ):
                raise ProjectError(
                    f"selection {selection.id} has an unknown clustering"
                )
            if isinstance(definition, MarkerCutoffDefinition) and (
                definition.marker_program_id not in marker_program_ids
            ):
                raise ProjectError(
                    f"selection {selection.id} has an unknown marker program"
                )

    def _prepare_workspace(self, document: WorkspaceDocument) -> WorkspaceDocument:
        raw = document.model_dump(mode="json")
        raw["revision"] = _document_revision(raw)
        saved = WorkspaceDocument.model_validate(raw)
        self._validate_workspace(saved)
        return saved

    def _save_workspace(self, document: WorkspaceDocument) -> WorkspaceDocument:
        saved = self._prepare_workspace(document)
        self._assert_generated_path(self.workspace_path)
        self._atomic_output_json(self.workspace_path, saved.model_dump(mode="json"))
        return saved

    def _workspace_for_update(self, expected_revision: str) -> WorkspaceDocument:
        current = self._read_workspace()
        if current.revision != expected_revision:
            raise RevisionConflict("workspace revision is stale")
        return current

    @staticmethod
    def _by_id(items: list[Any], object_id: str, kind: str) -> Any:
        try:
            return next(item for item in items if item.id == object_id)
        except StopIteration as exc:
            raise ProjectError(f"unknown {kind}: {object_id}") from exc

    def _resolve_recipe(
        self,
        workspace: WorkspaceDocument,
        zoom_id: str,
        operator: str,
        selection_ids: list[str],
    ) -> list[str]:
        if not selection_ids:
            raise ProjectError(
                "selection recipe must reference at least one named selection"
            )
        selections = [
            self._by_id(workspace.selections, item, "selection")
            for item in selection_ids
        ]
        if any(item.zoom_id != zoom_id for item in selections):
            raise ProjectError("selection recipe inputs must belong to the target zoom")
        resolved = [set(item.observation_ids) for item in selections]
        if operator == "union":
            selected = set().union(*resolved)
        elif operator == "intersection":
            selected = set.intersection(*resolved)
        elif operator == "exclusion":
            if len(resolved) < 2:
                raise ProjectError(
                    "exclusion needs a base selection and at least one exclusion"
                )
            selected = resolved[0] - set().union(*resolved[1:])
        else:  # pragma: no cover - Pydantic prevents this through the API
            raise ProjectError(f"unknown Boolean operator: {operator}")
        population = self._by_id(workspace.zooms, zoom_id, "zoom").observation_ids
        return [item for item in population if item in selected]

    def _clustering_values(
        self, clustering: ClusteringDescriptor
    ) -> tuple[list[str], list[str]]:
        path = self.spec.output_path / clustering.cache_path
        self._assert_generated_path(path)
        try:
            with np.load(path, allow_pickle=False) as cached:
                observations = cached["observation_ids"].astype(str).tolist()
                clusters = cached["cluster_ids"].astype(str).tolist()
        except (OSError, KeyError, ValueError) as exc:
            raise ProjectError(
                f"invalid clustering cache for {clustering.id}: {exc}"
            ) from exc
        return observations, clusters

    def clustering_result(self, clustering_id: str) -> ClusteringResult:
        workspace = self.current_workspace()
        clustering = self._by_id(workspace.clusterings, clustering_id, "clustering")
        observations, clusters = self._clustering_values(clustering)
        return ClusteringResult(
            clustering=clustering,
            observation_ids=observations,
            cluster_ids=clusters,
        )

    def _embedding_coordinates(
        self,
        workspace: WorkspaceDocument,
        zoom: Zoom,
        embedding: EmbeddingDescriptor,
    ) -> np.ndarray:
        positions = {item: index for index, item in enumerate(self.observation_ids)}
        if embedding.cache_path is None:
            source = next(
                (item for item in self.spec.embeddings if item.id == embedding.id), None
            )
            if source is None:
                raise ProjectError(f"embedding {embedding.id} has no coordinate source")
            with self._adata() as data:
                all_coordinates = np.asarray(data.obsm[source.obsm])
                coordinates = np.asarray(
                    [
                        all_coordinates[positions[item], :2]
                        for item in zoom.observation_ids
                    ]
                )
        else:
            path = self.spec.output_path / embedding.cache_path
            self._assert_generated_path(path)
            try:
                with np.load(path, allow_pickle=False) as cached:
                    cached_observations = cached["observation_ids"].astype(str).tolist()
                    cached_coordinates = np.asarray(cached["coordinates"], dtype=float)
            except (OSError, KeyError, ValueError) as exc:
                raise ProjectError(
                    f"invalid embedding cache for {embedding.id}: {exc}"
                ) from exc
            if cached_coordinates.ndim != 2 or cached_coordinates.shape[1] < 2:
                raise ProjectError(
                    f"embedding {embedding.id} has fewer than two dimensions"
                )
            coordinate_map = dict(
                zip(cached_observations, cached_coordinates, strict=True)
            )
            missing = set(zoom.observation_ids) - set(coordinate_map)
            if missing:
                raise ProjectError(
                    f"embedding {embedding.id} does not cover zoom observations: {sorted(missing)}"
                )
            coordinates = np.asarray(
                [coordinate_map[item][:2] for item in zoom.observation_ids]
            )
        if (
            coordinates.shape != (zoom.observation_count, 2)
            or not np.isfinite(coordinates).all()
        ):
            raise ProjectError(f"embedding {embedding.id} cannot render zoom {zoom.id}")
        return coordinates.astype(float)

    @staticmethod
    def _point_in_polygon(
        x: float, y: float, polygon: list[tuple[float, float]]
    ) -> bool:
        inside = False
        for index, (x1, y1) in enumerate(polygon):
            x2, y2 = polygon[(index + 1) % len(polygon)]
            cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            if (
                abs(cross) <= 1e-12
                and min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12
                and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12
            ):
                return True
            if (y1 > y) != (y2 > y):
                intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < intersection_x:
                    inside = not inside
        return inside

    def _write_cache(self, category: str, identity: str, **arrays: Any) -> str:
        directory = self.spec.output_path / "cache" / category
        self._assert_generated_path(directory)
        self._mkdir_output(directory)
        self._assert_generated_path(directory)
        path = directory / f"{identity}.npz"
        self._assert_generated_path(path)
        if not path.exists():
            descriptor, temporary = self._output_mkstemp(
                prefix=f".{identity}.", suffix=".npz", dir=directory
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    np.savez_compressed(handle, **arrays)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                self._chmod_output_file(path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return path.relative_to(self.spec.output_path).as_posix()

    def create_selection(
        self, expected_revision: str, request: SelectionCreate
    ) -> WorkspaceDocument:
        self.current_workspace()
        with self.writer_lock():
            workspace = self._workspace_for_update(expected_revision)
            self._validate_object_id(request.id, "selection")
            if any(item.id == request.id for item in workspace.selections):
                raise ProjectError(f"selection already exists: {request.id}")
            zoom = self._by_id(workspace.zooms, request.zoom_id, "zoom")
            definition = request.definition
            if isinstance(definition, ObservationSetDefinition):
                if len(definition.observation_ids) != len(
                    set(definition.observation_ids)
                ):
                    raise ProjectError("selection observations must be unique")
                if definition.kind == "lasso":
                    if definition.embedding_id is None or len(definition.polygon) < 3:
                        raise ProjectError(
                            "lasso selections require an embedding and at least three polygon points"
                        )
                    if not all(
                        np.isfinite(point).all() for point in definition.polygon
                    ):
                        raise ProjectError("lasso polygon coordinates must be finite")
                    embedding = self._by_id(
                        workspace.embeddings, definition.embedding_id, "embedding"
                    )
                    coordinates = self._embedding_coordinates(
                        workspace, zoom, embedding
                    )
                    selected = [
                        observation_id
                        for observation_id, (x, y) in zip(
                            zoom.observation_ids, coordinates, strict=True
                        )
                        if self._point_in_polygon(
                            float(x), float(y), definition.polygon
                        )
                    ]
                    if definition.observation_ids != selected:
                        raise ProjectError(
                            "submitted lasso observation IDs do not match server-side polygon replay"
                        )
                else:
                    selected_set = set(definition.observation_ids)
                    selected = [
                        item for item in zoom.observation_ids if item in selected_set
                    ]
                    if len(selected) != len(selected_set):
                        raise ProjectError(
                            "selection observations must be unique and within their zoom population"
                        )
            elif isinstance(definition, ClusterSelectionDefinition):
                clustering = self._by_id(
                    workspace.clusterings, definition.clustering_id, "clustering"
                )
                if clustering.zoom_id != request.zoom_id:
                    raise ProjectError(
                        "clustering selection must use a clustering from the same zoom"
                    )
                observations, clusters = self._clustering_values(clustering)
                wanted = set(definition.cluster_ids)
                unknown = wanted - set(clusters)
                if unknown:
                    raise ProjectError(f"unknown cluster IDs: {sorted(unknown)}")
                selected = [
                    observation
                    for observation, cluster in zip(observations, clusters, strict=True)
                    if cluster in wanted
                ]
            elif isinstance(definition, MarkerCutoffDefinition):
                program = self._by_id(
                    workspace.marker_programs,
                    definition.marker_program_id,
                    "marker program",
                )
                if program.zoom_id != request.zoom_id:
                    raise ProjectError(
                        "marker cutoff must use a program from the same zoom"
                    )
                result = self.marker_program_result(program.id, workspace=workspace)
                comparator = {
                    ">": lambda value: value > definition.cutoff,
                    ">=": lambda value: value >= definition.cutoff,
                    "<": lambda value: value < definition.cutoff,
                    "<=": lambda value: value <= definition.cutoff,
                }[definition.comparator]
                accepted = {
                    row.cluster_id
                    for row in result.cluster_table
                    if comparator(row.mean_score)
                }
                selected = [
                    item
                    for item, cluster in zip(
                        result.observation_ids, result.cluster_ids, strict=True
                    )
                    if cluster in accepted
                ]
            elif isinstance(definition, BooleanSelectionDefinition):
                if len(definition.selection_ids) < 2:
                    raise ProjectError(
                        "Boolean selections must reference at least two named selections"
                    )
                selected = self._resolve_recipe(
                    workspace,
                    request.zoom_id,
                    definition.operator,
                    definition.selection_ids,
                )
            else:  # pragma: no cover - discriminated union is exhaustive
                raise ProjectError("unsupported selection definition")
            saved = SavedSelection(
                id=request.id,
                name=request.name,
                zoom_id=request.zoom_id,
                definition=definition,
                observation_ids=selected,
                observation_sha256=_ordered_hash(selected),
                observation_count=len(selected),
            )
            workspace.selections.append(saved)
            return self._save_workspace(workspace)

    def create_zoom(
        self, expected_revision: str, request: ZoomCreate
    ) -> WorkspaceDocument:
        self.current_workspace()
        with self.writer_lock():
            workspace = self._workspace_for_update(expected_revision)
            self._validate_object_id(request.id, "zoom")
            if any(item.id == request.id for item in workspace.zooms):
                raise ProjectError(f"zoom already exists: {request.id}")
            parent = self._by_id(workspace.zooms, request.parent_id, "zoom")
            selected = self._resolve_recipe(
                workspace,
                request.parent_id,
                request.recipe.operator,
                request.recipe.selection_ids,
            )
            if not selected:
                raise ProjectError("child zoom population must not be empty")
            if len(selected) >= parent.observation_count:
                raise ProjectError("child zoom must be a strict subset of its parent")
            parent_embedding = self._by_id(
                workspace.embeddings, request.parent_embedding_id, "embedding"
            )
            self._embedding_coordinates(workspace, parent, parent_embedding)
            zoom = Zoom(
                id=request.id,
                name=request.name,
                parent_id=parent.id,
                recipe=request.recipe,
                observation_ids=selected,
                observation_sha256=_ordered_hash(selected),
                observation_count=len(selected),
                fraction_of_root=len(selected) / self.n_obs,
                fraction_of_parent=len(selected) / parent.observation_count,
                initial_embedding_id=parent_embedding.id,
            )
            workspace.zooms.append(zoom)
            return self._save_workspace(workspace)

    def import_clustering(
        self, expected_revision: str, request: ClusteringImport
    ) -> WorkspaceDocument:
        self.current_workspace()
        with self.writer_lock():
            workspace = self._workspace_for_update(expected_revision)
            self._validate_object_id(request.id, "clustering")
            if any(item.id == request.id for item in workspace.clusterings):
                raise ProjectError(f"clustering already exists: {request.id}")
            zoom = self._by_id(workspace.zooms, request.zoom_id, "zoom")
            if request.metadata_column not in self.spec.metadata_columns:
                raise ProjectError(
                    f"metadata column is not declared: {request.metadata_column}"
                )
            positions = {item: index for index, item in enumerate(self.observation_ids)}
            with self._adata() as data:
                values = [
                    "<missing>"
                    if pd.isna(data.obs.iloc[positions[item]][request.metadata_column])
                    else str(data.obs.iloc[positions[item]][request.metadata_column])
                    for item in zoom.observation_ids
                ]
            identity = canonical_sha256(
                {
                    "implementation": "eyck.metadata-column-clustering.v1",
                    "source": self.identities.source_sha256,
                    "zoom": zoom.observation_sha256,
                    "column": request.metadata_column,
                    "values": values,
                }
            )
            cache_path = self._write_cache(
                "clusterings",
                identity,
                observation_ids=np.asarray(zoom.observation_ids, dtype=str),
                cluster_ids=np.asarray(values, dtype=str),
            )
            workspace.clusterings.append(
                ClusteringDescriptor(
                    id=request.id,
                    name=request.name,
                    zoom_id=request.zoom_id,
                    identity=identity,
                    implementation="eyck.metadata-column-clustering.v1",
                    parameters={"metadata_column": request.metadata_column},
                    provenance={
                        "source_h5ad_sha256": self.identities.source_sha256,
                        "observation_sha256": zoom.observation_sha256,
                    },
                    cache_path=cache_path,
                    observation_sha256=zoom.observation_sha256,
                    cluster_count=len(set(values)),
                )
            )
            return self._save_workspace(workspace)

    def import_external_clustering(
        self,
        expected_revision: str,
        request: ExternalClusteringImport,
    ) -> WorkspaceDocument:
        self.current_workspace()
        with self.writer_lock():
            workspace = self._workspace_for_update(expected_revision)
            self._validate_object_id(request.id, "clustering")
            existing_ids = {
                item.id
                for collection in (
                    workspace.zooms,
                    workspace.selections,
                    workspace.clusterings,
                    workspace.embeddings,
                    workspace.marker_programs,
                )
                for item in collection
            }
            if request.id in existing_ids:
                raise ProjectError(f"workspace object already exists: {request.id}")
            zoom = self._by_id(workspace.zooms, request.zoom_id, "zoom")
            if request.observation_ids != zoom.observation_ids:
                raise ProjectError(
                    "external clustering observation_ids must exactly match the zoom population order"
                )
            identity = canonical_sha256(
                {
                    "source": self.identities.source_sha256,
                    "zoom": zoom.observation_sha256,
                    "observation_ids": request.observation_ids,
                    "cluster_ids": request.cluster_ids,
                    "implementation": request.implementation,
                    "parameters": request.parameters,
                    "provenance": request.provenance,
                }
            )
            cache_path = self._write_cache(
                "clusterings",
                identity,
                observation_ids=np.asarray(request.observation_ids, dtype=str),
                cluster_ids=np.asarray(request.cluster_ids, dtype=str),
            )
            workspace.clusterings.append(
                ClusteringDescriptor(
                    id=request.id,
                    name=request.name,
                    zoom_id=request.zoom_id,
                    identity=identity,
                    implementation=request.implementation,
                    parameters=request.parameters,
                    provenance={
                        **request.provenance,
                        "operator_origin": "external",
                        "source_h5ad_sha256": self.identities.source_sha256,
                        "observation_sha256": zoom.observation_sha256,
                    },
                    cache_path=cache_path,
                    observation_sha256=zoom.observation_sha256,
                    cluster_count=len(set(request.cluster_ids)),
                )
            )
            return self._save_workspace(workspace)

    def compute_local_analysis(
        self, expected_revision: str, request: LocalAnalysisCreate
    ) -> WorkspaceDocument:
        self.current_workspace()
        with self.writer_lock():
            workspace = self._workspace_for_update(expected_revision)
            requested_ids = [
                request.embedding_id,
                *(item.id for item in request.clustering_outputs),
            ]
            for value in requested_ids:
                kind = "embedding" if value == request.embedding_id else "clustering"
                self._validate_object_id(value, kind)
            existing_ids = {
                item.id
                for collection in (
                    workspace.zooms,
                    workspace.selections,
                    workspace.clusterings,
                    workspace.embeddings,
                    workspace.marker_programs,
                )
                for item in collection
            }
            duplicates = existing_ids & set(requested_ids)
            if duplicates:
                raise ProjectError(
                    f"analysis output IDs already exist: {sorted(duplicates)}"
                )
            zoom = self._by_id(workspace.zooms, request.zoom_id, "zoom")
            positions = [
                self.observation_ids.index(item) for item in zoom.observation_ids
            ]
            analysis_source = (
                "layer:counts"
                if request.use_counts
                else "X"
                if self.spec.expression_layer == "X"
                else f"layer:{self.spec.expression_layer}"
            )
            local = self._local_adata(analysis_source, positions)
            import scanpy as sc

            if request.use_counts:
                sc.pp.normalize_total(local, target_sum=1e4)
                sc.pp.log1p(local)
                expression_input = "layer:counts->normalize_total(10000)->log1p"
            elif self.spec.expression_layer != "X":
                local.X = local.layers[self.spec.expression_layer].copy()
                expression_input = f"layer:{self.spec.expression_layer}"
            else:
                expression_input = "X"
            resolved_n_top_genes = (
                request.n_top_genes or 3000
                if request.profile == "scanpy_fast"
                else None
            )
            resolved_n_comps = (
                request.n_comps or 50 if request.profile == "scanpy_fast" else None
            )
            resolved_n_pcs = (
                request.n_pcs or 50
                if request.profile == "scanpy_fast"
                else request.n_pcs
            )
            counts_parameters = {
                "enabled": request.use_counts,
                "source_layer": "counts" if request.use_counts else None,
                "normalize_total": {
                    "target_sum": 10000.0,
                    "exclude_highly_expressed": False,
                    "max_fraction": 0.05,
                    "key_added": None,
                    "layer": None,
                    "inplace": True,
                }
                if request.use_counts
                else None,
                "log1p": {"base": None, "chunked": False, "copy": False}
                if request.use_counts
                else None,
            }
            if request.profile == "scanpy_standard":
                hvg_parameters: dict[str, Any] | None = {
                    "n_top_genes": None,
                    "layer": None,
                    "min_disp": 0.5,
                    "max_disp": "inf",
                    "min_mean": 0.0125,
                    "max_mean": 3.0,
                    "span": 0.3,
                    "n_bins": 20,
                    "flavor": "seurat",
                    "subset": False,
                    "batch_key": None,
                    "inplace": True,
                    "filter_unexpressed_genes": None,
                    "check_values": True,
                }
                pca_parameters: dict[str, Any] | None = {
                    "n_comps": None,
                    "layer": None,
                    "obsm": None,
                    "zero_center": True,
                    "svd_solver": None,
                    "chunked": False,
                    "chunk_size": None,
                    "random_state": 0,
                    "return_info": False,
                    "mask_var": "highly_variable",
                    "use_highly_variable": None,
                    "dtype": "float32",
                    "key_added": None,
                    "copy": False,
                }
            elif request.profile == "scanpy_fast":
                hvg_parameters = {
                    "n_top_genes": resolved_n_top_genes,
                    "layer": None,
                    "min_disp": 0.5,
                    "max_disp": "inf",
                    "min_mean": 0.0125,
                    "max_mean": 3.0,
                    "span": 0.3,
                    "n_bins": 20,
                    "flavor": "seurat",
                    "subset": False,
                    "batch_key": None,
                    "inplace": True,
                    "filter_unexpressed_genes": None,
                    "check_values": True,
                }
                pca_parameters = {
                    "n_comps": resolved_n_comps,
                    "layer": None,
                    "obsm": None,
                    "zero_center": True,
                    "svd_solver": None,
                    "chunked": False,
                    "chunk_size": None,
                    "random_state": 0,
                    "return_info": False,
                    "mask_var": "highly_variable",
                    "use_highly_variable": None,
                    "dtype": "float32",
                    "key_added": None,
                    "copy": False,
                }
            else:
                hvg_parameters = None
                pca_parameters = None
            leiden_common = {
                "random_state": request.random_state,
                "restrict_to": None,
                "adjacency": None,
                "directed": None,
                "use_weights": True,
                "n_iterations": 2 if request.profile == "scanpy_fast" else -1,
                "partition_type": None,
                "neighbors_key": None,
                "obsp": None,
                "flavor": "igraph" if request.profile == "scanpy_fast" else None,
            }
            parameters = {
                "scanpy_version": sc.__version__,
                "profile": request.profile,
                "use_counts": request.use_counts,
                "expression_input": expression_input,
                "preprocessing": {
                    "counts": counts_parameters,
                    "highly_variable_genes": hvg_parameters,
                    "pca": pca_parameters,
                },
                "graph": {
                    "n_neighbors": request.n_neighbors,
                    "n_pcs": resolved_n_pcs,
                    "distances": None,
                    "use_rep": None,
                    "knn": True,
                    "method": "umap",
                    "transformer": None,
                    "metric": None,
                    "metric_kwds": {},
                    "random_state": request.random_state,
                    "key_added": None,
                    "copy": False,
                },
                "umap": {
                    "min_dist": 0.5,
                    "spread": 1.0,
                    "n_components": 2,
                    "maxiter": None,
                    "alpha": 1.0,
                    "gamma": 1.0,
                    "negative_sample_rate": 5,
                    "init_pos": "spectral",
                    "random_state": request.random_state,
                    "a": None,
                    "b": None,
                    "method": "umap",
                    "neighbors_key": "neighbors",
                    "key_added": None,
                    "copy": False,
                },
                "leiden": {
                    **leiden_common,
                    "outputs": [
                        {
                            "id": item.id,
                            "name": item.name,
                            "resolution": item.resolution,
                            "key_added": f"_eyck_leiden_{index}",
                        }
                        for index, item in enumerate(request.clustering_outputs)
                    ],
                    "copy": False,
                    "clustering_args": {},
                },
                "small_dataset_fallback": local.n_obs < 4,
            }
            implementation = f"scanpy-local-analysis:{request.profile}"
            clusters_by_id: dict[str, np.ndarray] = {}
            if local.n_obs < 4:
                matrix = (
                    local.X.toarray()
                    if sparse.issparse(local.X)
                    else np.asarray(local.X)
                )
                centered = matrix.astype(float) - np.mean(matrix, axis=0, keepdims=True)
                if local.n_obs > 1 and centered.shape[1] > 0:
                    left, singular, _ = np.linalg.svd(centered, full_matrices=False)
                    coordinates = left[:, :2] * singular[:2]
                else:
                    coordinates = np.zeros((local.n_obs, 0), dtype=float)
                coordinates = np.pad(
                    coordinates, ((0, 0), (0, max(0, 2 - coordinates.shape[1])))
                )[:, :2]
                for output in request.clustering_outputs:
                    clusters_by_id[output.id] = np.asarray(
                        ["0"] * local.n_obs, dtype=str
                    )
                implementation = "eyck.small-dataset-local-analysis.v1"
            else:
                if request.profile == "scanpy_standard":
                    sc.pp.highly_variable_genes(local)
                    sc.pp.pca(local)
                elif request.profile == "scanpy_fast":
                    sc.pp.highly_variable_genes(
                        local,
                        n_top_genes=resolved_n_top_genes,
                        subset=False,
                    )
                    sc.pp.pca(
                        local,
                        n_comps=resolved_n_comps,
                        mask_var="highly_variable",
                    )
                sc.pp.neighbors(
                    local,
                    n_neighbors=request.n_neighbors,
                    n_pcs=resolved_n_pcs,
                    random_state=request.random_state,
                )
                parameters["graph"]["scanpy_recorded_params"] = {
                    key: _json_scalar(value)
                    for key, value in local.uns["neighbors"].get("params", {}).items()
                }
                parameters["graph"]["resolved_representation"] = (
                    "X_pca" if "X_pca" in local.obsm else "X"
                )
                if hvg_parameters is not None:
                    hvg_parameters["selected_gene_count"] = int(
                        np.asarray(local.var["highly_variable"], dtype=bool).sum()
                    )
                if pca_parameters is not None:
                    pca_parameters["resolved_n_comps"] = int(
                        local.obsm["X_pca"].shape[1]
                    )
                sc.tl.umap(local, random_state=request.random_state)
                for index, output in enumerate(request.clustering_outputs):
                    key = f"_eyck_leiden_{index}"
                    leiden_arguments: dict[str, Any] = {
                        "resolution": output.resolution,
                        "random_state": request.random_state,
                        "key_added": key,
                    }
                    if request.profile == "scanpy_fast":
                        leiden_arguments.update(flavor="igraph", n_iterations=2)
                    sc.tl.leiden(local, **leiden_arguments)
                    clusters_by_id[output.id] = np.asarray(
                        local.obs[key].astype(str).tolist(), dtype=str
                    )
                coordinates = np.asarray(local.obsm["X_umap"], dtype=float)[:, :2]
            analysis_identity = canonical_sha256(
                {
                    "implementation": implementation,
                    "source": self.identities.source_sha256,
                    "zoom": zoom.observation_sha256,
                    "parameters": parameters,
                }
            )
            embedding_identity = canonical_sha256(
                {
                    "analysis": analysis_identity,
                    "result": "embedding",
                    "coordinates": coordinates.tolist(),
                }
            )
            embedding_cache = self._write_cache(
                "embeddings",
                embedding_identity,
                observation_ids=np.asarray(zoom.observation_ids, dtype=str),
                coordinates=coordinates,
            )
            provenance = {
                "source_h5ad_sha256": self.identities.source_sha256,
                "observation_sha256": zoom.observation_sha256,
                "analysis_identity": analysis_identity,
                "scanpy_version": sc.__version__,
                "profile": request.profile,
            }
            workspace.embeddings.append(
                EmbeddingDescriptor(
                    id=request.embedding_id,
                    name=request.embedding_name,
                    zoom_id=request.zoom_id,
                    identity=embedding_identity,
                    implementation=implementation,
                    parameters=parameters,
                    provenance=provenance,
                    cache_path=embedding_cache,
                    inherited_from_embedding_id=zoom.initial_embedding_id,
                )
            )
            for output in request.clustering_outputs:
                clusters = clusters_by_id[output.id]
                clustering_identity = canonical_sha256(
                    {
                        "analysis": analysis_identity,
                        "embedding": embedding_identity,
                        "result": "clustering",
                        "id": output.id,
                        "resolution": output.resolution,
                        "clusters": clusters.tolist(),
                    }
                )
                clustering_cache = self._write_cache(
                    "clusterings",
                    clustering_identity,
                    observation_ids=np.asarray(zoom.observation_ids, dtype=str),
                    cluster_ids=clusters,
                )
                workspace.clusterings.append(
                    ClusteringDescriptor(
                        id=output.id,
                        name=output.name,
                        zoom_id=request.zoom_id,
                        identity=clustering_identity,
                        implementation=implementation,
                        parameters={
                            **parameters,
                            "leiden_output": {
                                "id": output.id,
                                "resolution": output.resolution,
                            },
                        },
                        provenance={
                            **provenance,
                            "embedding_id": request.embedding_id,
                            "embedding_identity": embedding_identity,
                        },
                        cache_path=clustering_cache,
                        observation_sha256=zoom.observation_sha256,
                        cluster_count=len(set(clusters.tolist())),
                    )
                )
            return self._save_workspace(workspace)

    def _resolve_markers(
        self,
        requested: list[str],
        feature_ids: list[str],
        feature_symbols: list[str | None],
    ) -> list[str]:
        resolved: list[str] = []
        errors: list[str] = []
        for marker in requested:
            id_matches = [
                feature_id for feature_id in feature_ids if feature_id == marker
            ]
            if len(id_matches) == 1:
                feature_id = id_matches[0]
            elif len(id_matches) > 1:
                errors.append(f"{marker!r} is an ambiguous feature ID")
                continue
            else:
                matches = [
                    feature_ids[index]
                    for index, symbol in enumerate(feature_symbols)
                    if symbol == marker
                ]
                if not matches:
                    errors.append(f"{marker!r} is missing")
                    continue
                if len(matches) > 1:
                    errors.append(f"{marker!r} is ambiguous: {matches}")
                    continue
                feature_id = matches[0]
            if feature_id not in resolved:
                resolved.append(feature_id)
        if errors:
            raise ProjectError("marker resolution failed: " + "; ".join(errors))
        return resolved

    def create_marker_program(
        self, expected_revision: str, request: MarkerProgramCreate
    ) -> MarkerProgramResult:
        self.current_workspace()
        with self.writer_lock():
            workspace = self._workspace_for_update(expected_revision)
            self._validate_object_id(request.id, "marker program")
            if any(item.id == request.id for item in workspace.marker_programs):
                raise ProjectError(f"marker program already exists: {request.id}")
            zoom = self._by_id(workspace.zooms, request.zoom_id, "zoom")
            clustering = self._by_id(
                workspace.clusterings, request.clustering_id, "clustering"
            )
            if clustering.zoom_id != request.zoom_id:
                raise ProjectError(
                    "marker program clustering must belong to the same zoom"
                )
            positions = [
                self.observation_ids.index(item) for item in zoom.observation_ids
            ]
            marker_source = self._marker_source_name()
            local = self._local_adata(marker_source, positions)
            import scanpy as sc

            score_name = "_eyck_marker_score"
            scoring_var = local.var
            scoring_feature_ids = local.var_names.astype(str).tolist()
            expression_source = (
                marker_source.removeprefix("layer:")
                if marker_source.startswith("layer:")
                else marker_source
            )
            scoring_feature_symbols = (
                [
                    None if pd.isna(value) else str(value)
                    for value in scoring_var["symbol"]
                ]
                if "symbol" in scoring_var.columns
                else [None] * len(scoring_feature_ids)
            )
            marker_ids = self._resolve_markers(
                request.markers,
                scoring_feature_ids,
                scoring_feature_symbols,
            )
            score_arguments: dict[str, Any] = {
                "gene_list": marker_ids,
                "ctrl_size": request.ctrl_size,
                "n_bins": request.n_bins,
                "score_name": score_name,
                "random_state": request.random_state,
                "use_raw": False,
            }
            sc.tl.score_genes(local, **score_arguments)
            scores = local.obs[score_name].astype(float).to_numpy()
            observations, clusters = self._clustering_values(clustering)
            if observations != zoom.observation_ids:
                raise ProjectError(
                    "clustering cache observation order does not match the zoom"
                )
            cluster_array = np.asarray(clusters, dtype=str)
            means: dict[str, float] = {}
            table: list[MarkerClusterSummary] = []
            for cluster_id in sorted(set(clusters)):
                mask = cluster_array == cluster_id
                mean = float(np.mean(scores[mask]))
                means[cluster_id] = mean
                table.append(
                    MarkerClusterSummary(
                        cluster_id=cluster_id,
                        cell_count=int(np.sum(mask)),
                        mean_score=mean,
                    )
                )
            means_by_cell = np.asarray([means[item] for item in clusters], dtype=float)
            parameters = {
                "ctrl_size": request.ctrl_size,
                "n_bins": request.n_bins,
                "random_state": request.random_state,
                "use_raw": False if self.spec.expression_layer != "X" else None,
                "expression_layer": self.spec.expression_layer,
                "expression_source": expression_source,
            }
            provenance = {
                "source_h5ad_sha256": self.identities.source_sha256,
                "observation_sha256": zoom.observation_sha256,
                "expression_source": expression_source,
            }
            implementation = (
                f"scanpy.tl.score_genes:{sc.__version__}+arithmetic-cluster-mean"
            )
            identity = canonical_sha256(
                {
                    "implementation": implementation,
                    "source": self.identities.source_sha256,
                    "zoom": zoom.observation_sha256,
                    "markers": marker_ids,
                    "clustering": clustering.identity,
                    "parameters": parameters,
                    "provenance": provenance,
                    "scores": scores.tolist(),
                }
            )
            cache_path = self._write_cache(
                "marker_programs",
                identity,
                observation_ids=np.asarray(observations, dtype=str),
                per_cell_scores=scores,
                cluster_mean_by_cell=means_by_cell,
                cluster_ids=cluster_array,
            )
            program = MarkerProgramDescriptor(
                id=request.id,
                name=request.name,
                zoom_id=request.zoom_id,
                identity=identity,
                marker_ids=marker_ids,
                requested_markers=request.markers,
                clustering_id=clustering.id,
                clustering_identity=clustering.identity,
                implementation=implementation,
                parameters=parameters,
                provenance=provenance,
                cache_path=cache_path,
                cluster_table=table,
            )
            workspace.marker_programs.append(program)
            self._save_workspace(workspace)
            return MarkerProgramResult(
                program=program,
                observation_ids=observations,
                per_cell_scores=scores.tolist(),
                cluster_mean_by_cell=means_by_cell.tolist(),
                cluster_ids=clusters,
                cluster_table=table,
            )

    def marker_program_result(
        self, program_id: str, *, workspace: WorkspaceDocument | None = None
    ) -> MarkerProgramResult:
        workspace = workspace or self.current_workspace()
        program = self._by_id(workspace.marker_programs, program_id, "marker program")
        path = self.spec.output_path / program.cache_path
        self._assert_generated_path(path)
        try:
            with np.load(path, allow_pickle=False) as cached:
                observations = cached["observation_ids"].astype(str).tolist()
                scores = cached["per_cell_scores"].astype(float).tolist()
                means = cached["cluster_mean_by_cell"].astype(float).tolist()
                clusters = cached["cluster_ids"].astype(str).tolist()
        except (OSError, KeyError, ValueError) as exc:
            raise ProjectError(f"invalid marker cache for {program.id}: {exc}") from exc
        return MarkerProgramResult(
            program=program,
            observation_ids=observations,
            per_cell_scores=scores,
            cluster_mean_by_cell=means,
            cluster_ids=clusters,
            cluster_table=program.cluster_table,
        )

    def workspace_points(self, zoom_id: str, embedding_id: str) -> AnnotationPoints:
        workspace = self.current_workspace()
        zoom = self._by_id(workspace.zooms, zoom_id, "zoom")
        embedding = self._by_id(workspace.embeddings, embedding_id, "embedding")
        positions = {item: index for index, item in enumerate(self.observation_ids)}
        coordinates = self._embedding_coordinates(workspace, zoom, embedding)
        with self._adata() as data:
            metadata = {
                column: [
                    _json_scalar(data.obs.iloc[positions[item]][column])
                    for item in zoom.observation_ids
                ]
                for column in self.spec.metadata_columns
            }
            modalities: dict[str, list[Any]] = {}
            for modality in self.spec.modalities:
                values = np.asarray(data.obsm[modality.obsm])
                if values.shape[1] == 1:
                    modalities[modality.id] = [
                        _json_scalar(values[positions[item], 0])
                        for item in zoom.observation_ids
                    ]
                else:
                    for dimension in range(values.shape[1]):
                        modalities[f"{modality.id}:{dimension}"] = [
                            _json_scalar(values[positions[item], dimension])
                            for item in zoom.observation_ids
                        ]
        return AnnotationPoints(
            observation_ids=zoom.observation_ids,
            coordinates=[(float(x), float(y)) for x, y in coordinates[:, :2]],
            metadata=metadata,
            modalities=modalities,
        )

    def _deletion_impact(
        self,
        workspace: WorkspaceDocument,
        memberships: MembershipDocument,
        request: DeletionRequest,
    ) -> DeletionImpact:
        collections = {
            "zoom": workspace.zooms,
            "selection": workspace.selections,
            "clustering": workspace.clusterings,
            "embedding": workspace.embeddings,
            "marker_program": workspace.marker_programs,
        }
        if request.kind == "zoom" and request.id == workspace.root_zoom_id:
            raise ProjectError("the root zoom cannot be deleted")
        requested_object = self._by_id(
            collections[request.kind], request.id, request.kind
        )
        if request.kind == "embedding" and requested_object.cache_path is None:
            raise ProjectError("source H5AD embeddings cannot be deleted")
        removed = {(request.kind, request.id)}
        changed = True
        while changed:
            changed = False
            removed_zooms = {item_id for kind, item_id in removed if kind == "zoom"}
            removed_selections = {
                item_id for kind, item_id in removed if kind == "selection"
            }
            removed_clusterings = {
                item_id for kind, item_id in removed if kind == "clustering"
            }
            removed_embeddings = {
                item_id for kind, item_id in removed if kind == "embedding"
            }
            removed_programs = {
                item_id for kind, item_id in removed if kind == "marker_program"
            }
            candidates: set[tuple[str, str]] = set()
            for zoom in workspace.zooms:
                if (
                    zoom.parent_id in removed_zooms
                    or (
                        zoom.recipe
                        and set(zoom.recipe.selection_ids) & removed_selections
                    )
                    or zoom.initial_embedding_id in removed_embeddings
                ):
                    candidates.add(("zoom", zoom.id))
            for selection in workspace.selections:
                definition = selection.definition
                dependent = selection.zoom_id in removed_zooms
                if isinstance(definition, BooleanSelectionDefinition):
                    dependent |= bool(
                        set(definition.selection_ids) & removed_selections
                    )
                elif isinstance(definition, ClusterSelectionDefinition):
                    dependent |= definition.clustering_id in removed_clusterings
                elif isinstance(definition, MarkerCutoffDefinition):
                    dependent |= definition.marker_program_id in removed_programs
                elif isinstance(definition, ObservationSetDefinition):
                    dependent |= definition.embedding_id in removed_embeddings
                if dependent:
                    candidates.add(("selection", selection.id))
            for clustering in workspace.clusterings:
                if clustering.zoom_id in removed_zooms:
                    candidates.add(("clustering", clustering.id))
            for embedding in workspace.embeddings:
                if embedding.zoom_id in removed_zooms:
                    candidates.add(("embedding", embedding.id))
            for program in workspace.marker_programs:
                if (
                    program.zoom_id in removed_zooms
                    or program.clustering_id in removed_clusterings
                ):
                    candidates.add(("marker_program", program.id))
            before = len(removed)
            removed |= candidates
            changed = len(removed) != before
        order = {
            "zoom": 0,
            "selection": 1,
            "clustering": 2,
            "embedding": 3,
            "marker_program": 4,
        }
        references = [
            WorkspaceObjectRef(kind=kind, id=item_id)
            for kind, item_id in sorted(
                removed, key=lambda item: (order[item[0]], item[1])
            )
        ]
        counts = {
            kind: sum(item.kind == kind for item in references) for kind in collections
        }
        removed_selection_ids = {
            item.id for item in references if item.kind == "selection"
        }
        removed_zoom_ids = {item.id for item in references if item.kind == "zoom"}
        affected_memberships = [
            row
            for row in memberships.rows
            if row.selection_id in removed_selection_ids
            or row.support_id in removed_selection_ids
            or row.zoom_id in removed_zoom_ids
        ]
        payload = {
            "workspace_revision": workspace.revision,
            "membership_revision": memberships.revision,
            "requested": WorkspaceObjectRef(
                kind=request.kind, id=request.id
            ).model_dump(mode="json"),
            "removed": [item.model_dump(mode="json") for item in references],
            "counts": counts,
            "membership_decision_count": len(affected_memberships),
            "membership_observation_ids": sorted(
                {row.observation_id for row in affected_memberships}
            ),
        }
        return DeletionImpact(
            **payload,
            impact_sha256=canonical_sha256(payload),
        )

    def deletion_impact(self, request: DeletionRequest) -> DeletionImpact:
        workspace = self.current_workspace()
        memberships = self.current_memberships()
        return self._deletion_impact(workspace, memberships, request)

    def delete_workspace_objects(
        self, expected_revision: str, request: DeletionConfirm
    ) -> WorkspaceDocument:
        with self.writer_lock():
            workspace = self._workspace_for_update(expected_revision)
            memberships = self._current_memberships()
            if memberships.revision != request.expected_membership_revision:
                raise RevisionConflict("membership revision is stale")
            current_preview = self._deletion_impact(
                workspace,
                memberships,
                DeletionRequest(kind=request.kind, id=request.id),
            )
            if (
                current_preview.impact_sha256 != request.impact_sha256
                or current_preview.removed != request.confirmed_removed
            ):
                raise RevisionConflict("deletion impact changed")
            removed = {(item.kind, item.id) for item in current_preview.removed}
            workspace.zooms = [
                item for item in workspace.zooms if ("zoom", item.id) not in removed
            ]
            workspace.selections = [
                item
                for item in workspace.selections
                if ("selection", item.id) not in removed
            ]
            workspace.clusterings = [
                item
                for item in workspace.clusterings
                if ("clustering", item.id) not in removed
            ]
            workspace.embeddings = [
                item
                for item in workspace.embeddings
                if ("embedding", item.id) not in removed
            ]
            workspace.marker_programs = [
                item
                for item in workspace.marker_programs
                if ("marker_program", item.id) not in removed
            ]
            removed_selection_ids = {
                item_id for kind, item_id in removed if kind == "selection"
            }
            removed_zoom_ids = {item_id for kind, item_id in removed if kind == "zoom"}
            retained_rows = [
                row
                for row in memberships.rows
                if row.selection_id not in removed_selection_ids
                and row.support_id not in removed_selection_ids
                and row.zoom_id not in removed_zoom_ids
            ]
            if (
                len(retained_rows) != len(memberships.rows)
                and memberships.source == "reviewed"
            ):
                raise ProjectError(
                    "cannot cascade deletion into reviewed membership state"
                )
            saved = self._prepare_workspace(workspace)
            updates = {"workspace": saved.model_dump(mode="json")}
            if len(retained_rows) != len(memberships.rows):
                updates["memberships"] = {
                    "schema_version": 1,
                    "project_id": self.spec.id,
                    "revision": _revision(self.spec.id, retained_rows),
                    "rows": [row.model_dump(mode="json") for row in retained_rows],
                }
            self._commit_transaction(updates)
            return saved

    def current_label_state(self) -> LabelState:
        labels = self.current_labels()
        return LabelState(
            revision=canonical_sha256(labels.model_dump(mode="json")),
            labels=labels.labels,
        )

    def _proposed_labels(
        self,
        request: LabelMutation,
        current: LabelDocument,
        memberships: MembershipDocument,
        workspace: WorkspaceDocument,
    ) -> tuple[LabelDocument, set[str], bool, LabelImpact]:
        current_by_id = {item.id: item for item in current.labels}
        proposed_by_id: dict[str, Any] = {}
        for label in request.labels:
            self._validate_object_id(label.id, "label")
            if label.id in proposed_by_id:
                raise ProjectError(f"duplicate label ID: {label.id}")
            proposed_by_id[label.id] = label
        requested_deletions = set(request.delete_label_ids)
        unknown = requested_deletions - set(current_by_id)
        if unknown:
            raise ProjectError(
                f"unknown labels requested for deletion: {sorted(unknown)}"
            )
        for label_id in requested_deletions:
            self._validate_object_id(label_id, "label")
        removed = set(requested_deletions)
        changed = True
        while changed:
            before = len(removed)
            removed.update(
                label.id for label in current.labels if set(label.parent_ids) & removed
            )
            changed = len(removed) != before
        accidentally_missing = set(current_by_id) - set(proposed_by_id) - removed
        if accidentally_missing:
            raise ProjectError(
                "existing label IDs are immutable and may only be removed through delete_label_ids: "
                f"{sorted(accidentally_missing)}"
            )
        surviving = [label for label in request.labels if label.id not in removed]
        try:
            document = validate_label_document(
                LabelDocument(schema_version=1, labels=surviving)
            )
        except DiscoveryError as exc:
            raise ProjectError(str(exc)) from exc
        changed_ids = sorted(
            label.id
            for label in surviving
            if label.id in current_by_id and label != current_by_id[label.id]
        )
        added_ids = sorted(set(proposed_by_id) - set(current_by_id) - removed)
        current_edges = {
            (label.id, parent_id)
            for label in current.labels
            for parent_id in label.parent_ids
        }
        proposed_edges = {
            (label.id, parent_id)
            for label in surviving
            for parent_id in label.parent_ids
        }
        child_edges = sorted(
            f"{child}->{parent}" for child, parent in current_edges ^ proposed_edges
        )
        semantic_edge_change = any(
            (label.id not in current_by_id and bool(label.parent_ids))
            or (
                label.id in current_by_id
                and label.parent_ids != current_by_id[label.id].parent_ids
            )
            for label in surviving
        )
        if semantic_edge_change:
            affected_rows = list(memberships.rows)
        else:
            affected_rows = [row for row in memberships.rows if row.label_id in removed]
        workspace_selection_ids = {item.id for item in workspace.selections}
        affected_selection_ids = sorted(
            {
                selection_id
                for row in affected_rows
                for selection_id in (row.selection_id, row.support_id)
                if selection_id in workspace_selection_ids
            }
        )
        labels_revision = canonical_sha256(current.model_dump(mode="json"))
        proposed_revision = canonical_sha256(document.model_dump(mode="json"))
        impact_payload = {
            "labels_revision": labels_revision,
            "proposed_revision": proposed_revision,
            "membership_revision": memberships.revision,
            "removed_label_ids": sorted(removed),
            "changed_label_ids": changed_ids,
            "added_label_ids": added_ids,
            "child_edge_ids": child_edges,
            "membership_decision_count": len(affected_rows),
            "membership_observation_ids": sorted(
                {row.observation_id for row in affected_rows}
            ),
            "selection_ids": affected_selection_ids,
        }
        proposal_payload = {
            "labels": document.model_dump(mode="json"),
            "delete_label_ids": sorted(requested_deletions),
            "impact": impact_payload,
        }
        impact = LabelImpact(
            **impact_payload,
            impact_sha256=canonical_sha256(proposal_payload),
        )
        retained_rows = (
            []
            if semantic_edge_change
            else [row for row in memberships.rows if row.label_id not in removed]
        )
        self.validate_rows(retained_rows, labels=document)
        return document, removed, semantic_edge_change, impact

    def label_impact(self, request: LabelMutation) -> LabelImpact:
        workspace = self.current_workspace()
        current = self.current_labels()
        memberships = self.current_memberships()
        return self._proposed_labels(request, current, memberships, workspace)[3]

    def put_labels(self, expected_revision: str, request: LabelMutation) -> LabelState:
        self.current_workspace()
        if (
            request.expected_membership_revision is None
            or request.expected_impact_sha256 is None
        ):
            raise ProjectError(
                "label PUT requires expected_membership_revision and expected_impact_sha256"
            )
        with self.writer_lock():
            current = self.current_labels()
            current_revision = canonical_sha256(current.model_dump(mode="json"))
            if current_revision != expected_revision:
                raise RevisionConflict("labels revision is stale")
            memberships = self._current_memberships()
            if memberships.revision != request.expected_membership_revision:
                raise RevisionConflict("membership revision is stale")
            workspace = self._read_workspace()
            document, removed, semantic_edge_change, impact = self._proposed_labels(
                request,
                current,
                memberships,
                workspace,
            )
            if impact.impact_sha256 != request.expected_impact_sha256:
                raise RevisionConflict("label impact changed")
            if (removed or semantic_edge_change) and not request.confirm_cascade:
                raise ProjectError(
                    "label deletion and semantic parent changes require confirm_cascade"
                )
            retained_rows = (
                []
                if semantic_edge_change
                else [row for row in memberships.rows if row.label_id not in removed]
            )
            if (
                len(retained_rows) != len(memberships.rows)
                and memberships.source == "reviewed"
            ):
                raise ProjectError(
                    "cannot cascade label mutation into reviewed membership state"
                )
            updates = {"labels": document.model_dump(mode="json")}
            if len(retained_rows) != len(memberships.rows):
                updates["memberships"] = {
                    "schema_version": 1,
                    "project_id": self.spec.id,
                    "revision": _revision(self.spec.id, retained_rows),
                    "rows": [row.model_dump(mode="json") for row in retained_rows],
                }
            self._commit_transaction(updates)
            return LabelState(
                revision=impact.proposed_revision,
                labels=document.labels,
            )

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
            elif frame.index.name in {"index", "entity_id"} or not isinstance(
                frame.index, pd.RangeIndex
            ):
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
                raise ProjectError(
                    "wide initial memberships need a droplet_id column or named index"
                )
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
                            state=self._binary_state(
                                frame.iloc[row_index][label_id],
                                f"{observation_id}/{label_id}",
                            ),
                            decision_view_id="generated_initial",
                            provenance="generated_initial",
                        )
                    )
        return rows

    def _json_document(self, path: Path, source: str) -> MembershipDocument:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            rows = [DecisionRow.model_validate(item) for item in raw["rows"]]
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValidationError,
        ) as exc:
            raise ProjectError(f"invalid membership document {path}: {exc}") from exc
        if raw.get("schema_version") != 1 or raw.get("project_id") != self.spec.id:
            raise ProjectError(
                f"membership document does not belong to project {self.spec.id}"
            )
        self.validate_rows(rows)
        return MembershipDocument(
            project_id=self.spec.id,
            revision=_revision(self.spec.id, rows),
            source=source,
            rows=rows,
        )

    def current_memberships(self) -> MembershipDocument:
        self._recover_if_needed()
        document = self._current_memberships()
        if self.transaction_path.exists():
            self._recover_if_needed()
            return self.current_memberships()
        return document

    def _current_memberships(self) -> MembershipDocument:
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

    def validate_rows(
        self, rows: list[DecisionRow], *, labels: LabelDocument | None = None
    ) -> None:
        observations = set(self.observation_ids)
        label_map = {
            label.id: label for label in (labels or self.current_labels()).labels
        }
        decisions: dict[tuple[str, str, str, str], DecisionRow] = {}
        linked_rows = [
            row
            for row in rows
            if row.selection_id is not None or row.zoom_id is not None
        ]
        workspace = self.current_workspace() if linked_rows else None
        selection_map = (
            {item.id: item for item in workspace.selections} if workspace else {}
        )
        selection_observations = (
            {item.id: set(item.observation_ids) for item in workspace.selections}
            if workspace
            else {}
        )
        zoom_map = {item.id: item for item in workspace.zooms} if workspace else {}
        zoom_observations = (
            {item.id: set(item.observation_ids) for item in workspace.zooms}
            if workspace
            else {}
        )
        for row in rows:
            try:
                validate_safe_id(row.support_id, "support")
                validate_safe_id(row.decision_view_id, "view")
            except DiscoveryError as exc:
                raise ProjectError(str(exc)) from exc
            if not row.entity_id or len(row.entity_id) > 128:
                raise ProjectError(
                    "entity_id must contain between 1 and 128 characters"
                )
            if any(
                ord(character) < 32 or ord(character) == 127
                for character in row.entity_id
            ):
                raise ProjectError("entity_id must not contain control characters")
            if row.observation_id not in observations:
                raise ProjectError(f"unknown observation ID: {row.observation_id}")
            if row.label_id not in label_map:
                raise ProjectError(f"unknown label ID: {row.label_id}")
            if len(label_map[row.label_id].parent_ids) > 1:
                raise ProjectError(
                    f"label {row.label_id} has multiple parents and is a derived intersection"
                )
            if (row.selection_id is None) != (row.zoom_id is None):
                raise ProjectError("selection_id and zoom_id must be provided together")
            if row.selection_id is not None and row.zoom_id is not None:
                self._validate_object_id(row.selection_id, "selection")
                self._validate_object_id(row.zoom_id, "zoom")
                try:
                    selection = selection_map[row.selection_id]
                except KeyError as exc:
                    raise ProjectError(
                        f"unknown selection: {row.selection_id}"
                    ) from exc
                try:
                    zoom = zoom_map[row.zoom_id]
                except KeyError as exc:
                    raise ProjectError(f"unknown zoom: {row.zoom_id}") from exc
                if selection.zoom_id != zoom.id:
                    raise ProjectError(
                        "linked selection does not belong to the linked zoom"
                    )
                if row.observation_id not in selection_observations[row.selection_id]:
                    raise ProjectError(
                        "linked observation is not in the linked selection"
                    )
                if row.observation_id not in zoom_observations[row.zoom_id]:
                    raise ProjectError("linked observation is not in the linked zoom")
            key = (row.support_id, row.observation_id, row.entity_id, row.label_id)
            if key in decisions:
                raise ProjectError(f"duplicate membership decision: {key}")
            decisions[key] = row
        for key, row in decisions.items():
            if row.state != "present":
                continue
            for parent_id in label_map[row.label_id].parent_ids:
                parent = decisions.get((*key[:3], parent_id))
                if parent is not None and parent.state == "absent":
                    raise ProjectError(
                        f"present label {row.label_id} has absent parent {parent_id}"
                    )
        for key, row in decisions.items():
            parents = label_map[row.label_id].parent_ids
            if len(parents) < 2 or row.state == "unreviewed":
                continue
            parent_states = [decisions.get((*key[:3], parent)) for parent in parents]
            if any(parent is None for parent in parent_states):
                continue
            states = [parent.state for parent in parent_states if parent is not None]
            derived = (
                "absent"
                if "absent" in states
                else "present"
                if all(state == "present" for state in states)
                else "unreviewed"
            )
            if row.state != derived:
                raise ProjectError(
                    f"multiple-parent label {row.label_id} contradicts its parents"
                )

    def put_memberships(
        self, expected_revision: str, rows: list[DecisionRow]
    ) -> MembershipDocument:
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
            self._atomic_output_json(self.draft_path, stored)
            return MembershipDocument(
                project_id=self.spec.id,
                revision=revision,
                source="draft",
                rows=rows,
            )

    def _materialize(self, rows: list[DecisionRow]) -> list[DecisionRow]:
        labels = {label.id: label for label in self.current_labels().labels}
        materialized = {
            (r.support_id, r.observation_id, r.entity_id, r.label_id): r for r in rows
        }
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
                            raise ProjectError(
                                f"present label {row.label_id} has absent parent {parent_id}"
                            )
                        continue
                    materialized[parent_key] = DecisionRow(
                        support_id=row.support_id,
                        observation_id=row.observation_id,
                        entity_id=row.entity_id,
                        label_id=parent_id,
                        state="present",
                        decision_view_id=row.decision_view_id,
                        provenance="derived_ancestor",
                        selection_id=row.selection_id,
                        zoom_id=row.zoom_id,
                    )
                    changed = True
            for label in labels.values():
                if len(label.parent_ids) < 2:
                    continue
                domains = {key[:3] for key in materialized}
                for domain in domains:
                    parent_rows = [
                        materialized.get((*domain, parent))
                        for parent in label.parent_ids
                    ]
                    states = [
                        item.state if item else "unreviewed" for item in parent_rows
                    ]
                    state = (
                        "absent"
                        if "absent" in states
                        else "present"
                        if all(x == "present" for x in states)
                        else "unreviewed"
                    )
                    key = (*domain, label.id)
                    existing = materialized.get(key)
                    if (
                        existing is not None
                        and existing.state != "unreviewed"
                        and existing.state != state
                    ):
                        raise ProjectError(
                            f"multiple-parent label {label.id} contradicts its parents"
                        )
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
                            selection_id=source.selection_id,
                            zoom_id=source.zoom_id,
                        )
                        changed = True
        return sorted(
            materialized.values(),
            key=lambda row: (
                row.support_id,
                row.observation_id,
                row.entity_id,
                row.label_id,
            ),
        )

    def hierarchy_summary(self) -> HierarchySummary:
        current = self.current_memberships()
        materialized = self._materialize(current.rows)
        return HierarchySummary(
            explicit_decisions=len(current.rows),
            materialized_decisions=len(materialized),
            supported_observations=len({row.observation_id for row in current.rows}),
            entities=len({(row.observation_id, row.entity_id) for row in current.rows}),
            derived_ancestors=sum(
                row.provenance == "derived_ancestor" for row in materialized
            ),
            derived_intersections=sum(
                row.provenance == "derived_intersection" for row in materialized
            ),
        )

    def export(self, expected_revision: str) -> ExportResponse:
        with self.writer_lock():
            current = self.current_memberships()
            if current.revision != expected_revision:
                raise RevisionConflict("membership revision is stale")
            self.validate_rows(current.rows)
            materialized = [
                row
                for row in self._materialize(current.rows)
                if row.state != "unreviewed"
            ]
            membership_records = [
                {
                    "support_id": row.support_id,
                    "observation_id": row.observation_id,
                    "entity_id": row.entity_id,
                    "label_id": row.label_id,
                    "state": row.state,
                    "provenance": row.provenance,
                    "decision_view_id": row.decision_view_id,
                    "selection_id": row.selection_id,
                    "zoom_id": row.zoom_id,
                }
                for row in materialized
            ]
            support_map: dict[tuple[str, str, str], dict[str, str | None]] = {}
            for row in materialized:
                key = (row.support_id, row.observation_id, row.label_id)
                item = support_map.setdefault(
                    key,
                    {
                        "accepted_view_id": row.decision_view_id,
                        "selection_id": None,
                        "zoom_id": None,
                    },
                )
                if row.selection_id is not None:
                    item["selection_id"] = row.selection_id
                if row.zoom_id is not None:
                    item["zoom_id"] = row.zoom_id
            support_records = [
                {
                    "support_id": key[0],
                    "observation_id": key[1],
                    "label_id": key[2],
                    "selection_id": item["selection_id"] or key[0],
                    "accepted_view_id": item["accepted_view_id"],
                    "zoom_id": item["zoom_id"],
                }
                for key, item in sorted(support_map.items())
            ]
            self._write_export(current.revision, membership_records, support_records)
            return ExportResponse(
                revision=current.revision,
                report_url=f"/api/v1/annotations/{self.spec.id}/export/report",
                membership_rows=len(membership_records),
                support_rows=len(support_records),
            )

    def _write_export(
        self,
        revision: str,
        memberships: list[dict[str, Any]],
        support: list[dict[str, Any]],
    ) -> None:
        output = self.spec.output_path
        self._mkdir_output(output)
        stage = Path(tempfile.mkdtemp(prefix=".export-", dir=output))
        self._chmod_output_directory(stage)
        membership_path = stage / "memberships.parquet"
        support_path = stage / "support.parquet"
        report_path = stage / "export_report.json"
        membership_schema = pa.schema(
            [
                ("support_id", pa.string()),
                ("observation_id", pa.string()),
                ("entity_id", pa.string()),
                ("label_id", pa.string()),
                ("state", pa.string()),
                ("provenance", pa.string()),
                ("decision_view_id", pa.string()),
                ("selection_id", pa.string()),
                ("zoom_id", pa.string()),
            ]
        )
        support_schema = pa.schema(
            [
                ("support_id", pa.string()),
                ("observation_id", pa.string()),
                ("label_id", pa.string()),
                ("selection_id", pa.string()),
                ("accepted_view_id", pa.string()),
                ("zoom_id", pa.string()),
            ]
        )
        try:
            self._prepare_output_file(membership_path)
            self._prepare_output_file(support_path)
            pq.write_table(
                pa.Table.from_pylist(memberships, schema=membership_schema),
                membership_path,
                compression="zstd",
            )
            pq.write_table(
                pa.Table.from_pylist(support, schema=support_schema),
                support_path,
                compression="zstd",
            )
            self._chmod_output_file(membership_path)
            self._chmod_output_file(support_path)
            report = {
                "schema_version": 1,
                "project_id": self.spec.id,
                "project_state_sha256": canonical_sha256(
                    {
                        "manifest": self.identities.manifest_sha256,
                        "labels": self.identities.labels_semantic_sha256,
                        "memberships": revision,
                    }
                ),
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
                    "entities": len(
                        {
                            (row["observation_id"], row["entity_id"])
                            for row in memberships
                        }
                    ),
                    "labels": len({row["label_id"] for row in memberships}),
                    "supports": len({row["support_id"] for row in memberships}),
                    "present": sum(row["state"] == "present" for row in memberships),
                    "absent": sum(row["state"] == "absent" for row in memberships),
                },
                "validation": {"valid": True, "errors": []},
            }
            self._atomic_output_json(report_path, report)
            targets = [
                output / "memberships.parquet",
                output / "support.parquet",
                output / "export_report.json",
            ]
            staged = [membership_path, support_path, report_path]
            backups: list[tuple[Path, Path]] = []
            promoted: list[Path] = []
            try:
                for target in targets:
                    if target.exists():
                        self._chmod_output_file(target)
                        backup = stage / f"previous-{target.name}"
                        os.replace(target, backup)
                        backups.append((target, backup))
                for source, target in zip(staged, targets, strict=True):
                    os.replace(source, target)
                    self._chmod_output_file(target)
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
