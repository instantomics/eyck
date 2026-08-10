"""Declarative export and validation of neutral annotated count matrices."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import sparse

PLAN_SCHEMA = "eyck.annotated-matrix-export-plan.v1"
FORMAT_SCHEMA = "eyck.annotated_matrix.v1"
_MEMBERS = (
    "features.parquet",
    "observations.parquet",
    "counts/data.npy",
    "counts/indices.npy",
    "counts/indptr.npy",
    "counts/shape.npy",
)
_ARRAY_CHUNK_ELEMENTS = 1024 * 1024
_LOOP_CHECK_ITEMS = 4096
_INT32_MAX = int(np.iinfo(np.int32).max)
_Scalar = str | int | float | bool


class AnnotatedMatrixError(ValueError):
    """Raised when an export plan or annotated matrix is invalid."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ExactPredicate(_StrictModel):
    """An exact equality or inclusion predicate over one observation column."""

    column: str
    equals: _Scalar | None = None
    include: tuple[_Scalar, ...] | None = None

    @model_validator(mode="after")
    def _one_operation(self) -> ExactPredicate:
        if (self.equals is None) == (self.include is None):
            raise ValueError(
                "a predicate must declare exactly one of equals or include"
            )
        if self.include is not None and not self.include:
            raise ValueError("predicate include must not be empty")
        return self


class WideMembershipAssertion(_StrictModel):
    """A wide table against which resolved source labels are asserted."""

    path: Path
    observation_column: str | None = None
    observation_index: bool | str | None = None
    label_columns: dict[str, str] = Field(default_factory=dict)
    identity_columns: bool = False
    present_value: _Scalar = 1

    @model_validator(mode="after")
    def _valid_reference(self) -> WideMembershipAssertion:
        _exactly_one_reference(
            "observation", self.observation_column, self.observation_index
        )
        _validate_strings(tuple(self.label_columns), "wide assertion source label")
        _validate_strings(tuple(self.label_columns.values()), "wide assertion column")
        if not self.identity_columns and not self.label_columns:
            raise ValueError(
                "wide assertion requires label_columns or identity_columns=true"
            )
        if len(set(self.label_columns.values())) != len(self.label_columns):
            raise ValueError("wide assertion columns must be unique")
        return self


class ObservationColumnTruth(_StrictModel):
    """Source-label truth encoded in one H5AD observation column."""

    kind: Literal["obs_column"]
    column: str
    decisions: dict[str, str] = Field(default_factory=dict)
    identity_labels: tuple[str, ...] = ()
    ignored_labels: tuple[str, ...] = ()
    context_labels: tuple[str, ...] = ()
    wide_assertion: WideMembershipAssertion | None = None

    @model_validator(mode="after")
    def _disjoint(self) -> ObservationColumnTruth:
        _validate_label_decisions(
            self.decisions,
            self.identity_labels,
            self.ignored_labels,
            self.context_labels,
        )
        return self


class MembershipTable(_StrictModel):
    """One declaratively described long entity-membership table."""

    path: Path
    observation_column: str | None = None
    observation_index: bool | str | None = None
    entity_column: str | None = None
    entity_index: bool | str | None = None
    label_column: str | None = None
    label_index: bool | str | None = None
    value_column: str | None = None
    value_index: bool | str | None = None
    present_value: _Scalar = 1
    decisions: dict[str, str] = Field(default_factory=dict)
    identity_labels: tuple[str, ...] = ()
    ignored_labels: tuple[str, ...] = ()
    context_labels: tuple[str, ...] = ()
    priorities: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_table(self) -> MembershipTable:
        _exactly_one_reference(
            "observation", self.observation_column, self.observation_index
        )
        _exactly_one_reference("entity", self.entity_column, self.entity_index)
        _exactly_one_reference("label", self.label_column, self.label_index)
        _exactly_one_reference("value", self.value_column, self.value_index)
        _validate_label_decisions(
            self.decisions,
            self.identity_labels,
            self.ignored_labels,
            self.context_labels,
        )
        known = (
            set(self.decisions)
            | set(self.identity_labels)
            | set(self.ignored_labels)
            | set(self.context_labels)
        )
        unknown = set(self.priorities) - known
        if unknown:
            raise ValueError(f"priorities contain undeclared labels: {sorted(unknown)}")
        if any(priority < 0 for priority in self.priorities.values()):
            raise ValueError("membership priorities must be non-negative")
        return self


class MembershipTruth(_StrictModel):
    """Source-label truth resolved from one or more membership tables."""

    kind: Literal["memberships"]
    mode: Literal["combined_exactly_one", "per_table_exactly_one_union"]
    tables: tuple[MembershipTable, ...]
    wide_assertion: WideMembershipAssertion | None = None

    @model_validator(mode="after")
    def _has_tables(self) -> MembershipTruth:
        if not self.tables:
            raise ValueError("membership truth requires at least one table")
        return self


Truth = Annotated[
    ObservationColumnTruth | MembershipTruth,
    Field(discriminator="kind"),
]


class ExportExpectations(_StrictModel):
    """Optional exact assertions over materialization outcomes."""

    selected: int | None = None
    rejected: dict[str, int] | None = None
    source_labels: dict[str, int] | None = None

    @model_validator(mode="after")
    def _non_negative(self) -> ExportExpectations:
        values = (
            ([] if self.selected is None else [self.selected])
            + list((self.rejected or {}).values())
            + list((self.source_labels or {}).values())
        )
        if any(value < 0 for value in values):
            raise ValueError("expectation counts must be non-negative")
        return self


class ExportSpec(_StrictModel):
    """One annotated matrix export declared by an export plan."""

    id: str
    feature_symbols: tuple[str, ...] | None = None
    predicates: tuple[ExactPredicate, ...] = ()
    truth: Truth | None = None
    expectations: ExportExpectations | None = None

    @model_validator(mode="after")
    def _valid_export(self) -> ExportSpec:
        _safe_id(self.id)
        if self.feature_symbols is not None:
            if not self.feature_symbols:
                raise ValueError("feature_symbols must not be empty when specified")
            _validate_strings(self.feature_symbols, "feature symbol")
            if len(set(self.feature_symbols)) != len(self.feature_symbols):
                raise ValueError("feature_symbols must be unique")
        return self


class ExportPlan(_StrictModel):
    """A strict, task-neutral annotated matrix export plan."""

    schema_id: Literal[PLAN_SCHEMA] = Field(default=PLAN_SCHEMA, alias="schema")
    h5ad: Path
    matrix: str = "X"
    feature_id_column: str | None = None
    chunk_rows: int = 1024
    truth: Truth | None = None
    exports: tuple[ExportSpec, ...]

    @model_validator(mode="after")
    def _valid_plan(self) -> ExportPlan:
        if not self.matrix:
            raise ValueError("matrix must be X or a non-empty layer name")
        if self.chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        if not self.exports:
            raise ValueError("an export plan requires at least one export")
        ids = [export.id for export in self.exports]
        if len(set(ids)) != len(ids):
            raise ValueError("export IDs must be unique")
        export_truths = [export.truth is not None for export in self.exports]
        if self.truth is None and not all(export_truths):
            raise ValueError("each export requires an export-level or plan-level truth")
        return self


@dataclass(frozen=True)
class AnnotatedMatrix:
    """An opened and validated annotated count matrix."""

    root: Path
    manifest: Mapping[str, Any]
    feature_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    source_label_ids: tuple[str, ...]
    counts: sparse.csr_matrix


def load_export_plan(path: str | Path) -> ExportPlan:
    """Load a strict TOML export plan and resolve its paths relative to the TOML."""
    plan_path = Path(path).expanduser().resolve(strict=True)
    if plan_path.is_symlink() or not plan_path.is_file():
        raise AnnotatedMatrixError(f"export plan is not a regular file: {plan_path}")
    try:
        raw = tomllib.loads(plan_path.read_text(encoding="utf-8"))
        # JSON-mode validation retains strict scalar validation while accepting
        # TOML's path strings and arrays for immutable Path/tuple fields.
        plan = ExportPlan.model_validate_json(json.dumps(raw))
    except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
        raise AnnotatedMatrixError(
            f"invalid export plan {plan_path}: {error}"
        ) from error
    base = plan_path.parent
    exports: list[ExportSpec] = []
    for export in plan.exports:
        resolved_truth = (
            None if export.truth is None else _resolve_truth_paths(export.truth, base)
        )
        exports.append(export.model_copy(update={"truth": resolved_truth}))
    return plan.model_copy(
        update={
            "h5ad": _resolve_declared_path(base, plan.h5ad),
            "truth": None
            if plan.truth is None
            else _resolve_truth_paths(plan.truth, base),
            "exports": tuple(exports),
        }
    )


def materialize_exports(
    plan: ExportPlan,
    destination: str | Path,
    timeout_seconds: float = 300.0,
    *,
    replace: bool = False,
) -> tuple[Path, ...]:
    """Materialize every export atomically, returning its output directory."""
    if not isinstance(plan, ExportPlan):
        raise TypeError("plan must be an ExportPlan")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise AnnotatedMatrixError("timeout_seconds must be finite and positive")
    deadline = time.monotonic() + timeout_seconds
    destination_path = Path(destination).expanduser().absolute()
    _check_timeout(deadline)
    _prepare_destination(destination_path)
    _require_regular_input(plan.h5ad, "H5AD")

    outputs: list[Path] = []
    try:
        data = ad.read_h5ad(plan.h5ad, backed="r")
    except Exception as error:
        raise AnnotatedMatrixError(f"could not open H5AD read-only: {error}") from error
    try:
        observation_ids = _axis_strings(
            data.obs_names, "observation ID", deadline=deadline
        )
        feature_values = (
            data.var_names
            if plan.feature_id_column is None
            else _required_column(data.var, plan.feature_id_column)
        )
        source_features = _axis_strings(
            feature_values,
            "source feature symbol",
            unique=False,
            deadline=deadline,
        )
        matrix = data.X if plan.matrix == "X" else _required_layer(data, plan.matrix)
        if matrix is None or matrix.shape != (
            len(observation_ids),
            len(source_features),
        ):
            raise AnnotatedMatrixError("selected H5AD matrix has an invalid shape")
        _check_timeout(deadline)
        for export in plan.exports:
            outputs.append(
                _materialize_one(
                    plan,
                    export,
                    data,
                    matrix,
                    observation_ids,
                    source_features,
                    destination_path,
                    deadline,
                    replace,
                )
            )
    finally:
        data.file.close()
    return tuple(outputs)


def validate_annotated_matrix(root: str | Path, deep: bool = True) -> Mapping[str, Any]:
    """Validate an annotated matrix directory and return its manifest."""
    return _validate_annotated_matrix(root, deep, deadline=None)


def _validate_annotated_matrix(
    root: str | Path, deep: bool, deadline: float | None
) -> Mapping[str, Any]:
    root_path = Path(root).expanduser().absolute()
    _check_timeout(deadline)
    _validate_tree(root_path, deadline)
    try:
        manifest = json.loads((root_path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnnotatedMatrixError(f"invalid manifest.json: {error}") from error
    if not isinstance(manifest, dict):
        raise AnnotatedMatrixError("manifest.json must contain an object")
    required = {
        "schema",
        "format",
        "export_id",
        "members",
        "semantic_hashes",
        "counts",
        "validation",
    }
    if set(manifest) != required:
        raise AnnotatedMatrixError(
            "manifest fields do not match the annotated matrix schema"
        )
    if manifest["schema"] != FORMAT_SCHEMA or manifest["format"] != "csr_int32":
        raise AnnotatedMatrixError("unsupported annotated matrix schema or format")
    _safe_id(manifest["export_id"])
    members = manifest["members"]
    if not isinstance(members, dict) or set(members) != set(_MEMBERS):
        raise AnnotatedMatrixError("manifest member set is invalid")
    for member_index, name in enumerate(_MEMBERS):
        _check_loop_timeout(deadline, member_index)
        descriptor = members[name]
        if not isinstance(descriptor, dict) or set(descriptor) != {"sha256", "size"}:
            raise AnnotatedMatrixError(f"invalid member descriptor: {name}")
        if not isinstance(descriptor["sha256"], str) or len(descriptor["sha256"]) != 64:
            raise AnnotatedMatrixError(f"invalid member SHA256: {name}")
        if type(descriptor["size"]) is not int or descriptor["size"] < 0:
            raise AnnotatedMatrixError(f"invalid member size: {name}")
        path = root_path / name
        if path.stat().st_size != descriptor["size"]:
            raise AnnotatedMatrixError(f"member size mismatch: {name}")
        if deep and _file_sha256(path, deadline) != descriptor["sha256"]:
            raise AnnotatedMatrixError(f"member SHA256 mismatch: {name}")
    semantic = manifest["semantic_hashes"]
    if not isinstance(semantic, dict) or set(semantic) != {
        "feature_axis",
        "observation_axis",
        "csr",
        "plan_selection",
    }:
        raise AnnotatedMatrixError("semantic hash set is invalid")
    if any(
        not isinstance(value, str) or len(value) != 64 for value in semantic.values()
    ):
        raise AnnotatedMatrixError("semantic hashes must be SHA256 strings")
    if manifest["validation"] != {"status": "valid", "mode": "deep"}:
        raise AnnotatedMatrixError("manifest validation status is invalid")
    features, observations, arrays = _read_members(root_path, deadline)
    _validate_loaded_members(features, observations, arrays, deadline)
    feature_ids = tuple(features["feature_id"].to_pylist())
    observation_ids = tuple(observations["observation_id"].to_pylist())
    source_labels = tuple(observations["source_label_id"].to_pylist())
    selected = len(observation_ids)
    count_summary = manifest["counts"]
    expected_labels = dict(sorted(_count_strings(source_labels, deadline).items()))
    if not isinstance(count_summary, dict) or set(count_summary) != {
        "selected",
        "rejected",
        "source_labels",
    }:
        raise AnnotatedMatrixError("count summary fields are invalid")
    if count_summary["selected"] != selected:
        raise AnnotatedMatrixError("selected count does not match members")
    if count_summary["source_labels"] != expected_labels:
        raise AnnotatedMatrixError("source-label counts do not match members")
    rejected = count_summary["rejected"]
    if not isinstance(rejected, dict) or any(
        type(value) is not int or value < 0 for value in rejected.values()
    ):
        raise AnnotatedMatrixError("rejected count summary is invalid")
    if deep:
        if semantic["feature_axis"] != _ordered_axis_hash(feature_ids, deadline):
            raise AnnotatedMatrixError("feature axis semantic hash mismatch")
        if semantic["observation_axis"] != _ordered_pair_hash(
            observation_ids, source_labels, deadline
        ):
            raise AnnotatedMatrixError("observation axis semantic hash mismatch")
        if semantic["csr"] != _csr_hash(*arrays, deadline=deadline):
            raise AnnotatedMatrixError("CSR semantic hash mismatch")
    _check_timeout(deadline)
    return manifest


def open_annotated_matrix(
    root: str | Path, validate: Literal["deep", "shallow"] = "deep"
) -> AnnotatedMatrix:
    """Open an annotated matrix after deep or shallow validation."""
    if validate not in {"deep", "shallow"}:
        raise AnnotatedMatrixError("validate must be 'deep' or 'shallow'")
    root_path = Path(root).expanduser().absolute()
    manifest = validate_annotated_matrix(root_path, deep=validate == "deep")
    features, observations, arrays = _read_members(root_path)
    data_values, indices, indptr, shape = arrays
    counts = sparse.csr_matrix(
        (data_values, indices, indptr),
        shape=tuple(int(value) for value in shape),
        copy=False,
    )
    return AnnotatedMatrix(
        root=root_path,
        manifest=manifest,
        feature_ids=tuple(features["feature_id"].to_pylist()),
        observation_ids=tuple(observations["observation_id"].to_pylist()),
        source_label_ids=tuple(observations["source_label_id"].to_pylist()),
        counts=counts,
    )


def _materialize_one(
    plan: ExportPlan,
    export: ExportSpec,
    data: ad.AnnData,
    matrix: Any,
    observation_ids: tuple[str, ...],
    source_features: tuple[str, ...],
    destination: Path,
    deadline: float,
    replace: bool,
) -> Path:
    _check_timeout(deadline)
    truth = _resolved_truth(plan, export)
    selected, labels, rejected = _resolve_observations(
        export, truth, data.obs, observation_ids, deadline
    )
    selected_ids = _selected_axis_values(observation_ids, selected, deadline)
    _assert_wide(truth.wide_assertion, selected_ids, labels, deadline)
    _check_expectations(
        export.expectations,
        len(selected),
        rejected,
        _count_strings(labels, deadline),
    )
    feature_ids, columns = _resolve_feature_columns(
        export.feature_symbols, source_features, deadline
    )
    source_columns, _destinations = _flatten_feature_columns(columns, deadline)
    _require_int32_csr_layout(len(selected), len(feature_ids), source_columns, deadline)
    _check_timeout(deadline)

    temporary = Path(tempfile.mkdtemp(prefix=f".{export.id}.", dir=destination))
    try:
        _write_export(
            temporary,
            plan,
            export,
            truth,
            feature_ids,
            selected_ids,
            labels,
            rejected,
            matrix,
            selected,
            columns,
            plan.chunk_rows,
            deadline,
        )
        _check_timeout(deadline)
        _validate_annotated_matrix(temporary, deep=True, deadline=deadline)
        _check_timeout(deadline)
        target = destination / export.id
        _publish(temporary, target, replace, deadline)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _resolve_observations(
    export: ExportSpec,
    truth: Truth,
    obs: pd.DataFrame,
    observation_ids: tuple[str, ...],
    deadline: float,
) -> tuple[list[int], tuple[str, ...], dict[str, int]]:
    eligible = np.ones(len(observation_ids), dtype=bool)
    for predicate_index, predicate in enumerate(export.predicates):
        _check_loop_timeout(deadline, predicate_index)
        values = _required_column(obs, predicate.column)
        if predicate.include is not None:
            eligible &= values.isin(predicate.include).to_numpy(dtype=bool)
        else:
            eligible &= values.eq(predicate.equals).fillna(False).to_numpy(dtype=bool)
        _check_timeout(deadline)
    rejected: Counter[str] = Counter()
    predicate_mismatches = int(np.count_nonzero(~eligible))
    if predicate_mismatches:
        rejected["predicate_mismatch"] = predicate_mismatches
    selected: list[int] = []
    labels: list[str] = []
    if isinstance(truth, ObservationColumnTruth):
        values = _required_column(obs, truth.column)
        decisions = _effective_decisions(truth.decisions, truth.identity_labels)
        for index, _observation_id in enumerate(observation_ids):
            _check_loop_timeout(deadline, index)
            if not eligible[index]:
                continue
            raw = values.iloc[index]
            if pd.isna(raw):
                rejected["missing_label"] += 1
                continue
            raw_label = _string(raw, "observation truth label")
            reason, source_label = _decide_label(
                raw_label,
                decisions,
                truth.ignored_labels,
                truth.context_labels,
            )
            if reason is not None:
                rejected[reason] += 1
            else:
                assert source_label is not None
                selected.append(index)
                labels.append(source_label)
    else:
        table_rows = []
        for table_index, table in enumerate(truth.tables):
            _check_loop_timeout(deadline, table_index)
            table_rows.append(_membership_rows(table, deadline))
        _check_timeout(deadline)
        for index, observation_id in enumerate(observation_ids):
            _check_loop_timeout(deadline, index)
            if not eligible[index]:
                continue
            reason, source_label = _resolve_membership_observation(
                observation_id, truth, table_rows, deadline
            )
            if reason is not None:
                rejected[reason] += 1
            else:
                assert source_label is not None
                selected.append(index)
                labels.append(source_label)
    _check_timeout(deadline)
    return selected, tuple(labels), dict(sorted(rejected.items()))


def _membership_rows(
    table: MembershipTable, deadline: float
) -> dict[str, list[tuple[str, str, int]]]:
    _check_timeout(deadline)
    _require_regular_input(table.path, "membership parquet")
    try:
        frame = pd.read_parquet(table.path)
    except Exception as error:
        raise AnnotatedMatrixError(
            f"could not read membership table {table.path}: {error}"
        ) from error
    _check_timeout(deadline)
    observations = _referenced_values(
        frame, table.observation_column, table.observation_index, "observation"
    )
    entities = _referenced_values(
        frame, table.entity_column, table.entity_index, "entity"
    )
    labels = _referenced_values(frame, table.label_column, table.label_index, "label")
    values = _referenced_values(frame, table.value_column, table.value_index, "value")
    decisions = _effective_decisions(table.decisions, table.identity_labels)
    rows: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for row_index, (observation, entity, raw_label, value) in enumerate(
        zip(observations, entities, labels, values, strict=True)
    ):
        _check_loop_timeout(deadline, row_index)
        if not _exact_equal(value, table.present_value):
            continue
        observation_id = _string(observation, "membership observation")
        entity_id = _string(entity, "membership entity")
        label = _string(raw_label, "membership label")
        priority = table.priorities.get(label, 0)
        if label in decisions:
            decision = decisions[label]
        elif label in table.ignored_labels:
            decision = "\0ignored"
        elif label in table.context_labels:
            decision = "\0context"
        else:
            decision = "\0unmapped"
        rows[observation_id].append((entity_id, decision, priority))
    _check_timeout(deadline)
    return rows


def _resolve_membership_observation(
    observation_id: str,
    truth: MembershipTruth,
    tables: Sequence[dict[str, list[tuple[str, str, int]]]],
    deadline: float,
) -> tuple[str | None, str | None]:
    groups: list[Iterable[tuple[str, str, int]]]
    if truth.mode == "combined_exactly_one":
        groups = [_combined_membership_rows(tables, observation_id, deadline)]
    else:
        groups = []
        for table_index, table in enumerate(tables):
            _check_loop_timeout(deadline, table_index)
            if observation_id in table:
                groups.append(table[observation_id])
    if not groups:
        return "zero_entities", None
    resolved: list[tuple[str, int]] = []
    for group_index, rows in enumerate(groups):
        _check_loop_timeout(deadline, group_index)
        entities: set[str] = set()
        maximum: int | None = None
        maximum_labels: set[str] = set()
        has_context = False
        has_unmapped = False
        for row_index, (entity, decision, priority) in enumerate(rows):
            _check_loop_timeout(deadline, row_index)
            entities.add(entity)
            if decision == "\0unmapped":
                has_unmapped = True
            elif decision == "\0context":
                has_context = True
            elif decision != "\0ignored":
                if maximum is None or priority > maximum:
                    maximum = priority
                    maximum_labels = {decision}
                elif priority == maximum:
                    maximum_labels.add(decision)
        if not entities:
            return "zero_entities", None
        if len(entities) > 1:
            return "multiple_entities", None
        if has_unmapped:
            return "unmapped_label", None
        if maximum is None:
            if has_context:
                return "context_label", None
            return "ignored_label", None
        if len(maximum_labels) != 1:
            return "conflicting_labels", None
        resolved.append((maximum_labels.pop(), maximum))
    labels = set()
    for resolved_index, (label, _priority) in enumerate(resolved):
        _check_loop_timeout(deadline, resolved_index)
        labels.add(label)
    if len(labels) != 1:
        return "conflicting_labels", None
    return None, labels.pop()


def _combined_membership_rows(
    tables: Sequence[dict[str, list[tuple[str, str, int]]]],
    observation_id: str,
    deadline: float,
) -> Iterable[tuple[str, str, int]]:
    item_index = 0
    for table_index, table in enumerate(tables):
        _check_loop_timeout(deadline, table_index)
        for row in table.get(observation_id, ()):
            _check_loop_timeout(deadline, item_index)
            item_index += 1
            yield row


def _resolve_feature_columns(
    requested: tuple[str, ...] | None,
    source_features: tuple[str, ...],
    deadline: float,
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, symbol in enumerate(source_features):
        _check_loop_timeout(deadline, index)
        positions[symbol].append(index)
    output_axis = tuple(sorted(positions)) if requested is None else requested
    missing = []
    for index, symbol in enumerate(output_axis):
        _check_loop_timeout(deadline, index)
        if symbol not in positions:
            missing.append(symbol)
    if missing:
        raise AnnotatedMatrixError(
            f"feature symbols are missing with exact case: {missing}"
        )
    _check_timeout(deadline)
    resolved = []
    for index, symbol in enumerate(output_axis):
        _check_loop_timeout(deadline, index)
        resolved.append(tuple(positions[symbol]))
    return output_axis, tuple(resolved)


def _flatten_feature_columns(
    columns: Sequence[Sequence[int]], deadline: float
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    source_columns = []
    destinations = []
    item_index = 0
    for destination, group in enumerate(columns):
        for column in group:
            _check_loop_timeout(deadline, item_index)
            source_columns.append(column)
            destinations.append(destination)
            item_index += 1
    _check_timeout(deadline)
    return tuple(source_columns), tuple(destinations)


def _iter_count_chunks(
    matrix: Any,
    rows: list[int],
    columns: tuple[tuple[int, ...], ...],
    output_columns: int,
    chunk_rows: int,
    deadline: float,
) -> Iterable[sparse.csr_matrix]:
    source_columns, destinations = _flatten_feature_columns(columns, deadline)
    projection = sparse.csr_matrix(
        (
            np.ones(len(source_columns), dtype=np.int64),
            (np.arange(len(source_columns)), np.asarray(destinations)),
        ),
        shape=(len(source_columns), output_columns),
    )
    offset = 0
    while offset < len(rows):
        _check_timeout(deadline)
        start = rows[offset]
        stop_offset = offset + 1
        while (
            stop_offset < len(rows)
            and rows[stop_offset] == rows[stop_offset - 1] + 1
            and stop_offset - offset < chunk_rows
        ):
            _check_loop_timeout(deadline, stop_offset - offset)
            stop_offset += 1
        stop = rows[stop_offset - 1] + 1
        raw = matrix[start:stop, :]
        _check_timeout(deadline)
        if not sparse.issparse(raw):
            raw = sparse.csr_matrix(np.asarray(raw))
        else:
            raw = raw.tocsr(copy=True)
        selected_values = raw[:, source_columns]
        _validate_numeric_values(selected_values.data, deadline)
        selected_values.data = selected_values.data.astype(np.int64)
        combined = (selected_values @ projection).tocsr()
        _check_timeout(deadline)
        combined.sum_duplicates()
        combined.eliminate_zeros()
        combined.sort_indices()
        if combined.data.size and int(combined.data.max()) > _INT32_MAX:
            raise AnnotatedMatrixError("summed counts exceed int32 range")
        _check_timeout(deadline)
        yield combined
        offset = stop_offset


def _write_counts(
    root: Path,
    matrix: Any,
    rows: list[int],
    columns: tuple[tuple[int, ...], ...],
    output_columns: int,
    chunk_rows: int,
    deadline: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_columns, _destinations = _flatten_feature_columns(columns, deadline)
    _require_int32_csr_layout(len(rows), output_columns, source_columns, deadline)
    mapped: list[np.memmap] = []
    try:
        indptr = np.lib.format.open_memmap(
            root / "indptr.npy",
            mode="w+",
            dtype=np.int32,
            shape=(len(rows) + 1,),
        )
        mapped.append(indptr)
        indptr[0] = 0
        row_offset = 0
        nnz = 0
        for chunk_index, combined in enumerate(
            _iter_count_chunks(
                matrix, rows, columns, output_columns, chunk_rows, deadline
            )
        ):
            _check_loop_timeout(deadline, chunk_index)
            next_nnz = _checked_int32_nnz(nnz, int(combined.nnz))
            next_row_offset = row_offset + combined.shape[0]
            pointers = np.asarray(combined.indptr[1:], dtype=np.int64) + nnz
            indptr[row_offset + 1 : next_row_offset + 1] = pointers
            row_offset = next_row_offset
            nnz = next_nnz
        if row_offset != len(rows):
            raise AnnotatedMatrixError("streamed CSR row count is inconsistent")
        indptr.flush()
        _check_timeout(deadline)

        data_values = np.lib.format.open_memmap(
            root / "data.npy", mode="w+", dtype=np.int32, shape=(nnz,)
        )
        mapped.append(data_values)
        indices = np.lib.format.open_memmap(
            root / "indices.npy", mode="w+", dtype=np.int32, shape=(nnz,)
        )
        mapped.append(indices)
        write_offset = 0
        row_offset = 0
        for chunk_index, combined in enumerate(
            _iter_count_chunks(
                matrix, rows, columns, output_columns, chunk_rows, deadline
            )
        ):
            _check_loop_timeout(deadline, chunk_index)
            stop = write_offset + int(combined.nnz)
            next_row_offset = row_offset + combined.shape[0]
            if stop != int(indptr[next_row_offset]):
                raise AnnotatedMatrixError("count matrix changed while it was streamed")
            data_values[write_offset:stop] = combined.data
            indices[write_offset:stop] = combined.indices
            write_offset = stop
            row_offset = next_row_offset
            _check_timeout(deadline)
        if write_offset != nnz or row_offset != len(rows):
            raise AnnotatedMatrixError("streamed CSR lengths are inconsistent")
        data_values.flush()
        indices.flush()

        shape = np.lib.format.open_memmap(
            root / "shape.npy", mode="w+", dtype=np.int32, shape=(2,)
        )
        mapped.append(shape)
        shape[:] = (len(rows), output_columns)
        shape.flush()
        _check_timeout(deadline)
        return data_values, indices, indptr, shape
    except BaseException:
        _close_memmaps(mapped)
        raise


def _close_memmaps(arrays: Iterable[np.ndarray]) -> None:
    for array in arrays:
        if isinstance(array, np.memmap):
            array.flush()
            array._mmap.close()


def _write_export(
    root: Path,
    plan: ExportPlan,
    export: ExportSpec,
    truth: Truth,
    feature_ids: tuple[str, ...],
    observation_ids: tuple[str, ...],
    labels: tuple[str, ...],
    rejected: dict[str, int],
    matrix: Any,
    rows: list[int],
    columns: tuple[tuple[int, ...], ...],
    chunk_rows: int,
    deadline: float,
) -> None:
    _check_timeout(deadline)
    (root / "counts").mkdir()
    feature_table = pa.Table.from_arrays(
        [pa.array(feature_ids, type=pa.string())],
        schema=pa.schema([pa.field("feature_id", pa.string(), nullable=False)]),
    )
    observation_table = pa.Table.from_arrays(
        [
            pa.array(observation_ids, type=pa.string()),
            pa.array(labels, type=pa.string()),
        ],
        schema=pa.schema(
            [
                pa.field("observation_id", pa.string(), nullable=False),
                pa.field("source_label_id", pa.string(), nullable=False),
            ]
        ),
    )
    pq.write_table(
        feature_table,
        root / "features.parquet",
        compression="NONE",
        write_statistics=False,
    )
    pq.write_table(
        observation_table,
        root / "observations.parquet",
        compression="NONE",
        write_statistics=False,
    )
    _check_timeout(deadline)
    arrays = _write_counts(
        root / "counts",
        matrix,
        rows,
        columns,
        len(feature_ids),
        chunk_rows,
        deadline,
    )
    try:
        plan_selection = _semantic_selection_payload(
            plan,
            export,
            truth,
            feature_ids,
            observation_ids,
            labels,
            rejected,
            deadline,
        )
        members: dict[str, dict[str, str | int]] = {}
        for member_index, name in enumerate(_MEMBERS):
            _check_loop_timeout(deadline, member_index)
            member_path = root / name
            members[name] = {
                "sha256": _file_sha256(member_path, deadline),
                "size": member_path.stat().st_size,
            }
        manifest = {
            "schema": FORMAT_SCHEMA,
            "format": "csr_int32",
            "export_id": export.id,
            "members": members,
            "semantic_hashes": {
                "feature_axis": _ordered_axis_hash(feature_ids, deadline),
                "observation_axis": _ordered_pair_hash(
                    observation_ids, labels, deadline
                ),
                "csr": _csr_hash(*arrays, deadline=deadline),
                "plan_selection": _object_hash(plan_selection, deadline),
            },
            "counts": {
                "selected": len(observation_ids),
                "rejected": rejected,
                "source_labels": dict(sorted(_count_strings(labels, deadline).items())),
            },
            "validation": {"status": "valid", "mode": "deep"},
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        _check_timeout(deadline)
    finally:
        _close_memmaps(arrays)


def _semantic_truth_payload(
    truth: Truth, labels: Sequence[str], deadline: float
) -> dict[str, Any]:
    _check_timeout(deadline)
    payload = truth.model_dump(
        mode="json",
        exclude={"decisions", "identity_labels", "wide_assertion", "tables"},
    )
    if isinstance(truth, ObservationColumnTruth):
        payload["decisions"] = _effective_decisions(
            truth.decisions, truth.identity_labels
        )
    else:
        payload["tables"] = []
        for table_index, table in enumerate(truth.tables):
            _check_loop_timeout(deadline, table_index)
            table_payload = table.model_dump(
                mode="json",
                exclude={"path", "decisions", "identity_labels"},
            )
            table_payload["decisions"] = _effective_decisions(
                table.decisions, table.identity_labels
            )
            payload["tables"].append(table_payload)
    if truth.wide_assertion is None:
        payload["wide_assertion"] = None
    else:
        assertion_payload = truth.wide_assertion.model_dump(
            mode="json",
            exclude={"path", "label_columns", "identity_columns"},
        )
        assertion_payload["label_columns"] = _effective_assertion_columns(
            truth.wide_assertion, labels, deadline
        )
        payload["wide_assertion"] = assertion_payload
    _check_timeout(deadline)
    return payload


def _semantic_selection_payload(
    plan: ExportPlan,
    export: ExportSpec,
    truth: Truth,
    feature_ids: tuple[str, ...],
    observation_ids: tuple[str, ...],
    labels: tuple[str, ...],
    rejected: Mapping[str, int],
    deadline: float,
) -> dict[str, Any]:
    truth_payload = _semantic_truth_payload(truth, labels, deadline)
    predicates = []
    for predicate_index, predicate in enumerate(export.predicates):
        _check_loop_timeout(deadline, predicate_index)
        predicates.append(predicate.model_dump(mode="json"))
    selected = []
    for index, pair in enumerate(zip(observation_ids, labels, strict=True)):
        _check_loop_timeout(deadline, index)
        selected.append(pair)
    feature_axis = []
    for index, feature_id in enumerate(feature_ids):
        _check_loop_timeout(deadline, index)
        feature_axis.append(feature_id)
    _check_timeout(deadline)
    return {
        "export_id": export.id,
        "matrix": {
            "slot": plan.matrix,
            "feature_id_column": plan.feature_id_column,
        },
        "predicates": predicates,
        "truth": truth_payload,
        "feature_axis": feature_axis,
        "selected": selected,
        "rejected": dict(rejected),
    }


def _publish(temporary: Path, target: Path, replace: bool, deadline: float) -> None:
    _check_timeout(deadline)
    if target.is_symlink():
        raise AnnotatedMatrixError(f"output target must not be a symlink: {target}")
    if not target.exists():
        _check_timeout(deadline)
        os.replace(temporary, target)
        return
    try:
        existing = _validate_annotated_matrix(target, deep=True, deadline=deadline)
        candidate = _validate_annotated_matrix(temporary, deep=True, deadline=deadline)
    except AnnotatedMatrixError:
        if not replace:
            raise AnnotatedMatrixError(
                f"existing output is invalid or differs: {target}"
            ) from None
    else:
        if existing == candidate:
            shutil.rmtree(temporary)
            return
        if not replace:
            raise AnnotatedMatrixError(
                f"existing output differs; explicit replace is required: {target}"
            )
    backup = target.with_name(f".{target.name}.replaced-{os.getpid()}")
    if backup.exists() or backup.is_symlink():
        raise AnnotatedMatrixError(f"replacement backup already exists: {backup}")
    _check_timeout(deadline)
    os.replace(target, backup)
    try:
        os.replace(temporary, target)
    except Exception:
        os.replace(backup, target)
        raise
    if backup.is_dir():
        shutil.rmtree(backup)
    else:
        backup.unlink()


def _read_members(
    root: Path, deadline: float | None = None
) -> tuple[pa.Table, pa.Table, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    _check_timeout(deadline)
    try:
        features = pq.read_table(root / "features.parquet")
        observations = pq.read_table(root / "observations.parquet")
        arrays = tuple(
            np.load(
                root / "counts" / f"{name}.npy",
                allow_pickle=False,
                mmap_mode="r",
            )
            for name in ("data", "indices", "indptr", "shape")
        )
    except Exception as error:
        raise AnnotatedMatrixError(
            f"could not read annotated matrix members: {error}"
        ) from error
    _check_timeout(deadline)
    return features, observations, arrays  # type: ignore[return-value]


def _validate_loaded_members(
    features: pa.Table,
    observations: pa.Table,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    deadline: float | None = None,
) -> None:
    _check_timeout(deadline)
    expected_features = pa.schema([pa.field("feature_id", pa.string(), nullable=False)])
    expected_observations = pa.schema(
        [
            pa.field("observation_id", pa.string(), nullable=False),
            pa.field("source_label_id", pa.string(), nullable=False),
        ]
    )
    if (
        features.schema != expected_features
        or observations.schema != expected_observations
    ):
        raise AnnotatedMatrixError("axis parquet schema is invalid")
    feature_ids = tuple(features["feature_id"].to_pylist())
    observation_ids = tuple(observations["observation_id"].to_pylist())
    labels = tuple(observations["source_label_id"].to_pylist())
    _validate_strings(feature_ids, "feature ID", deadline)
    _validate_strings(observation_ids, "observation ID", deadline)
    _validate_strings(labels, "source label ID", deadline)
    if len(feature_ids) != len(set(feature_ids)) or len(observation_ids) != len(
        set(observation_ids)
    ):
        raise AnnotatedMatrixError("axis IDs must be unique")
    if len(observation_ids) != len(labels):
        raise AnnotatedMatrixError("observation columns have different lengths")
    data_values, indices, indptr, shape = arrays
    if any(array.dtype != np.dtype(np.int32) for array in arrays):
        raise AnnotatedMatrixError("CSR arrays must have int32 dtype")
    if (
        data_values.ndim != 1
        or indices.ndim != 1
        or indptr.ndim != 1
        or shape.shape != (2,)
    ):
        raise AnnotatedMatrixError("CSR array dimensions are invalid")
    rows, columns = (int(value) for value in shape)
    if rows != len(observation_ids) or columns != len(feature_ids):
        raise AnnotatedMatrixError("CSR shape does not match axes")
    if len(data_values) != len(indices) or len(indptr) != rows + 1:
        raise AnnotatedMatrixError("CSR array lengths are invalid")
    if (
        not len(indptr)
        or indptr[0] != 0
        or indptr[-1] != len(data_values)
        or _has_adjacent_order_violation(indptr, strict=False, deadline=deadline)
    ):
        raise AnnotatedMatrixError("CSR indptr is malformed")
    if (
        _any_chunked(data_values, lambda chunk: chunk <= 0, deadline)
        or _any_chunked(indices, lambda chunk: chunk < 0, deadline)
        or _any_chunked(indices, lambda chunk: chunk >= columns, deadline)
    ):
        raise AnnotatedMatrixError("CSR data or indices are invalid")
    if _has_csr_index_order_violation(indices, indptr, deadline):
        raise AnnotatedMatrixError("CSR indices are not canonical")
    _check_timeout(deadline)


def _array_chunks(
    array: np.ndarray, deadline: float | None = None
) -> Iterable[np.ndarray]:
    for start in range(0, len(array), _ARRAY_CHUNK_ELEMENTS):
        _check_timeout(deadline)
        yield array[start : start + _ARRAY_CHUNK_ELEMENTS]


def _any_chunked(
    array: np.ndarray, predicate: Any, deadline: float | None = None
) -> bool:
    return any(
        bool(np.any(predicate(chunk))) for chunk in _array_chunks(array, deadline)
    )


def _has_adjacent_order_violation(
    array: np.ndarray, *, strict: bool, deadline: float | None = None
) -> bool:
    if len(array) < 2:
        return False
    previous = int(array[0])
    for chunk in _array_chunks(array[1:], deadline):
        if int(chunk[0]) <= previous if strict else int(chunk[0]) < previous:
            return True
        if len(chunk) > 1:
            violation = chunk[1:] <= chunk[:-1] if strict else chunk[1:] < chunk[:-1]
            if bool(np.any(violation)):
                return True
        previous = int(chunk[-1])
    return False


def _has_csr_index_order_violation(
    indices: np.ndarray, indptr: np.ndarray, deadline: float | None
) -> bool:
    for start in range(1, len(indices), _ARRAY_CHUNK_ELEMENTS):
        _check_timeout(deadline)
        stop = min(start + _ARRAY_CHUNK_ELEMENTS, len(indices))
        positions = np.arange(start, stop, dtype=np.int64)
        row_start_locations = np.searchsorted(indptr, positions, side="left")
        row_boundaries = indptr[row_start_locations] == positions
        violations = indices[start:stop] <= indices[start - 1 : stop - 1]
        if bool(np.any(violations & ~row_boundaries)):
            return True
    return False


def _validate_tree(root: Path, deadline: float | None = None) -> None:
    _check_timeout(deadline)
    if root.is_symlink() or not root.is_dir():
        raise AnnotatedMatrixError(
            f"annotated matrix root must be a real directory: {root}"
        )
    if {path.name for path in root.iterdir()} != {
        "manifest.json",
        "features.parquet",
        "observations.parquet",
        "counts",
    }:
        raise AnnotatedMatrixError(
            "annotated matrix has missing or extra top-level members"
        )
    counts = root / "counts"
    if counts.is_symlink() or not counts.is_dir():
        raise AnnotatedMatrixError("counts must be a real directory")
    if {path.name for path in counts.iterdir()} != {
        "data.npy",
        "indices.npy",
        "indptr.npy",
        "shape.npy",
    }:
        raise AnnotatedMatrixError("counts has missing or extra members")
    for name in ("manifest.json", *_MEMBERS):
        _check_timeout(deadline)
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise AnnotatedMatrixError(
                f"member must be a regular non-symlink file: {name}"
            )


def _assert_wide(
    assertion: WideMembershipAssertion | None,
    observation_ids: tuple[str, ...],
    labels: tuple[str, ...],
    deadline: float,
) -> None:
    if assertion is None:
        return
    _check_timeout(deadline)
    _require_regular_input(assertion.path, "wide membership parquet")
    frame = pd.read_parquet(assertion.path)
    observations = _referenced_values(
        frame, assertion.observation_column, assertion.observation_index, "observation"
    )
    positions: dict[str, int] = {}
    for index, value in enumerate(observations):
        _check_loop_timeout(deadline, index)
        observation_id = _string(value, "wide assertion observation")
        if observation_id in positions:
            raise AnnotatedMatrixError(
                f"wide assertion has duplicate observation: {observation_id}"
            )
        positions[observation_id] = index
    columns = _effective_assertion_columns(assertion, labels, deadline)
    column_values = {
        column: _required_column(frame, column) for column in set(columns.values())
    }
    for index, (observation_id, label) in enumerate(
        zip(observation_ids, labels, strict=True)
    ):
        _check_loop_timeout(deadline, index)
        if observation_id not in positions:
            raise AnnotatedMatrixError(
                f"wide assertion is missing observation: {observation_id}"
            )
        values = column_values[columns[label]]
        if not _exact_equal(
            values.iloc[positions[observation_id]], assertion.present_value
        ):
            raise AnnotatedMatrixError(
                f"wide assertion disagrees for observation: {observation_id}"
            )
    _check_timeout(deadline)


def _effective_assertion_columns(
    assertion: WideMembershipAssertion,
    labels: Sequence[str],
    deadline: float | None = None,
) -> dict[str, str]:
    columns: dict[str, str] = {}
    for index, label in enumerate(sorted(set(labels))):
        _check_loop_timeout(deadline, index)
        if label in assertion.label_columns:
            columns[label] = assertion.label_columns[label]
        elif assertion.identity_columns:
            columns[label] = label
        else:
            raise AnnotatedMatrixError(
                f"wide assertion has no column for source label: {label}"
            )
    if len(set(columns.values())) != len(columns):
        raise AnnotatedMatrixError(
            "wide assertion resolves multiple labels to one column"
        )
    return columns


def _check_expectations(
    expectations: ExportExpectations | None,
    selected: int,
    rejected: Mapping[str, int],
    labels: Counter[str],
) -> None:
    if expectations is None:
        return
    if expectations.selected is not None and expectations.selected != selected:
        raise AnnotatedMatrixError(
            f"selected expectation failed: expected {expectations.selected}, observed {selected}"
        )
    if expectations.rejected is not None and dict(
        sorted(expectations.rejected.items())
    ) != dict(rejected):
        raise AnnotatedMatrixError("rejected-count expectations failed")
    if expectations.source_labels is not None and dict(
        sorted(expectations.source_labels.items())
    ) != dict(sorted(labels.items())):
        raise AnnotatedMatrixError("source-label count expectations failed")


def _count_strings(
    values: Sequence[str], deadline: float | None = None
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for index, value in enumerate(values):
        _check_loop_timeout(deadline, index)
        counts[value] += 1
    _check_timeout(deadline)
    return counts


def _prepare_destination(destination: Path) -> None:
    existing = destination
    while not existing.exists():
        existing = existing.parent
    if existing.is_symlink() or not existing.is_dir():
        raise AnnotatedMatrixError(
            f"destination ancestor is not a real directory: {existing}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    current = destination
    while True:
        if current.is_symlink() or not current.is_dir():
            raise AnnotatedMatrixError(
                f"destination contains a symlink component: {current}"
            )
        if current.parent == current:
            break
        current = current.parent


def _require_regular_input(path: Path, kind: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise AnnotatedMatrixError(
            f"{kind} must be an existing regular non-symlink file: {path}"
        )


def _required_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise AnnotatedMatrixError(f"required column is missing: {column}")
    return frame[column]


def _required_layer(data: ad.AnnData, layer: str) -> Any:
    if layer not in data.layers:
        raise AnnotatedMatrixError(f"required H5AD layer is missing: {layer}")
    return data.layers[layer]


def _referenced_values(
    frame: pd.DataFrame,
    column: str | None,
    index: bool | str | None,
    role: str,
) -> Sequence[Any]:
    if column is not None:
        return _required_column(frame, column).tolist()
    if index is True:
        if isinstance(frame.index, pd.MultiIndex):
            raise AnnotatedMatrixError(
                f"{role}_index=true is ambiguous for a multi-index"
            )
        return frame.index.tolist()
    if isinstance(index, str):
        if index not in frame.index.names:
            raise AnnotatedMatrixError(f"required index level is missing: {index}")
        return frame.index.get_level_values(index).tolist()
    raise AssertionError("validated reference")


def _exactly_one_reference(
    role: str, column: str | None, index: bool | str | None
) -> None:
    if (column is None) == (index is None or index is False):
        raise ValueError(f"{role} must declare exactly one column or index")


def _validate_label_decisions(
    decisions: Mapping[str, str],
    identity: Sequence[str],
    ignored: Sequence[str],
    context: Sequence[str],
) -> None:
    _validate_strings(tuple(decisions), "raw label")
    _validate_strings(tuple(decisions.values()), "source label")
    _validate_strings(tuple(identity), "identity label")
    _validate_strings(tuple(ignored), "ignored label")
    _validate_strings(tuple(context), "context label")
    for name, values in (
        ("identity", identity),
        ("ignored", ignored),
        ("context", context),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"{name} labels must be unique")
    groups = [set(decisions), set(identity), set(ignored), set(context)]
    if any(
        groups[left] & groups[right]
        for left in range(4)
        for right in range(left + 1, 4)
    ):
        raise ValueError(
            "decision, identity, ignored, and context labels must be disjoint"
        )


def _effective_decisions(
    decisions: Mapping[str, str], identity: Sequence[str]
) -> dict[str, str]:
    return {**decisions, **{label: label for label in identity}}


def _decide_label(
    raw: str,
    decisions: Mapping[str, str],
    ignored: Sequence[str],
    context: Sequence[str],
) -> tuple[str | None, str | None]:
    if raw in decisions:
        return None, decisions[raw]
    if raw in ignored:
        return "ignored_label", None
    if raw in context:
        return "context_label", None
    return "unmapped_label", None


def _axis_strings(
    values: Sequence[Any],
    role: str,
    *,
    unique: bool = True,
    deadline: float | None = None,
) -> tuple[str, ...]:
    converted = []
    for index, value in enumerate(values):
        _check_loop_timeout(deadline, index)
        converted.append(_string(value, role))
    result = tuple(converted)
    if unique and len(result) != len(set(result)):
        raise AnnotatedMatrixError(f"{role}s must be unique")
    _check_timeout(deadline)
    return result


def _selected_axis_values(
    values: Sequence[str], selected: Sequence[int], deadline: float
) -> tuple[str, ...]:
    result = []
    for index, selected_index in enumerate(selected):
        _check_loop_timeout(deadline, index)
        result.append(values[selected_index])
    _check_timeout(deadline)
    return tuple(result)


def _string(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnnotatedMatrixError(f"{role} must be a non-empty string")
    return value


def _validate_strings(
    values: Sequence[Any], role: str, deadline: float | None = None
) -> None:
    for index, value in enumerate(values):
        _check_loop_timeout(deadline, index)
        _string(value, role)


def _validate_numeric_values(values: np.ndarray, deadline: float) -> None:
    if values.size == 0:
        return
    _check_timeout(deadline)
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AnnotatedMatrixError("counts must be numeric") from error
    if not np.all(np.isfinite(numeric)):
        raise AnnotatedMatrixError("counts must be finite")
    if np.any(numeric < 0):
        raise AnnotatedMatrixError("counts must be non-negative")
    if np.any(numeric != np.floor(numeric)):
        raise AnnotatedMatrixError("counts must be integral")
    if np.any(numeric > _INT32_MAX):
        raise AnnotatedMatrixError("counts exceed int32 range")
    _check_timeout(deadline)


def _exact_equal(left: Any, right: Any) -> bool:
    if pd.isna(left):
        return False
    return (
        type(left) is type(right)
        and bool(left == right)
        or (
            isinstance(left, np.generic)
            and bool(left == right)
            and left.item() == right
        )
    )


def _require_int32_csr_layout(
    rows: int,
    features: int,
    selected_columns: Sequence[int],
    deadline: float | None = None,
) -> None:
    for value, role in ((rows, "row count"), (features, "feature count")):
        if value < 0 or value > _INT32_MAX:
            raise AnnotatedMatrixError(f"{role} exceeds int32 range")
    if len(selected_columns) > _INT32_MAX:
        raise AnnotatedMatrixError("selected column count exceeds int32 range")
    for index, column in enumerate(selected_columns):
        _check_loop_timeout(deadline, index)
        if column < 0 or column > _INT32_MAX:
            raise AnnotatedMatrixError("selected column indices exceed int32 range")


def _checked_int32_nnz(current: int, additional: int) -> int:
    if current < 0 or additional < 0 or additional > _INT32_MAX - current:
        raise AnnotatedMatrixError("CSR NNZ exceeds int32 range")
    return current + additional


def _safe_id(value: Any) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("export ID must be a safe non-empty path component")
    if any(ord(character) < 32 for character in value):
        raise ValueError("export ID must not contain control characters")


def _resolve_declared_path(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve(strict=False)


def _resolve_truth_paths(truth: Truth, base: Path) -> Truth:
    assertion = truth.wide_assertion
    if assertion is not None:
        assertion = assertion.model_copy(
            update={"path": _resolve_declared_path(base, assertion.path)}
        )
    if isinstance(truth, ObservationColumnTruth):
        return truth.model_copy(update={"wide_assertion": assertion})
    tables = tuple(
        table.model_copy(update={"path": _resolve_declared_path(base, table.path)})
        for table in truth.tables
    )
    return truth.model_copy(update={"tables": tables, "wide_assertion": assertion})


def _resolved_truth(plan: ExportPlan, export: ExportSpec) -> Truth:
    truth = export.truth if export.truth is not None else plan.truth
    if truth is None:
        raise AssertionError("validated export truth")
    return truth


def _check_timeout(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("annotated matrix materialization timed out")


def _check_loop_timeout(deadline: float | None, index: int) -> None:
    if index % _LOOP_CHECK_ITEMS == 0:
        _check_timeout(deadline)


def _file_sha256(path: Path, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            _check_timeout(deadline)
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    _check_timeout(deadline)
    return digest.hexdigest()


def _hash_parts(parts: Iterable[bytes], deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    for index, part in enumerate(parts):
        _check_loop_timeout(deadline, index)
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    _check_timeout(deadline)
    return digest.hexdigest()


def _ordered_axis_hash(values: Sequence[str], deadline: float | None = None) -> str:
    return _hash_parts((value.encode("utf-8") for value in values), deadline)


def _ordered_pair_hash(
    left: Sequence[str], right: Sequence[str], deadline: float | None = None
) -> str:
    return _hash_parts(
        (
            item.encode("utf-8")
            for pair in zip(left, right, strict=True)
            for item in pair
        ),
        deadline,
    )


def _csr_hash(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: np.ndarray,
    *,
    deadline: float | None = None,
) -> str:
    digest = hashlib.sha256()
    for array in (data, indices, indptr, shape):
        _check_timeout(deadline)
        digest.update((array.size * np.dtype("<i4").itemsize).to_bytes(8, "little"))
        for chunk in _array_chunks(array, deadline):
            canonical = np.asarray(chunk, dtype="<i4", order="C")
            digest.update(memoryview(canonical).cast("B"))
    _check_timeout(deadline)
    return digest.hexdigest()


def _object_hash(value: Any, deadline: float | None = None) -> str:
    _check_timeout(deadline)
    encoded = json.dumps(
        _jsonable(value, deadline), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    _check_timeout(deadline)
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any, deadline: float | None = None) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(sorted(value.items())):
            _check_loop_timeout(deadline, index)
            result[str(key)] = _jsonable(item, deadline)
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            _check_loop_timeout(deadline, index)
            result.append(_jsonable(item, deadline))
        return result
    return value


__all__ = [
    "AnnotatedMatrix",
    "AnnotatedMatrixError",
    "ExactPredicate",
    "ExportExpectations",
    "ExportPlan",
    "ExportSpec",
    "MembershipTable",
    "MembershipTruth",
    "ObservationColumnTruth",
    "WideMembershipAssertion",
    "load_export_plan",
    "materialize_exports",
    "open_annotated_matrix",
    "validate_annotated_matrix",
]
