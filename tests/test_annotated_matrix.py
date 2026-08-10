from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from eyck import annotated_matrix as annotated_matrix_module
from eyck.annotated_matrix import (
    AnnotatedMatrixError,
    ExactPredicate,
    ExportPlan,
    ExportSpec,
    MembershipTable,
    MembershipTruth,
    ObservationColumnTruth,
    WideMembershipAssertion,
    load_export_plan,
    materialize_exports,
    open_annotated_matrix,
    validate_annotated_matrix,
)
from scipy import sparse


def _write_h5ad(path: Path, matrix=None, *, identity_truth: bool = False) -> Path:
    values = np.array(
        [
            [1, 2, 4, 0],
            [9, 0, 1, 0],
            [0, 3, 2, 1],
            [8, 1, 0, 0],
            [5, 0, 7, 2],
        ],
        dtype=np.float64,
    )
    data = ad.AnnData(
        X=sparse.csr_matrix(values if matrix is None else matrix),
        obs=pd.DataFrame(
            {
                "truth": (
                    ["A", "ignore", "B", "context", "A"]
                    if identity_truth
                    else ["raw-a", "ignore", "raw-b", "context", "raw-a"]
                ),
                "cohort": ["keep", "drop", "keep", "drop", "keep"],
            },
            index=["o0", "o1", "o2", "o3", "o4"],
        ),
        var=pd.DataFrame(
            {"symbol": ["GeneA", "GeneA", "genea", "GeneB"]},
            index=["v0", "v1", "v2", "v3"],
        ),
        layers={"counts": sparse.csr_matrix(values * 2)},
    )
    data.write_h5ad(path)
    return path


def _obs_plan(h5ad: Path, *, matrix: str = "X", export_id: str = "cells") -> ExportPlan:
    return ExportPlan(
        h5ad=h5ad,
        matrix=matrix,
        feature_id_column="symbol",
        chunk_rows=2,
        exports=(
            ExportSpec(
                id=export_id,
                feature_symbols=("GeneA", "genea", "GeneB"),
                predicates=(ExactPredicate(column="cohort", include=("keep",)),),
                truth=ObservationColumnTruth(
                    kind="obs_column",
                    column="truth",
                    decisions={"raw-a": "A", "raw-b": "B"},
                    ignored_labels=("ignore",),
                    context_labels=("context",),
                ),
            ),
        ),
    )


def test_backed_x_layer_duplicate_symbols_case_predicate_rerun_and_toml(
    tmp_path: Path,
) -> None:
    h5ad = _write_h5ad(tmp_path / "cells.h5ad")
    wide = pd.DataFrame(
        {"A": [1, 0, 1], "B": [0, 1, 0]},
        index=pd.Index(["o0", "o2", "o4"], name="observation"),
    )
    wide.to_parquet(tmp_path / "wide.parquet")
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(
        """schema = "eyck.annotated-matrix-export-plan.v1"
h5ad = "cells.h5ad"
matrix = "X"
feature_id_column = "symbol"
chunk_rows = 2

[[exports]]
id = "cells"
feature_symbols = ["GeneA", "genea", "GeneB"]
predicates = [{column = "cohort", include = ["keep"]}]

[exports.truth]
kind = "obs_column"
column = "truth"
decisions = {raw-a = "A", raw-b = "B"}
ignored_labels = ["ignore"]
context_labels = ["context"]

[exports.truth.wide_assertion]
path = "wide.parquet"
observation_index = true
label_columns = {A = "A", B = "B"}
present_value = 1
""",
        encoding="utf-8",
    )
    plan = load_export_plan(plan_path)
    assert plan.h5ad == h5ad
    output = materialize_exports(plan, tmp_path / "out")[0]
    opened = open_annotated_matrix(output)
    assert opened.observation_ids == ("o0", "o2", "o4")
    assert opened.source_label_ids == ("A", "B", "A")
    assert opened.feature_ids == ("GeneA", "genea", "GeneB")
    np.testing.assert_array_equal(
        opened.counts.toarray(),
        np.array([[3, 4, 0], [3, 2, 1], [5, 7, 2]], dtype=np.int32),
    )
    manifest_before = (output / "manifest.json").read_bytes()
    modified_before = output.stat().st_mtime_ns
    assert materialize_exports(plan, tmp_path / "out") == (output,)
    assert (output / "manifest.json").read_bytes() == manifest_before
    assert output.stat().st_mtime_ns == modified_before

    layer_output = materialize_exports(
        _obs_plan(h5ad, matrix="counts", export_id="layer"), tmp_path / "out"
    )[0]
    np.testing.assert_array_equal(
        open_annotated_matrix(layer_output).counts.toarray(),
        opened.counts.toarray() * 2,
    )


def test_omitted_features_export_all_deduplicated_symbols_in_sorted_order(
    tmp_path: Path,
) -> None:
    h5ad = _write_h5ad(tmp_path / "cells.h5ad")
    plan = _obs_plan(h5ad).model_copy(
        update={
            "exports": (
                _obs_plan(h5ad).exports[0].model_copy(update={"feature_symbols": None}),
            )
        }
    )
    opened = open_annotated_matrix(materialize_exports(plan, tmp_path / "out")[0])
    assert opened.feature_ids == ("GeneA", "GeneB", "genea")
    np.testing.assert_array_equal(
        opened.counts.toarray(),
        np.array([[3, 0, 4], [3, 1, 2], [5, 2, 7]], dtype=np.int32),
    )


def test_shared_truth_and_observation_identity_shorthand_are_semantic(
    tmp_path: Path,
) -> None:
    h5ad = _write_h5ad(tmp_path / "identity.h5ad", identity_truth=True)
    shared_truth = ObservationColumnTruth(
        kind="obs_column",
        column="truth",
        decisions={"B": "B"},
        identity_labels=("A",),
        ignored_labels=("ignore",),
        context_labels=("context",),
    )
    shared = ExportPlan(
        h5ad=h5ad,
        feature_id_column="symbol",
        truth=shared_truth,
        exports=(
            ExportSpec(id="first", feature_symbols=("GeneA",)),
            ExportSpec(id="second", feature_symbols=("GeneB",)),
        ),
    )
    outputs = materialize_exports(shared, tmp_path / "shared")
    assert [open_annotated_matrix(output).source_label_ids for output in outputs] == [
        ("A", "B", "A"),
        ("A", "B", "A"),
    ]

    explicit = ExportPlan(
        h5ad=h5ad,
        feature_id_column="symbol",
        exports=(
            ExportSpec(
                id="first",
                feature_symbols=("GeneA",),
                truth=ObservationColumnTruth(
                    kind="obs_column",
                    column="truth",
                    decisions={"A": "A", "B": "B"},
                    ignored_labels=("ignore",),
                    context_labels=("context",),
                ),
            ),
        ),
    )
    explicit_output = materialize_exports(explicit, tmp_path / "explicit")[0]
    shared_manifest = validate_annotated_matrix(outputs[0])
    explicit_manifest = validate_annotated_matrix(explicit_output)
    assert shared_manifest["semantic_hashes"] == explicit_manifest["semantic_hashes"]
    assert shared_manifest["members"] == explicit_manifest["members"]


def test_truth_and_identity_shorthand_conflicts_are_rejected(tmp_path: Path) -> None:
    h5ad = _write_h5ad(tmp_path / "cells.h5ad")
    truth = ObservationColumnTruth(
        kind="obs_column",
        column="truth",
        decisions={"raw-a": "A", "raw-b": "B"},
        ignored_labels=("ignore",),
        context_labels=("context",),
    )
    override = ExportPlan(
        h5ad=h5ad,
        feature_id_column="symbol",
        truth=truth,
        exports=(
            ExportSpec(
                id="override",
                feature_symbols=("GeneB",),
                truth=ObservationColumnTruth(
                    kind="obs_column",
                    column="truth",
                    decisions={"raw-a": "X", "raw-b": "Y"},
                    ignored_labels=("ignore",),
                    context_labels=("context",),
                ),
            ),
        ),
    )
    override_output = materialize_exports(override, tmp_path / "override")[0]
    assert open_annotated_matrix(override_output).source_label_ids == ("X", "Y", "X")
    with pytest.raises(
        ValueError, match="requires an export-level or plan-level truth"
    ):
        ExportPlan(h5ad=h5ad, exports=(ExportSpec(id="missing"),))
    with pytest.raises(ValueError, match="must be disjoint"):
        ObservationColumnTruth(
            kind="obs_column",
            column="truth",
            decisions={"A": "mapped"},
            identity_labels=("A",),
        )
    with pytest.raises(ValueError, match="must be disjoint"):
        MembershipTable(
            path=tmp_path / "members.parquet",
            observation_column="observation",
            entity_column="entity",
            label_column="label",
            value_column="present",
            identity_labels=("A",),
            ignored_labels=("A",),
        )
    with pytest.raises(ValueError, match="columns must be unique"):
        WideMembershipAssertion(
            path=tmp_path / "wide.parquet",
            observation_column="observation",
            label_columns={"A": "same", "B": "same"},
        )
    identity_h5ad = _write_h5ad(tmp_path / "identity.h5ad", identity_truth=True)
    pd.DataFrame(
        {
            "observation": ["o0", "o2", "o4"],
            "A": [1, 0, 1],
            "B": [0, 1, 0],
        }
    ).to_parquet(tmp_path / "wide.parquet", index=False)
    ambiguous = ExportPlan(
        h5ad=identity_h5ad,
        truth=ObservationColumnTruth(
            kind="obs_column",
            column="truth",
            identity_labels=("A", "B"),
            ignored_labels=("ignore",),
            context_labels=("context",),
            wide_assertion=WideMembershipAssertion(
                path=tmp_path / "wide.parquet",
                observation_column="observation",
                identity_columns=True,
                label_columns={"A": "B"},
            ),
        ),
        exports=(ExportSpec(id="ambiguous", feature_symbols=("GeneB",)),),
    )
    with pytest.raises(AnnotatedMatrixError, match="multiple labels to one column"):
        materialize_exports(ambiguous, tmp_path / "ambiguous-output")


def test_plan_selection_is_independent_of_plan_and_input_placement(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first-root"
    second = tmp_path / "second-root"
    first.mkdir()
    second.mkdir()
    _write_h5ad(first / "source.h5ad")
    shutil.copyfile(first / "source.h5ad", second / "relocated.h5ad")
    memberships = pd.DataFrame(
        {
            "observation": ["o0", "o2", "o4"],
            "entity": ["e0", "e2", "e4"],
            "label": ["A", "B", "A"],
            "present": [1, 1, 1],
        }
    )
    memberships.to_parquet(first / "members.parquet", index=False)
    shutil.copyfile(first / "members.parquet", second / "renamed.parquet")
    wide = pd.DataFrame(
        {
            "observation": ["o0", "o2", "o4"],
            "A": [1, 0, 1],
            "B": [0, 1, 0],
            "B_override": [0, 1, 0],
        }
    )
    wide.to_parquet(first / "assertion.parquet", index=False)
    shutil.copyfile(first / "assertion.parquet", second / "other-name.parquet")
    assert (first / "source.h5ad").read_bytes() == (
        second / "relocated.h5ad"
    ).read_bytes()
    assert (first / "members.parquet").read_bytes() == (
        second / "renamed.parquet"
    ).read_bytes()
    assert (first / "assertion.parquet").read_bytes() == (
        second / "other-name.parquet"
    ).read_bytes()

    def write_plan(
        path: Path,
        *,
        h5ad: str,
        membership: str,
        assertion: str,
        chunk_rows: int,
        shorthand: bool,
    ) -> None:
        label_declaration = (
            'identity_labels = ["A"]\ndecisions = {B = "B"}'
            if shorthand
            else 'decisions = {A = "A", B = "B"}'
        )
        assertion_declaration = (
            'identity_columns = true\nlabel_columns = {B = "B_override"}'
            if shorthand
            else 'label_columns = {A = "A", B = "B_override"}'
        )
        path.write_text(
            f'''schema = "eyck.annotated-matrix-export-plan.v1"
h5ad = "{h5ad}"
matrix = "X"
feature_id_column = "symbol"
chunk_rows = {chunk_rows}

[[exports]]
id = "portable"
predicates = [{{column = "cohort", include = ["keep"]}}]

[truth]
kind = "memberships"
mode = "combined_exactly_one"

[[truth.tables]]
path = "{membership}"
observation_column = "observation"
entity_column = "entity"
label_column = "label"
value_column = "present"
present_value = 1
{label_declaration}
priorities = {{A = 1, B = 2}}

[truth.wide_assertion]
path = "{assertion}"
observation_column = "observation"
{assertion_declaration}
present_value = 1
''',
            encoding="utf-8",
        )

    write_plan(
        first / "plan.toml",
        h5ad="source.h5ad",
        membership="members.parquet",
        assertion="assertion.parquet",
        chunk_rows=1,
        shorthand=True,
    )
    write_plan(
        second / "different-plan-name.toml",
        h5ad="relocated.h5ad",
        membership="renamed.parquet",
        assertion="other-name.parquet",
        chunk_rows=4096,
        shorthand=False,
    )
    first_output = materialize_exports(
        load_export_plan(first / "plan.toml"), first / "output"
    )[0]
    second_output = materialize_exports(
        load_export_plan(second / "different-plan-name.toml"), second / "elsewhere"
    )[0]
    first_manifest = validate_annotated_matrix(first_output)
    second_manifest = validate_annotated_matrix(second_output)
    assert (
        first_manifest["semantic_hashes"]["plan_selection"]
        == second_manifest["semantic_hashes"]["plan_selection"]
    )
    assert first_manifest["semantic_hashes"] == second_manifest["semantic_hashes"]
    assert first_manifest["members"] == second_manifest["members"]
    assert open_annotated_matrix(first_output).feature_ids == (
        "GeneA",
        "GeneB",
        "genea",
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (-1, "non-negative"),
        (0.5, "integral"),
        (float("nan"), "finite"),
        (np.iinfo(np.int32).max + 1, "int32"),
    ],
)
def test_invalid_counts_are_rejected(
    tmp_path: Path, value: float, message: str
) -> None:
    matrix = np.zeros((5, 4), dtype=np.float64)
    matrix[0, 0] = value
    h5ad = _write_h5ad(tmp_path / "bad.h5ad", matrix)
    with pytest.raises(AnnotatedMatrixError, match=message):
        materialize_exports(_obs_plan(h5ad), tmp_path / "out")


def test_duplicate_symbol_sum_overflow_is_rejected(tmp_path: Path) -> None:
    matrix = np.zeros((5, 4), dtype=np.float64)
    matrix[0, :2] = 1_500_000_000
    h5ad = _write_h5ad(tmp_path / "overflow.h5ad", matrix)
    with pytest.raises(AnnotatedMatrixError, match="summed counts exceed int32"):
        materialize_exports(_obs_plan(h5ad), tmp_path / "out")


def test_count_materialization_does_not_use_sparse_vstack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5ad = _write_h5ad(tmp_path / "cells.h5ad")

    def reject_vstack(*_args, **_kwargs):
        raise AssertionError("full-matrix CSR stacking is forbidden")

    monkeypatch.setattr(sparse, "vstack", reject_vstack)
    output = materialize_exports(_obs_plan(h5ad), tmp_path / "out")[0]

    validate_annotated_matrix(output, deep=True)
    np.testing.assert_array_equal(
        open_annotated_matrix(output).counts.toarray(),
        np.array([[3, 4, 0], [3, 2, 1], [5, 7, 2]], dtype=np.int32),
    )


@pytest.mark.parametrize(
    ("rows", "features", "selected_columns", "message"),
    [
        (annotated_matrix_module._INT32_MAX + 1, 1, (), "row count"),
        (1, annotated_matrix_module._INT32_MAX + 1, (), "feature count"),
        (1, 1, (annotated_matrix_module._INT32_MAX + 1,), "column indices"),
    ],
)
def test_int32_csr_layout_overflow_is_rejected_without_large_allocations(
    rows: int, features: int, selected_columns: tuple[int, ...], message: str
) -> None:
    with pytest.raises(AnnotatedMatrixError, match=message):
        annotated_matrix_module._require_int32_csr_layout(
            rows, features, selected_columns
        )


def test_int32_nnz_overflow_is_rejected_without_large_allocations() -> None:
    with pytest.raises(AnnotatedMatrixError, match="NNZ"):
        annotated_matrix_module._checked_int32_nnz(
            annotated_matrix_module._INT32_MAX, 1
        )


def test_chunked_csr_order_scan_ignores_row_boundaries() -> None:
    indptr = np.array([0, 2, 2, 4], dtype=np.int32)
    canonical = np.array([0, 2, 0, 1], dtype=np.int32)
    duplicate = np.array([0, 0, 0, 1], dtype=np.int32)

    assert not annotated_matrix_module._has_csr_index_order_violation(
        canonical, indptr, deadline=None
    )
    assert annotated_matrix_module._has_csr_index_order_violation(
        duplicate, indptr, deadline=None
    )


def test_streaming_timeout_cleans_temporary_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5ad = _write_h5ad(tmp_path / "cells.h5ad")
    clock = [0.0]
    original_chunks = annotated_matrix_module._iter_count_chunks
    stream_pass = 0

    def expiring_chunks(*args, **kwargs):
        nonlocal stream_pass
        stream_pass += 1
        for chunk_index, chunk in enumerate(original_chunks(*args, **kwargs)):
            yield chunk
            if stream_pass == 2 and chunk_index == 0:
                clock[0] = 2.0

    monkeypatch.setattr(annotated_matrix_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(annotated_matrix_module, "_iter_count_chunks", expiring_chunks)
    destination = tmp_path / "out"

    with pytest.raises(TimeoutError, match="timed out"):
        materialize_exports(_obs_plan(h5ad), destination, timeout_seconds=1.0)

    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_validation_deadline_interrupts_member_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5ad = _write_h5ad(tmp_path / "cells.h5ad")
    output = materialize_exports(_obs_plan(h5ad), tmp_path / "out")[0]
    clock = [0.0]
    original_file_sha256 = annotated_matrix_module._file_sha256
    hash_calls = 0

    def expiring_hash(path, deadline=None):
        nonlocal hash_calls
        assert deadline == 1.0
        result = original_file_sha256(path, deadline)
        hash_calls += 1
        clock[0] = 2.0
        return result

    monkeypatch.setattr(annotated_matrix_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(annotated_matrix_module, "_file_sha256", expiring_hash)

    with pytest.raises(TimeoutError, match="timed out"):
        annotated_matrix_module._validate_annotated_matrix(
            output, deep=True, deadline=1.0
        )

    assert hash_calls == 1
    monkeypatch.setattr(annotated_matrix_module, "_file_sha256", original_file_sha256)
    clock[0] = 0.0
    assert open_annotated_matrix(output).counts.shape == (3, 3)


def _membership_table(
    path: Path, decisions: dict[str, str], priorities=None
) -> MembershipTable:
    return MembershipTable(
        path=path,
        observation_column="observation",
        entity_column="entity",
        label_column="label",
        value_column="present",
        decisions=decisions,
        ignored_labels=("ignored",),
        context_labels=("context",),
        priorities=priorities or {},
    )


def test_combined_membership_priority_and_rejection_reasons(tmp_path: Path) -> None:
    h5ad = _write_h5ad(tmp_path / "cells.h5ad")
    first = pd.DataFrame(
        {
            "observation": ["o0", "o0", "o1", "o1", "o2", "o3", "o3"],
            "entity": ["e0", "e0", "left", "right", "e2", "e3", "e3"],
            "label": ["ancestor", "zone", "ancestor", "zone", "unknown", "a", "b"],
            "present": [1] * 7,
        }
    )
    first.to_parquet(tmp_path / "first.parquet", index=False)
    second = pd.DataFrame(
        {
            "observation": ["o4"],
            "entity": ["e4"],
            "label": ["zone"],
            "present": [1],
        }
    )
    second.to_parquet(tmp_path / "second.parquet", index=False)
    truth = MembershipTruth(
        kind="memberships",
        mode="combined_exactly_one",
        tables=(
            _membership_table(
                tmp_path / "first.parquet",
                {"ancestor": "hepatocyte", "zone": "portal", "a": "A", "b": "B"},
                {"ancestor": 0, "zone": 2, "a": 1, "b": 1},
            ),
            _membership_table(
                tmp_path / "second.parquet", {"zone": "portal"}, {"zone": 2}
            ),
        ),
    )
    plan = ExportPlan(
        h5ad=h5ad,
        feature_id_column="symbol",
        exports=(ExportSpec(id="combined", feature_symbols=("GeneB",), truth=truth),),
    )
    output = materialize_exports(plan, tmp_path / "out")[0]
    opened = open_annotated_matrix(output)
    assert opened.observation_ids == ("o0", "o4")
    assert opened.source_label_ids == ("portal", "portal")
    assert opened.manifest["counts"]["rejected"] == {
        "conflicting_labels": 1,
        "multiple_entities": 1,
        "unmapped_label": 1,
    }


def test_per_table_singlet_union_conflict_and_timeout(tmp_path: Path) -> None:
    h5ad = _write_h5ad(tmp_path / "cells.h5ad")
    pd.DataFrame(
        {
            "observation": ["o0", "o1", "o2", "o2"],
            "entity": ["e0", "e1", "left", "right"],
            "label": ["a", "a", "a", "a"],
            "present": [1, 1, 1, 1],
        }
    ).to_parquet(tmp_path / "a.parquet", index=False)
    pd.DataFrame(
        {
            "observation": ["o0", "o1", "o3"],
            "entity": ["x0", "x1", "x3"],
            "label": ["a", "b", "ignored"],
            "present": [1, 1, 1],
        }
    ).to_parquet(tmp_path / "b.parquet", index=False)
    truth = MembershipTruth(
        kind="memberships",
        mode="per_table_exactly_one_union",
        tables=(
            _membership_table(tmp_path / "a.parquet", {"a": "A"}),
            _membership_table(tmp_path / "b.parquet", {"a": "A", "b": "B"}),
        ),
    )
    plan = ExportPlan(
        h5ad=h5ad,
        feature_id_column="symbol",
        exports=(ExportSpec(id="union", feature_symbols=("GeneB",), truth=truth),),
    )
    output = materialize_exports(plan, tmp_path / "out")[0]
    opened = open_annotated_matrix(output)
    assert opened.observation_ids == ("o0",)
    assert opened.manifest["counts"]["rejected"] == {
        "conflicting_labels": 1,
        "ignored_label": 1,
        "multiple_entities": 1,
        "zero_entities": 1,
    }
    with pytest.raises(TimeoutError, match="timed out"):
        materialize_exports(plan, tmp_path / "later", timeout_seconds=1e-300)


def test_deep_validation_rejects_tamper_extra_symlink_and_malformed_csr(
    tmp_path: Path,
) -> None:
    h5ad = _write_h5ad(tmp_path / "cells.h5ad")
    output = materialize_exports(_obs_plan(h5ad), tmp_path / "out")[0]

    (output / "extra").write_text("x", encoding="utf-8")
    with pytest.raises(AnnotatedMatrixError, match="extra"):
        validate_annotated_matrix(output)
    (output / "extra").unlink()

    data_path = output / "counts" / "data.npy"
    data_path.write_bytes(data_path.read_bytes() + b"tamper")
    with pytest.raises(AnnotatedMatrixError, match="size mismatch"):
        validate_annotated_matrix(output)

    materialize_exports(_obs_plan(h5ad), tmp_path / "out", replace=True)
    feature_path = output / "features.parquet"
    feature_path.unlink()
    feature_path.symlink_to(tmp_path / "cells.h5ad")
    with pytest.raises(AnnotatedMatrixError, match="non-symlink"):
        validate_annotated_matrix(output)

    materialize_exports(_obs_plan(h5ad), tmp_path / "out", replace=True)
    indices_path = output / "counts" / "indices.npy"
    indices = np.load(indices_path, allow_pickle=False)
    indices[0] = 99
    with indices_path.open("wb") as handle:
        np.save(handle, indices, allow_pickle=False)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = indices_path.read_bytes()
    manifest["members"]["counts/indices.npy"] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(AnnotatedMatrixError, match="indices"):
        validate_annotated_matrix(output, deep=False)
    with pytest.raises(AnnotatedMatrixError, match="indices"):
        validate_annotated_matrix(output)

    materialize_exports(_obs_plan(h5ad), tmp_path / "out", replace=True)
    data_path = output / "counts" / "data.npy"
    data_values = np.load(data_path, allow_pickle=False)
    data_values[0] += 1
    with data_path.open("wb") as handle:
        np.save(handle, data_values, allow_pickle=False)
    with pytest.raises(AnnotatedMatrixError, match="SHA256 mismatch"):
        validate_annotated_matrix(output, deep=True)


def test_open_keeps_csr_members_read_only_and_memory_backed(tmp_path: Path) -> None:
    h5ad = _write_h5ad(tmp_path / "cells.h5ad")
    output = materialize_exports(_obs_plan(h5ad), tmp_path / "out")[0]

    opened = open_annotated_matrix(output)

    for values in (opened.counts.data, opened.counts.indices, opened.counts.indptr):
        assert not values.flags.writeable
        current = values
        while isinstance(current, np.ndarray) and not isinstance(current, np.memmap):
            current = current.base
        assert isinstance(current, np.memmap)
    with pytest.raises(ValueError, match="read-only"):
        opened.counts.data[0] = 0


def test_open_allows_empty_csr_members(tmp_path: Path) -> None:
    h5ad = _write_h5ad(tmp_path / "cells.h5ad")
    plan = _obs_plan(h5ad)
    empty_export = plan.exports[0].model_copy(
        update={
            "predicates": (ExactPredicate(column="cohort", equals="absent"),),
        }
    )
    output = materialize_exports(
        plan.model_copy(update={"exports": (empty_export,)}), tmp_path / "out"
    )[0]

    opened = open_annotated_matrix(output)

    assert opened.counts.shape == (0, 3)
    assert opened.counts.nnz == 0


def test_existing_different_export_requires_replace(tmp_path: Path) -> None:
    h5ad = _write_h5ad(tmp_path / "cells.h5ad")
    output = materialize_exports(_obs_plan(h5ad), tmp_path / "out")[0]
    changed = _obs_plan(h5ad).model_copy(
        update={
            "exports": (
                _obs_plan(h5ad)
                .exports[0]
                .model_copy(update={"feature_symbols": ("GeneB",)}),
            )
        }
    )
    with pytest.raises(AnnotatedMatrixError, match="replace"):
        materialize_exports(changed, tmp_path / "out")
    materialize_exports(changed, tmp_path / "out", replace=True)
    assert open_annotated_matrix(output).feature_ids == ("GeneB",)


def test_root_symlink_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link, target_is_directory=True)
    with pytest.raises(AnnotatedMatrixError, match="real directory"):
        validate_annotated_matrix(link)
