import numpy as np
import pandas as pd
import pytest
from eyck.pseudobulk_de import contrasts, pseudobulk_differential_expression
from scipy import sparse


def samples(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "cell_type",
            "control",
            "perturbation",
            "batch",
            "n_cells",
        ],
    )


def test_controls_match_by_block_and_small_or_unmatched_samples_drop():
    table = samples(
        [
            ("t1", "T", False, "X", "b1", 50),
            ("t2", "T", False, "X", "b2", 50),
            ("t3", "T", False, "X", "b3", 50),  # no control in b3
            ("t4", "T", False, "X", "b1", 5),  # too few cells
            ("c1", "T", True, None, "b1", 50),
            ("c2", "T", True, None, "b2", 50),
            ("c9", "B", True, None, "b1", 50),  # other cell type
        ]
    )
    (row,) = contrasts(table, group="cell_type", block="batch").itertuples()
    assert sorted(row.treated) == ["t1", "t2"] and sorted(row.controls) == ["c1", "c2"]


@pytest.mark.parametrize("bulk", [False, True])
def test_recovers_a_planted_paired_effect(bulk):
    pytest.importorskip("pydeseq2")
    rng = np.random.default_rng(0)
    rows, matrix = [], []
    mean = rng.lognormal(3, 1, 300)
    effect = np.zeros(300)
    effect[:20] = 3
    for donor in range(4):
        shift = rng.normal(0, 0.3, 300)
        for label in ("PBS", "X"):
            rows.append(
                (
                    f"{donor}{label}",
                    "T",
                    label == "PBS",
                    None if label == "PBS" else "X",
                    f"d{donor}",
                    100,
                )
            )
            rate = mean * np.exp(shift + (0 if label == "PBS" else effect))
            matrix.append(rng.negative_binomial(20, 20 / (20 + rate)))
    table = samples(rows)
    if bulk:
        table = table.drop(columns="n_cells")
    result = pseudobulk_differential_expression(
        table,
        sparse.csr_matrix(matrix),
        [f"g{i}" for i in range(300)],
        group="cell_type",
        block="batch",
        min_cells=None if bulk else 10,
    )
    hits = result[(result.padj <= 0.1) & (result.log2fc.abs() >= 1)].gene
    assert set(hits) >= {f"g{i}" for i in range(20)}
    assert len(set(hits) - {f"g{i}" for i in range(20)}) <= 3
