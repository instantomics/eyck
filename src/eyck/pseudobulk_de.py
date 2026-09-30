"""Differential expression of perturbed against control pseudobulks with DESeq2.

Samples are pseudobulks of one cell group (e.g. a cell type) from one
replicate under one perturbation or a control. Treated samples are compared
with controls of the same group that share their block: the replicate in a
paired design, or the processing batch when controls are shared per batch.
One negative-binomial model per group, with the block as a covariate, shares
dispersion estimates across that group's perturbations. Fold changes are
unshrunk; adjusted p-values follow DESeq2's independent filtering, and genes
it filters out or flags as outliers are omitted.

Requires the optional `de` extra (pydeseq2).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

COLUMNS = (
    "perturbation",
    "group",
    "gene",
    "log2fc",
    "pvalue",
    "padj",
    "n_treated",
    "n_controls",
)


def contrasts(samples, *, group, block, min_cells=10, min_replicates=2):
    """Treated and matched control sample IDs for every perturbation x group."""
    usable = samples[samples.n_cells >= min_cells]
    rows = []
    for (perturbation, name), treated in usable[~usable.control].groupby(
        ["perturbation", group], sort=True
    ):
        controls = usable[usable.control & (usable[group] == name)]
        blocks = set(treated[block]) & set(controls[block])
        treated = treated[treated[block].isin(blocks)]
        controls = controls[controls[block].isin(blocks)]
        if len(treated) >= min_replicates and len(controls) >= min_replicates:
            rows.append(
                {
                    "perturbation": perturbation,
                    "group": name,
                    "treated": treated.sample_id.tolist(),
                    "controls": controls.sample_id.tolist(),
                }
            )
    return pd.DataFrame(rows, columns=["perturbation", "group", "treated", "controls"])


def pseudobulk_differential_expression(
    samples,
    counts,
    genes,
    *,
    group,
    block,
    min_cells=10,
    min_replicates=2,
    alpha=0.1,
    n_cpus=1,
):
    """DESeq2 statistics for every testable perturbation x group.

    `samples` needs `sample_id`, `control` (bool), `perturbation` (null for
    controls), `n_cells`, and the `group` and `block` columns; `counts` is a
    samples x genes integer matrix in `samples` order, `genes` the gene names.
    Returns one row per tested gene and contrast with the columns in `COLUMNS`.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    logging.getLogger("pydeseq2").setLevel(logging.ERROR)
    genes = np.asarray(genes)
    table = contrasts(
        samples,
        group=group,
        block=block,
        min_cells=min_cells,
        min_replicates=min_replicates,
    )
    position = pd.Series(np.arange(len(samples)), index=samples.sample_id.to_numpy())
    inference = DefaultInference(n_cpus=n_cpus)
    results = []
    for name, selected in table.groupby("group", sort=True):
        members = sorted(
            {
                s
                for column in ("treated", "controls")
                for ids in selected[column]
                for s in ids
            },
            key=position.get,
        )
        rows = position[members].to_numpy()
        matrix = counts[rows]
        matrix = np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix)
        kept = matrix.sum(axis=0) >= 10
        chosen = samples.iloc[rows]
        # Formula-safe levels; "c" sorts before "p" so control is the reference.
        codes = {p: f"p{i:04d}" for i, p in enumerate(sorted(selected.perturbation))}
        levels = np.where(chosen.control, "c", chosen.perturbation.map(codes))
        blocks = {b: f"b{i:04d}" for i, b in enumerate(sorted(set(chosen[block])))}
        metadata = pd.DataFrame(
            {"perturbation": levels, "block": chosen[block].map(blocks).to_numpy()},
            index=chosen.sample_id.to_numpy(),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dds = DeseqDataSet(
                counts=pd.DataFrame(
                    matrix[:, kept], index=metadata.index, columns=genes[kept]
                ),
                metadata=metadata,
                design="~block + perturbation" if len(blocks) > 1 else "~perturbation",
                inference=inference,
                quiet=True,
            )
            dds.deseq2()
            for contrast in selected.itertuples():
                stats = DeseqStats(
                    dds,
                    contrast=["perturbation", codes[contrast.perturbation], "c"],
                    alpha=alpha,
                    inference=inference,
                    quiet=True,
                )
                stats.summary()
                frame = stats.results_df.dropna(subset=["padj"])
                results.append(
                    pd.DataFrame(
                        {
                            "perturbation": contrast.perturbation,
                            "group": name,
                            "gene": frame.index.to_numpy(),
                            "log2fc": frame.log2FoldChange.to_numpy(float),
                            "pvalue": frame.pvalue.to_numpy(float),
                            "padj": frame.padj.to_numpy(float),
                            "n_treated": len(contrast.treated),
                            "n_controls": len(contrast.controls),
                        }
                    )
                )
    if not results:
        return pd.DataFrame(columns=list(COLUMNS))
    return pd.concat(results, ignore_index=True)[list(COLUMNS)]
