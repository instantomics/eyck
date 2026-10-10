"""Export a small labeled raw-count population without an AnnData container."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def write_labeled_counts(
    directory: Path,
    counts: sparse.spmatrix,
    observation_ids: list[str],
    feature_ids: list[str],
    labels: list[str],
) -> None:
    """Write CSR counts with aligned axes, flat memberships and a flat ontology.

    The caller owns directory creation, permissions and publication. Counts stay
    raw; duplicate feature identities and nonintegral values are rejected.
    """
    matrix = sparse.csr_matrix(counts, copy=True)
    if matrix.shape != (len(observation_ids), len(feature_ids)):
        raise ValueError("matrix shape does not match the supplied axes")
    if len(labels) != len(observation_ids):
        raise ValueError("memberships do not match the observation axis")
    for values in (observation_ids, feature_ids):
        if len(set(values)) != len(values) or any(not item for item in values):
            raise ValueError("axis identities must be nonempty and unique")
    if any(not item for item in labels):
        raise ValueError("labels must be nonempty")
    if (
        not np.isfinite(matrix.data).all()
        or np.any(matrix.data < 0)
        or np.any(matrix.data != np.floor(matrix.data))
    ):
        raise ValueError("expected finite nonnegative integer counts")
    if matrix.data.size and matrix.data.max() > np.iinfo(np.int32).max:
        raise ValueError("counts exceed int32 range")
    matrix = matrix.astype(np.int64)
    matrix.sum_duplicates()
    matrix.sort_indices()
    if matrix.data.size and matrix.data.max() > np.iinfo(np.int32).max:
        raise ValueError("counts exceed int32 range")
    np.savez_compressed(
        directory / "counts.npz",
        data=matrix.data.astype(np.int32),
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape),
        format=np.asarray("csr"),
        observation_ids=np.asarray(observation_ids, dtype=str),
        feature_ids=np.asarray(feature_ids, dtype=str),
    )
    pd.DataFrame(
        {"observation_id": observation_ids, "source_label_id": labels}
    ).to_parquet(directory / "membership.parquet", index=False)
    identities = sorted(set(labels))
    pd.DataFrame(
        {
            "source_label_id": identities,
            "parent_source_label_id": pd.array(
                [None] * len(identities), dtype="string"
            ),
        }
    ).to_parquet(directory / "ontology.parquet", index=False)
