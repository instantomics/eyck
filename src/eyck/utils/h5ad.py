from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype, is_string_dtype
from scipy import sparse


def write_h5ad(adata, path: Path) -> None:
    """Write ``adata`` with native pandas dtypes, skipping an identical existing file.

    h5ad cannot serialize Arrow-backed arrays (such as ``ArrowStringArray``), so
    obs, var and raw.var are converted to object-backed strings first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for frame in (adata.obs, adata.var):
        _fix_frame(frame)
    adata.obs_names = _object_string_index(adata.obs_names)
    adata.var_names = _object_string_index(adata.var_names)
    if adata.raw is not None:
        _fix_frame(adata.raw.var)
    if _h5ad_matches(path, adata):
        return
    adata.write_h5ad(path)


def _is_arrow_backed(dtype) -> bool:
    return "[pyarrow]" in str(dtype)


def _object_string_index(index: pd.Index) -> pd.Index:
    return pd.Index(np.asarray([str(x) for x in index], dtype=object), name=index.name, dtype=object)


def _fix_series(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.CategoricalDtype):
        categories = series.cat.categories
        if _is_arrow_backed(categories.dtype) or is_string_dtype(categories.dtype):
            series = series.cat.set_categories(pd.Index(np.asarray(categories, dtype=object), dtype=object))
        return series
    if _is_arrow_backed(series.dtype):
        if is_string_dtype(series.dtype):
            return series.astype(object)
        return pd.Series(np.asarray(series.array), index=series.index, name=series.name)
    if is_extension_array_dtype(series.dtype) and is_string_dtype(series.dtype):
        return series.astype(object)
    return series


def _fix_frame(frame: pd.DataFrame) -> None:
    for column in frame.columns:
        frame[column] = _fix_series(frame[column])
    if isinstance(frame.index, pd.MultiIndex):
        frame.index = frame.index.set_levels(
            [
                pd.Index(np.asarray(level, dtype=object), name=level.name)
                if _is_arrow_backed(level.dtype) or is_string_dtype(level.dtype)
                else level
                for level in frame.index.levels
            ]
        )
    elif _is_arrow_backed(frame.index.dtype) or is_string_dtype(frame.index.dtype):
        frame.index = _object_string_index(frame.index)


def _array_equal(a, b) -> bool:
    if sparse.issparse(a) or sparse.issparse(b):
        return (sparse.csr_matrix(a) != sparse.csr_matrix(b)).nnz == 0
    return np.array_equal(np.asarray(a), np.asarray(b))


def _h5ad_matches(path: Path, adata) -> bool:
    if not path.exists():
        return False
    try:
        from anndata import read_h5ad

        existing = read_h5ad(path)
        return (
            existing.shape == adata.shape
            and existing.obs_names.equals(adata.obs_names)
            and existing.var_names.equals(adata.var_names)
            and existing.obs.equals(adata.obs)
            and existing.var.equals(adata.var)
            and _array_equal(existing.X, adata.X)
            and set(existing.layers.keys()) == set(adata.layers.keys())
            and all(_array_equal(existing.layers[key], adata.layers[key]) for key in adata.layers.keys())
        )
    except Exception:
        return False
