"""Derived views: extra 2D embeddings kept as files beside the project's H5AD.

A project may declare a ``views`` directory. Each ``<view_id>.parquet`` in it
has an ``observation_id`` column and ``x`` and ``y`` coordinates for any subset
of the project's observations, so a view can embed a subpopulation or the whole
population after different preprocessing. Cells absent from a view have no
coordinates there. The optional ``eyck_view`` schema metadata is a JSON object
with a ``title`` and a ``description`` of how the view was made.

Views are read on request, so producers can add or replace them without
rewriting the H5AD or restarting Eyck.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .discovery import SAFE_ID
from .models import ViewDescriptor


class ViewError(ValueError):
    """A view file is invalid for its project."""


def view_paths(directory: Path | None) -> dict[str, Path]:
    if directory is None or not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(directory.glob("*.parquet"))
        if SAFE_ID.fullmatch(path.stem)
    }


def read_view_descriptor(view_id: str, path: Path) -> ViewDescriptor:
    try:
        file = pq.ParquetFile(path)
        raw = (file.schema_arrow.metadata or {}).get(b"eyck_view", b"{}")
        info = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ViewError(f"invalid view {view_id}: {exc}") from exc
    return ViewDescriptor(
        id=view_id,
        title=str(info.get("title", view_id)),
        description=str(info.get("description", "")),
        observations=file.metadata.num_rows,
        source="view",
    )


def read_view_coordinates(
    view_id: str, path: Path, positions: dict[str, int]
) -> np.ndarray:
    """Coordinates in project observation order; NaN for cells outside the view."""
    try:
        table = pq.read_table(path, columns=["observation_id", "x", "y"])
    except (OSError, ValueError, KeyError) as exc:
        raise ViewError(f"invalid view {view_id}: {exc}") from exc
    observations = table.column("observation_id").to_pylist()
    xy = np.column_stack(
        [table.column("x").to_numpy(), table.column("y").to_numpy()]
    ).astype(float)
    if len(set(observations)) != len(observations):
        raise ViewError(f"view {view_id} repeats observations")
    unknown = [item for item in observations if item not in positions]
    if unknown:
        raise ViewError(
            f"view {view_id} has {len(unknown)} unknown observations, e.g. {unknown[0]}"
        )
    if not np.isfinite(xy).all():
        raise ViewError(f"view {view_id} has non-finite coordinates")
    coordinates = np.full((len(positions), 2), np.nan)
    coordinates[[positions[item] for item in observations]] = xy
    return coordinates


__all__ = [
    "ViewError",
    "read_view_coordinates",
    "read_view_descriptor",
    "view_paths",
]
