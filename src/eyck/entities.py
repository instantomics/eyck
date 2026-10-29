"""Entity annotation tables: one row per annotated entity within a droplet.

A droplet holding several cells (a doublet, or a hepatocyte with its attached
partner) has one row per cell, indexed by ``droplet_id``, with an
``entity_id`` unique within the droplet, its ``cell_type`` and an integer
``value``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def membership_to_entities(membership: pd.DataFrame) -> pd.DataFrame:
    """Expand a droplet-by-cell-type count matrix into one row per entity."""
    rows = []
    for droplet_id, row in membership.fillna(0).astype(int).iterrows():
        for celltype, count in row.items():
            for entity_index in range(int(count)):
                rows.append(
                    {
                        "droplet_id": str(droplet_id),
                        "entity_id": f"{celltype}:{entity_index}",
                        "cell_type": celltype,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["entity_id", "cell_type"])
    return pd.DataFrame(rows).set_index("droplet_id")


def write_entity_annotations(entities: pd.DataFrame, path: Path) -> Path:
    """Write entities as a sorted parquet table, dropping zero-valued rows."""
    missing = {"entity_id", "cell_type"}.difference(entities.columns)
    if missing:
        raise ValueError(f"entity annotations are missing required columns: {sorted(missing)}")
    frame = entities.copy()
    if "value" not in frame.columns:
        frame["value"] = 1
    frame = frame[["entity_id", "cell_type", "value"]]
    frame.index = frame.index.astype(str)
    frame.index.name = "droplet_id"
    frame["entity_id"] = frame["entity_id"].astype(str)
    frame["cell_type"] = frame["cell_type"].astype(str)
    frame["value"] = frame["value"].astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[frame["value"] != 0].sort_index().to_parquet(path, compression="zstd")
    return path


__all__ = ["membership_to_entities", "write_entity_annotations"]
