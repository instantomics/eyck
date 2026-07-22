from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


LABELS = {
    "schema_version": 1,
    "labels": [
        {"id": "cell", "name": "Cell", "parent_ids": []},
        {"id": "a", "name": "A", "parent_ids": ["cell"]},
        {"id": "b", "name": "B", "parent_ids": ["cell"]},
        {"id": "a-b", "name": "A and B", "parent_ids": ["a", "b"]},
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_project(root: Path, *, initial: str = "wide") -> Path:
    source = root / "nested" / "source"
    data_dir = source / "data"
    annotation_dir = source / "annotations" / "cells"
    output = source / "output" / "annotation_projects" / "cells"
    data_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    output.mkdir(parents=True)

    data = ad.AnnData(
        X=np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 4.0], [5.0, 1.0, 0.0]], dtype=np.float32),
        obs=pd.DataFrame({"sample": ["s1", "s1", "s2"], "score": [0.1, 0.2, 0.3]}, index=["d1", "d2", "d3"]),
        var=pd.DataFrame(index=["GeneA", "GeneB", "Other"]),
        obsm={
            "X_umap": np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
            "X_crispr": np.array([[0.2], [0.4], [0.6]], dtype=np.float32),
        },
    )
    h5ad = data_dir / "cells.h5ad"
    data.write_h5ad(h5ad)
    (annotation_dir / "labels.json").write_text(json.dumps(LABELS), encoding="utf-8")

    initial_path = output / "generated_initial.parquet"
    if initial == "wide":
        pd.DataFrame(
            {"cell": [1, 1, 1], "a": [1, 0, 0], "b": [0, 1, 0]},
            index=pd.Index(["d1", "d2", "d3"], name="droplet_id"),
        ).to_parquet(initial_path)
    else:
        long_frame = pd.DataFrame(
            {
                "droplet_id": ["d1", "d1", "d1"],
                "cell_type": ["cell", "a", "b"],
                "value": [1, 1, 1],
            },
            index=pd.Index(["left", "left", "right"], name="index"),
        )
        if initial == "long":
            long_frame["entity_id"] = long_frame.index
            long_frame = long_frame.reset_index(drop=True)
        long_frame.to_parquet(initial_path)

    manifest = source / "eyck.toml"
    manifest.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "",
                "[[projects]]",
                'id = "cells"',
                'title = "Cells"',
                'description = "Synthetic cells"',
                'dataset_id = "synthetic-v1"',
                'source_owner = "test-source"',
                'h5ad = "data/cells.h5ad"',
                'labels = "annotations/cells/labels.json"',
                'output = "output/annotation_projects/cells"',
                'expression_layer = "X"',
                'embeddings = [{id = "umap", obsm = "X_umap"}]',
                'metadata_columns = ["sample", "score"]',
                'modalities = [{id = "crispr", obsm = "X_crispr"}]',
                'initial_memberships = "output/annotation_projects/cells/generated_initial.parquet"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    write_project(tmp_path)
    return tmp_path


@pytest.fixture
def mutation_headers():
    def make(client):
        detail = client.get("/api/v1/annotations/cells").json()
        return {"origin": "http://testserver", "x-eyck-csrf": detail["csrf_token"]}

    return make
