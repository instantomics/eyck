from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from fastapi.testclient import TestClient
from scipy import sparse

from eyck.app import create_app
from eyck.discovery import canonical_sha256, discover_projects
from eyck.project import EyckProject

from conftest import sha256


API = "/api/v1/annotations/cells"


def mutation_headers(client: TestClient, revision: str) -> dict[str, str]:
    token = client.get(API).json()["csrf_token"]
    return {
        "origin": "http://testserver",
        "x-eyck-csrf": token,
        "If-Match": revision,
    }


def save_selection(
    client: TestClient,
    workspace: dict,
    selection_id: str,
    observations: list[str],
) -> dict:
    response = client.post(
        f"{API}/workspace/selections",
        headers=mutation_headers(client, workspace["revision"]),
        json={
            "id": selection_id,
            "name": selection_id,
            "zoom_id": "root",
            "definition": {
                "kind": "manual",
                "observation_ids": observations,
                "provenance": {
                    "method": "explicit-test-review",
                    "parameters": {"reviewer": "test"},
                    "input_ids": [],
                },
            },
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_root_boolean_selections_and_strict_child_zoom(project_root: Path):
    with TestClient(create_app([project_root])) as client:
        detail = client.get(API).json()
        assert detail["workspace_url"] == f"{API}/workspace"
        workspace = client.get(detail["workspace_url"]).json()
        root = workspace["zooms"][0]
        assert root["id"] == "root"
        assert root["parent_id"] is None
        assert root["observation_ids"] == ["d1", "d2", "d3"]
        assert root["observation_count"] == 3
        workspace_path = (
            project_root
            / "nested"
            / "source"
            / "output"
            / "annotation_projects"
            / "cells"
            / "workspace.json"
        )
        assert workspace_path.is_file()
        assert client.get(detail["workspace_url"]).json()["revision"] == workspace["revision"]

        workspace = save_selection(client, workspace, "left", ["d1", "d2"])
        workspace = save_selection(client, workspace, "right", ["d2", "d3"])
        combined = client.post(
            f"{API}/workspace/selections",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "left-only",
                "name": "Left excluding right",
                "zoom_id": "root",
                "definition": {
                    "kind": "boolean",
                    "operator": "exclusion",
                    "selection_ids": ["left", "right"],
                },
            },
        )
        assert combined.status_code == 200, combined.text
        workspace = combined.json()
        assert workspace["selections"][-1]["observation_ids"] == ["d1"]

        missing_embedding = client.post(
            f"{API}/workspace/zooms",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "missing-embedding",
                "name": "Missing embedding",
                "parent_id": "root",
                "recipe": {"operator": "union", "selection_ids": ["left"]},
            },
        )
        assert missing_embedding.status_code == 422

        child = client.post(
            f"{API}/workspace/zooms",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "focused",
                "name": "Focused",
                "parent_id": "root",
                "parent_embedding_id": "umap",
                "recipe": {"operator": "union", "selection_ids": ["left"]},
            },
        )
        assert child.status_code == 200, child.text
        workspace = child.json()
        focused = workspace["zooms"][-1]
        assert focused["observation_ids"] == ["d1", "d2"]
        assert focused["initial_embedding_id"] == root["initial_embedding_id"]
        focused_points = client.get(
            f"{API}/workspace/zooms/focused/points", params={"embedding": "umap"}
        ).json()
        assert focused_points["observation_ids"] == ["d1", "d2"]
        assert focused_points["modalities"]["crispr"] == [
            0.20000000298023224,
            0.4000000059604645,
        ]

        not_strict = client.post(
            f"{API}/workspace/zooms",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "not-strict",
                "name": "Not strict",
                "parent_id": "root",
                "parent_embedding_id": "umap",
                "recipe": {"operator": "union", "selection_ids": ["left", "right"]},
            },
        )
        assert not_strict.status_code == 422
        assert "strict subset" in not_strict.json()["detail"]


def test_import_local_analysis_marker_shape_cutoff_and_h5ad_immutability(project_root: Path):
    h5ad = project_root / "nested" / "source" / "data" / "cells.h5ad"
    source_before = sha256(h5ad)
    with TestClient(create_app([project_root])) as client:
        workspace = client.get(f"{API}/workspace").json()
        imported = client.post(
            f"{API}/workspace/clusterings/import",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "sample-clusters",
                "name": "Sample clusters",
                "zoom_id": "root",
                "metadata_column": "sample",
            },
        )
        assert imported.status_code == 200, imported.text
        workspace = imported.json()
        clustering = workspace["clusterings"][0]
        assert clustering["cluster_count"] == 2
        assert clustering["identity"]
        output = h5ad.parents[1] / "output" / "annotation_projects" / "cells"
        assert (output / clustering["cache_path"]).is_file()
        clustering_result = client.get(
            f"{API}/workspace/clusterings/sample-clusters/result"
        ).json()
        assert clustering_result["observation_ids"] == ["d1", "d2", "d3"]
        assert clustering_result["cluster_ids"] == ["s1", "s1", "s2"]

        marker = client.post(
            f"{API}/workspace/marker-programs",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "gene-a",
                "name": "Gene A",
                "zoom_id": "root",
                "clustering_id": "sample-clusters",
                "markers": ["GeneA"],
                "ctrl_size": 1,
                "n_bins": 2,
            },
        )
        assert marker.status_code == 200, marker.text
        result = marker.json()
        assert len(result["per_cell_scores"]) == 3
        assert len(result["cluster_mean_by_cell"]) == 3
        assert sum(row["cell_count"] for row in result["cluster_table"]) == 3
        assert result["program"]["marker_ids"] == ["GeneA"]
        assert result["program"]["parameters"]["expression_source"] == "X"
        assert result["program"]["parameters"]["use_raw"] is None
        assert result["program"]["provenance"]["expression_source"] == "X"
        assert (output / result["program"]["cache_path"]).is_file()

        workspace = client.get(f"{API}/workspace").json()
        cutoff = client.post(
            f"{API}/workspace/selections",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "positive-clusters",
                "name": "Positive clusters",
                "zoom_id": "root",
                "definition": {
                    "kind": "marker_cutoff",
                    "marker_program_id": "gene-a",
                    "comparator": ">=",
                    "cutoff": min(row["mean_score"] for row in result["cluster_table"]),
                },
            },
        )
        assert cutoff.status_code == 200, cutoff.text
        assert cutoff.json()["selections"][-1]["observation_count"] == 3

        workspace = cutoff.json()
        local = client.post(
            f"{API}/workspace/analyses/local",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "embedding_id": "local-umap",
                "embedding_name": "Local UMAP",
                "zoom_id": "root",
                "clustering_outputs": [
                    {
                        "id": "local-clusters",
                        "name": "Local clusters",
                        "resolution": 1.0,
                    },
                    {
                        "id": "local-clusters-fine",
                        "name": "Local clusters fine",
                        "resolution": 2.0,
                    },
                ],
                "profile": "scanpy_default",
                "n_neighbors": 2,
            },
        )
        assert local.status_code == 200, local.text
        local_workspace = local.json()
        assert any(item["id"] == "local-clusters" for item in local_workspace["clusterings"])
        assert any(
            item["id"] == "local-clusters-fine" for item in local_workspace["clusterings"]
        )
        assert any(item["id"] == "local-umap" for item in local_workspace["embeddings"])
        local_descriptors = [
            item
            for item in local_workspace["clusterings"]
            if item["id"] in {"local-clusters", "local-clusters-fine"}
        ]
        assert len({item["provenance"]["analysis_identity"] for item in local_descriptors}) == 1
        assert len({item["provenance"]["embedding_identity"] for item in local_descriptors}) == 1
        assert len({item["identity"] for item in local_descriptors}) == 2
        assert {item["parameters"]["profile"] for item in local_descriptors} == {
            "scanpy_default"
        }
        assert {item["parameters"]["small_dataset_fallback"] for item in local_descriptors} == {
            True
        }
        computed_clusters = client.get(
            f"{API}/workspace/clusterings/local-clusters/result"
        ).json()
        assert computed_clusters["observation_ids"] == ["d1", "d2", "d3"]
        assert len(computed_clusters["cluster_ids"]) == 3
        local_points = client.get(
            f"{API}/workspace/zooms/root/points", params={"embedding": "local-umap"}
        )
        assert len(local_points.json()["coordinates"]) == 3

        fast_defaults = client.post(
            f"{API}/workspace/analyses/local",
            headers=mutation_headers(client, local_workspace["revision"]),
            json={
                "embedding_id": "fast-default-umap",
                "embedding_name": "Fast defaults UMAP",
                "zoom_id": "root",
                "clustering_outputs": [
                    {
                        "id": "fast-default-clusters",
                        "name": "Fast default clusters",
                        "resolution": 1.0,
                    }
                ],
                "profile": "scanpy_fast",
            },
        )
        assert fast_defaults.status_code == 200, fast_defaults.text
        descriptor = next(
            item
            for item in fast_defaults.json()["clusterings"]
            if item["id"] == "fast-default-clusters"
        )
        assert descriptor["parameters"]["preprocessing"]["highly_variable_genes"][
            "n_top_genes"
        ] == 3000
        assert descriptor["parameters"]["preprocessing"]["pca"]["n_comps"] == 50
        assert descriptor["parameters"]["graph"]["n_pcs"] == 50

    assert sha256(h5ad) == source_before


def test_marker_program_uses_raw_and_resolves_raw_features(project_root: Path):
    h5ad = project_root / "nested" / "source" / "data" / "cells.h5ad"
    data = ad.read_h5ad(h5ad)
    raw_values = np.asarray(
        [[100.0, 100.0, 1.0], [20.0, 120.0, 0.0], [60.0, 110.0, 2.0]],
        dtype=np.float32,
    )
    raw_var = data.var.copy()
    raw_var.index = ["GeneA", "GeneB", "RawOnly"]
    raw_var["symbol"] = ["raw-a", "raw-b", "raw-only"]
    data.raw = ad.AnnData(X=raw_values, obs=data.obs.copy(), var=raw_var)
    data.write_h5ad(h5ad)

    reference = ad.read_h5ad(h5ad)
    sc.tl.score_genes(
        reference,
        gene_list=["GeneA"],
        ctrl_size=1,
        n_bins=2,
        score_name="expected_raw",
        random_state=0,
    )
    expected_raw = reference.obs["expected_raw"].to_numpy()
    x_reference = ad.AnnData(
        X=reference.X.copy(),
        obs=reference.obs.copy(),
        var=reference.var.copy(),
    )
    sc.tl.score_genes(
        x_reference,
        gene_list=["GeneA"],
        ctrl_size=1,
        n_bins=2,
        score_name="expected_x",
        random_state=0,
        use_raw=False,
    )
    expected_x = x_reference.obs["expected_x"].to_numpy()
    assert not np.allclose(expected_raw, expected_x)

    with TestClient(create_app([project_root])) as client:
        workspace = client.get(f"{API}/workspace").json()
        workspace = client.post(
            f"{API}/workspace/clusterings/import",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "samples",
                "name": "Samples",
                "zoom_id": "root",
                "metadata_column": "sample",
            },
        ).json()
        marker = client.post(
            f"{API}/workspace/marker-programs",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "raw-program",
                "name": "Raw program",
                "zoom_id": "root",
                "clustering_id": "samples",
                "markers": ["raw-a"],
                "ctrl_size": 1,
                "n_bins": 2,
            },
        )
        assert marker.status_code == 200, marker.text
        result = marker.json()
        assert result["program"]["marker_ids"] == ["GeneA"]
        assert result["program"]["parameters"]["expression_source"] == "raw"
        assert result["program"]["parameters"]["use_raw"] is None
        assert result["program"]["provenance"]["expression_source"] == "raw"
        assert np.allclose(result["per_cell_scores"], expected_raw)
        assert not np.allclose(result["per_cell_scores"], expected_x)

        workspace = client.get(f"{API}/workspace").json()
        missing_from_raw = client.post(
            f"{API}/workspace/marker-programs",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "x-only-program",
                "name": "X-only marker",
                "zoom_id": "root",
                "clustering_id": "samples",
                "markers": ["Other"],
            },
        )
        assert missing_from_raw.status_code == 422
        assert "missing" in missing_from_raw.json()["detail"]


def test_sparse_backed_raw_subset_materializes_without_fancy_indexing(project_root: Path):
    h5ad = project_root / "nested" / "source" / "data" / "cells.h5ad"
    data = ad.read_h5ad(h5ad)
    data.X = sparse.csr_matrix(data.X)
    raw_var = data.var.copy()
    raw_var.index = ["GeneA", "GeneB", "RawOnly"]
    raw_var["symbol"] = ["raw-a", "raw-b", "raw-only"]
    data.raw = ad.AnnData(
        X=sparse.csr_matrix(
            np.asarray(
                [[100.0, 100.0, 1.0], [20.0, 120.0, 0.0], [60.0, 110.0, 2.0]],
                dtype=np.float32,
            )
        ),
        obs=data.obs.copy(),
        var=raw_var,
    )
    data.write_h5ad(h5ad)
    source_before = sha256(h5ad)
    backed = ad.read_h5ad(h5ad, backed="r")
    try:
        assert "csr" in type(backed.raw.X).__name__.lower()
    finally:
        backed.file.close()

    with TestClient(create_app([project_root])) as client:
        workspace = client.get(f"{API}/workspace").json()
        workspace = save_selection(client, workspace, "noncontiguous", ["d1", "d3"])
        zoomed = client.post(
            f"{API}/workspace/zooms",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "subset",
                "name": "Noncontiguous subset",
                "parent_id": "root",
                "parent_embedding_id": "umap",
                "recipe": {"operator": "union", "selection_ids": ["noncontiguous"]},
            },
        )
        assert zoomed.status_code == 200, zoomed.text
        workspace = zoomed.json()
        imported = client.post(
            f"{API}/workspace/clusterings/external",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "subset-clusters",
                "name": "Subset clusters",
                "zoom_id": "subset",
                "observation_ids": ["d1", "d3"],
                "cluster_ids": ["0", "0"],
                "implementation": "regression-test.external",
                "parameters": {},
                "provenance": {"purpose": "backed sparse regression"},
            },
        )
        assert imported.status_code == 200, imported.text
        workspace = imported.json()
        marker = client.post(
            f"{API}/workspace/marker-programs",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "subset-marker",
                "name": "Subset marker",
                "zoom_id": "subset",
                "clustering_id": "subset-clusters",
                "markers": ["raw-a"],
                "ctrl_size": 1,
                "n_bins": 2,
            },
        )
        assert marker.status_code == 200, marker.text
        assert marker.json()["program"]["parameters"]["expression_source"] == "raw"
        assert marker.json()["observation_ids"] == ["d1", "d3"]

        workspace = client.get(f"{API}/workspace").json()
        analysis = client.post(
            f"{API}/workspace/analyses/local",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "embedding_id": "subset-local-umap",
                "embedding_name": "Subset local UMAP",
                "zoom_id": "subset",
                "clustering_outputs": [
                    {
                        "id": "subset-local-clusters",
                        "name": "Subset local clusters",
                        "resolution": 1.0,
                    }
                ],
                "profile": "scanpy_default",
                "n_neighbors": 1,
            },
        )
        assert analysis.status_code == 200, analysis.text
        project = client.app.state.projects["cells"]
        assert set(project._matrix_cache) == {"raw", "X"}

    assert sha256(h5ad) == source_before


def test_scanpy_fast_profile_records_and_reuses_one_graph(project_root: Path):
    h5ad = project_root / "nested" / "source" / "data" / "cells.h5ad"
    rng = np.random.default_rng(4)
    counts = rng.poisson(4, size=(6, 8)).astype(np.float32)
    data = ad.AnnData(
        X=np.log1p(counts),
        obs=ad.read_h5ad(h5ad).obs.reindex([f"d{index}" for index in range(1, 7)]),
        var={"symbol": [f"G{index}" for index in range(8)]},
        obsm={
            "X_umap": rng.normal(size=(6, 2)).astype(np.float32),
            "X_crispr": rng.normal(size=(6, 1)).astype(np.float32),
        },
    )
    data.obs["sample"] = ["s1", "s1", "s1", "s2", "s2", "s2"]
    data.obs["score"] = np.linspace(0.1, 0.6, 6)
    data.var_names = [f"Gene{index}" for index in range(8)]
    data.layers["counts"] = counts
    data.write_h5ad(h5ad)

    with TestClient(create_app([project_root])) as client:
        workspace = client.get(f"{API}/workspace").json()
        duplicate = client.post(
            f"{API}/workspace/analyses/local",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "embedding_id": "fast-umap",
                "embedding_name": "Fast UMAP",
                "zoom_id": "root",
                "clustering_outputs": [
                    {"id": "duplicate", "name": "One", "resolution": 0.5},
                    {"id": "duplicate", "name": "Two", "resolution": 1.0},
                ],
                "profile": "scanpy_fast",
            },
        )
        assert duplicate.status_code == 422

        response = client.post(
            f"{API}/workspace/analyses/local",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "embedding_id": "fast-umap",
                "embedding_name": "Fast UMAP",
                "zoom_id": "root",
                "clustering_outputs": [
                    {"id": "leiden-05", "name": "Leiden 0.5", "resolution": 0.5},
                    {"id": "leiden-10", "name": "Leiden 1.0", "resolution": 1.0},
                ],
                "profile": "scanpy_fast",
                "use_counts": True,
                "n_neighbors": 3,
                "n_pcs": 3,
                "n_top_genes": 4,
                "n_comps": 3,
                "random_state": 7,
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()
        outputs = [
            item
            for item in result["clusterings"]
            if item["id"] in {"leiden-05", "leiden-10"}
        ]
        assert len(outputs) == 2
        assert len({item["provenance"]["analysis_identity"] for item in outputs}) == 1
        assert len({item["provenance"]["embedding_identity"] for item in outputs}) == 1
        assert len({item["identity"] for item in outputs}) == 2
        parameters = outputs[0]["parameters"]
        assert parameters["profile"] == "scanpy_fast"
        assert parameters["use_counts"] is True
        assert parameters["preprocessing"]["counts"]["normalize_total"]["target_sum"] == 10000.0
        assert parameters["scanpy_version"]
        assert parameters["preprocessing"]["highly_variable_genes"]["n_top_genes"] == 4
        assert parameters["preprocessing"]["pca"]["n_comps"] == 3
        assert parameters["preprocessing"]["pca"]["mask_var"] == "highly_variable"
        assert parameters["graph"]["n_neighbors"] == 3
        assert parameters["graph"]["n_pcs"] == 3
        assert parameters["umap"]["random_state"] == 7
        assert parameters["leiden"]["flavor"] == "igraph"
        assert parameters["leiden"]["n_iterations"] == 2
        assert [
            (item["id"], item["resolution"])
            for item in parameters["leiden"]["outputs"]
        ] == [("leiden-05", 0.5), ("leiden-10", 1.0)]
        for clustering_id in ("leiden-05", "leiden-10"):
            assignments = client.get(
                f"{API}/workspace/clusterings/{clustering_id}/result"
            ).json()
            assert assignments["observation_ids"] == [f"d{index}" for index in range(1, 7)]
            assert len(assignments["cluster_ids"]) == 6


def test_external_clustering_import_requires_exact_zoom_order(project_root: Path):
    with TestClient(create_app([project_root], proxy_prefix="/eyck")) as client:
        detail = client.get(API).json()
        assert detail["workspace_url"] == f"/eyck{API}/workspace"
        workspace = client.get(f"{API}/workspace").json()
        body = {
            "id": "script-leiden",
            "name": "Script Leiden",
            "zoom_id": "root",
            "observation_ids": ["d1", "d2", "d3"],
            "cluster_ids": ["0", "1", "0"],
            "implementation": "BuglerLamb.annotate_EC_KC_doublits:leiden",
            "parameters": {"resolution": 0.2, "random_state": 0},
            "provenance": {
                "script": "annotate_EC_KC_doublits.py",
                "operator": "Leiden on sliced inherited graph",
            },
        }
        imported = client.post(
            f"{API}/workspace/clusterings/external",
            headers=mutation_headers(client, workspace["revision"]),
            json=body,
        )
        assert imported.status_code == 200, imported.text
        saved_workspace = imported.json()
        descriptor = next(
            item for item in saved_workspace["clusterings"] if item["id"] == "script-leiden"
        )
        assert descriptor["identity"]
        assert descriptor["implementation"] == body["implementation"]
        assert descriptor["parameters"] == body["parameters"]
        assert descriptor["provenance"]["operator_origin"] == "external"
        assert descriptor["provenance"]["script"] == "annotate_EC_KC_doublits.py"
        result = client.get(
            f"{API}/workspace/clusterings/script-leiden/result"
        ).json()
        assert result["observation_ids"] == body["observation_ids"]
        assert result["cluster_ids"] == body["cluster_ids"]

        mismatch = {
            **body,
            "id": "wrong-order",
            "observation_ids": ["d2", "d1", "d3"],
        }
        rejected = client.post(
            f"{API}/workspace/clusterings/external",
            headers=mutation_headers(client, saved_workspace["revision"]),
            json=mismatch,
        )
        assert rejected.status_code == 422
        assert "exactly match" in rejected.json()["detail"]


def test_marker_resolution_blocks_missing_and_ambiguous(project_root: Path):
    h5ad = project_root / "nested" / "source" / "data" / "cells.h5ad"
    data = ad.read_h5ad(h5ad)
    data.var["symbol"] = ["duplicate", "duplicate", "other"]
    data.write_h5ad(h5ad)
    with TestClient(create_app([project_root])) as client:
        workspace = client.get(f"{API}/workspace").json()
        workspace = client.post(
            f"{API}/workspace/clusterings/import",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "samples",
                "name": "Samples",
                "zoom_id": "root",
                "metadata_column": "sample",
            },
        ).json()
        for marker, message in (("missing", "missing"), ("duplicate", "ambiguous")):
            response = client.post(
                f"{API}/workspace/marker-programs",
                headers=mutation_headers(client, workspace["revision"]),
                json={
                    "id": f"program-{marker}",
                    "name": marker,
                    "zoom_id": "root",
                    "clustering_id": "samples",
                    "markers": [marker],
                },
            )
            assert response.status_code == 422
            assert message in response.json()["detail"]


def test_lasso_is_replayed_from_embedding_coordinates(project_root: Path):
    polygon = [[-1.0, 0.0], [1.0, 0.0], [1.0, 2.0], [-1.0, 2.0]]
    with TestClient(create_app([project_root])) as client:
        workspace = client.get(f"{API}/workspace").json()
        body = {
            "id": "lasso",
            "name": "Lasso",
            "zoom_id": "root",
            "definition": {
                "kind": "lasso",
                "observation_ids": ["d2"],
                "embedding_id": "umap",
                "polygon": polygon,
                "provenance": {
                    "method": "data-coordinate-polygon",
                    "parameters": {},
                    "input_ids": ["umap"],
                },
            },
        }
        mismatch = client.post(
            f"{API}/workspace/selections",
            headers=mutation_headers(client, workspace["revision"]),
            json=body,
        )
        assert mismatch.status_code == 422
        assert "polygon replay" in mismatch.json()["detail"]

        body["definition"]["observation_ids"] = ["d1"]
        saved = client.post(
            f"{API}/workspace/selections",
            headers=mutation_headers(client, workspace["revision"]),
            json=body,
        )
        assert saved.status_code == 200, saved.text
        assert saved.json()["selections"][-1]["observation_ids"] == ["d1"]


def test_membership_links_and_derived_intersections_are_validated(project_root: Path):
    with TestClient(create_app([project_root])) as client:
        workspace = client.get(f"{API}/workspace").json()
        workspace = save_selection(client, workspace, "support", ["d1", "d2"])
        workspace = client.post(
            f"{API}/workspace/zooms",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "child",
                "name": "Child",
                "parent_id": "root",
                "parent_embedding_id": "umap",
                "recipe": {"operator": "union", "selection_ids": ["support"]},
            },
        ).json()
        membership = client.get(f"{API}/memberships").json()
        base_row = {
            "support_id": "support",
            "observation_id": "d1",
            "entity_id": "0",
            "label_id": "cell",
            "state": "present",
            "decision_view_id": "root",
            "provenance": "explicit",
        }
        invalid_rows = [
            ({**base_row, "selection_id": "support"}, "provided together"),
            (
                {**base_row, "selection_id": "support", "zoom_id": "child"},
                "does not belong",
            ),
            (
                {
                    **base_row,
                    "observation_id": "d3",
                    "selection_id": "support",
                    "zoom_id": "root",
                },
                "not in the linked selection",
            ),
            ({**base_row, "label_id": "a-b"}, "derived intersection"),
        ]
        for row, message in invalid_rows:
            response = client.put(
                f"{API}/memberships",
                headers=mutation_headers(client, membership["revision"]),
                json={"rows": [row]},
            )
            assert response.status_code == 422
            assert message in response.json()["detail"]

        linked_row = {**base_row, "selection_id": "support", "zoom_id": "root"}
        linked = client.put(
            f"{API}/memberships",
            headers=mutation_headers(client, membership["revision"]),
            json={"rows": [linked_row]},
        )
        assert linked.status_code == 200, linked.text

        labels = client.get(f"{API}/labels").json()
        changed = [dict(item) for item in labels["labels"]]
        b_index = next(index for index, item in enumerate(changed) if item["id"] == "b")
        changed[b_index] = {**changed[b_index], "parent_ids": []}
        proposal = {"labels": changed}
        impact = client.post(
            f"{API}/labels/impact",
            headers=mutation_headers(client, labels["revision"]),
            json=proposal,
        ).json()
        assert impact["selection_ids"] == ["support"]
        assert impact["membership_decision_count"] == 1
        proposal.update(
            {
                "confirm_cascade": True,
                "expected_membership_revision": impact["membership_revision"],
                "expected_impact_sha256": impact["impact_sha256"],
            }
        )
        changed_membership = client.put(
            f"{API}/memberships",
            headers=mutation_headers(client, linked.json()["revision"]),
            json={"rows": [{**linked_row, "state": "absent"}]},
        )
        assert changed_membership.status_code == 200
        stale = client.put(
            f"{API}/labels",
            headers=mutation_headers(client, labels["revision"]),
            json=proposal,
        )
        assert stale.status_code == 409


def test_transaction_journal_recovers_intermediate_label_membership_state(project_root: Path):
    project = EyckProject(discover_projects([project_root])["cells"])
    project.current_workspace()
    labels = project.current_labels().model_dump(mode="json")
    labels["labels"][0]["name"] = "Recovered cell"
    membership_update = {
        "schema_version": 1,
        "project_id": "cells",
        "revision": "recovered-on-read",
        "rows": [],
    }
    updates = {"labels": labels, "memberships": membership_update}
    journal = {
        "schema_version": 1,
        "project_id": "cells",
        "generation": canonical_sha256(updates),
        "updates": updates,
    }
    project.transaction_path.write_text(json.dumps(journal), encoding="utf-8")
    project.spec.labels_path.write_text(json.dumps(labels), encoding="utf-8")

    recovered_memberships = project.current_memberships()
    assert recovered_memberships.rows == []
    assert project.current_labels().labels[0].name == "Recovered cell"
    assert not project.transaction_path.exists()


def test_label_dag_revision_cycle_and_confirmed_cascade(project_root: Path):
    labels_path = project_root / "nested" / "source" / "annotations" / "cells" / "labels.json"
    with TestClient(create_app([project_root])) as client:
        state = client.get(f"{API}/labels").json()
        labels = state["labels"]
        cyclic = [dict(item) for item in labels]
        cyclic[0] = {**cyclic[0], "parent_ids": ["a"]}
        cycle = client.post(
            f"{API}/labels/impact",
            headers=mutation_headers(client, state["revision"]),
            json={"labels": cyclic},
        )
        assert cycle.status_code == 422
        assert "cycle" in cycle.json()["detail"]

        proposed = {
            "labels": [
                *labels,
                {
                    "id": "new-label",
                    "name": "New label",
                    "description": "Editable identity",
                    "ontology_ids": [],
                    "parent_ids": [],
                },
            ]
        }
        proposed_preview = client.post(
            f"{API}/labels/impact",
            headers=mutation_headers(client, state["revision"]),
            json=proposed,
        ).json()
        assert proposed_preview["membership_decision_count"] == 0
        proposed.update(
            {
                "expected_membership_revision": proposed_preview["membership_revision"],
                "expected_impact_sha256": proposed_preview["impact_sha256"],
            }
        )
        added = client.put(
            f"{API}/labels",
            headers=mutation_headers(client, state["revision"]),
            json=proposed,
        )
        assert added.status_code == 200, added.text
        new_state = added.json()
        assert "new-label" in {item["id"] for item in new_state["labels"]}
        assert len(client.get(f"{API}/memberships").json()["rows"]) == 9
        assert client.get(API).json()["identities"]["labels_revision"] == new_state["revision"]
        stale = client.put(
            f"{API}/labels",
            headers=mutation_headers(client, state["revision"]),
            json=proposed,
        )
        assert stale.status_code == 409

        cascade_request = {
            "labels": new_state["labels"],
            "delete_label_ids": ["a"],
        }
        preview = client.post(
            f"{API}/labels/impact",
            headers=mutation_headers(client, new_state["revision"]),
            json=cascade_request,
        ).json()
        assert preview["labels_revision"] == new_state["revision"]
        assert preview["proposed_revision"] != preview["labels_revision"]
        assert preview["membership_revision"]
        assert preview["impact_sha256"]
        assert preview["removed_label_ids"] == ["a", "a-b"]
        assert preview["membership_decision_count"] == 3
        cascade_request.update(
            {
                "expected_membership_revision": preview["membership_revision"],
                "expected_impact_sha256": preview["impact_sha256"],
            }
        )
        unconfirmed = client.put(
            f"{API}/labels",
            headers=mutation_headers(client, new_state["revision"]),
            json=cascade_request,
        )
        assert unconfirmed.status_code == 422
        cascade_request["confirm_cascade"] = True
        deleted = client.put(
            f"{API}/labels",
            headers=mutation_headers(client, new_state["revision"]),
            json=cascade_request,
        )
        assert deleted.status_code == 200, deleted.text
        assert not {"a", "a-b"} & {item["id"] for item in deleted.json()["labels"]}
        memberships = client.get(f"{API}/memberships").json()["rows"]
        assert not {"a", "a-b"} & {item["label_id"] for item in memberships}

        semantic_labels = [dict(item) for item in deleted.json()["labels"]]
        b_index = next(index for index, item in enumerate(semantic_labels) if item["id"] == "b")
        semantic_labels[b_index] = {**semantic_labels[b_index], "parent_ids": []}
        semantic_request = {"labels": semantic_labels}
        semantic_preview = client.post(
            f"{API}/labels/impact",
            headers=mutation_headers(client, deleted.json()["revision"]),
            json=semantic_request,
        ).json()
        assert semantic_preview["membership_decision_count"] == len(memberships)
        semantic_request.update(
            {
                "expected_membership_revision": semantic_preview["membership_revision"],
                "expected_impact_sha256": semantic_preview["impact_sha256"],
            }
        )
        rejected_semantic = client.put(
            f"{API}/labels",
            headers=mutation_headers(client, deleted.json()["revision"]),
            json=semantic_request,
        )
        assert rejected_semantic.status_code == 422
        semantic_request["confirm_cascade"] = True
        accepted_semantic = client.put(
            f"{API}/labels",
            headers=mutation_headers(client, deleted.json()["revision"]),
            json=semantic_request,
        )
        assert accepted_semantic.status_code == 200, accepted_semantic.text
        assert client.get(f"{API}/memberships").json()["rows"] == []

    stored = json.loads(labels_path.read_text(encoding="utf-8"))
    assert not {"a", "a-b"} & {item["id"] for item in stored["labels"]}


def test_dependency_deletion_preview_and_exact_cascade(project_root: Path):
    with TestClient(create_app([project_root])) as client:
        workspace = client.get(f"{API}/workspace").json()
        workspace = save_selection(client, workspace, "base", ["d1", "d2"])
        workspace = save_selection(client, workspace, "other", ["d2"])
        response = client.post(
            f"{API}/workspace/selections",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "combined",
                "name": "Combined",
                "zoom_id": "root",
                "definition": {
                    "kind": "boolean",
                    "operator": "union",
                    "selection_ids": ["base", "other"],
                },
            },
        )
        workspace = response.json()
        workspace = client.post(
            f"{API}/workspace/zooms",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "id": "child",
                "name": "Child",
                "parent_id": "root",
                "parent_embedding_id": "umap",
                "recipe": {"operator": "union", "selection_ids": ["combined"]},
            },
        ).json()

        memberships = client.get(f"{API}/memberships").json()
        linked = client.put(
            f"{API}/memberships",
            headers=mutation_headers(client, memberships["revision"]),
            json={
                "rows": [
                    {
                        "support_id": "base",
                        "observation_id": "d1",
                        "entity_id": "0",
                        "label_id": "cell",
                        "state": "present",
                        "decision_view_id": "child",
                        "provenance": "explicit",
                        "selection_id": "base",
                        "zoom_id": "root",
                    }
                ]
            },
        )
        assert linked.status_code == 200, linked.text

        preview = client.post(
            f"{API}/workspace/deletions/preview",
            headers=mutation_headers(client, workspace["revision"]),
            json={"kind": "selection", "id": "base"},
        )
        assert preview.status_code == 200, preview.text
        impact = preview.json()
        assert {(item["kind"], item["id"]) for item in impact["removed"]} == {
            ("selection", "base"),
            ("selection", "combined"),
            ("zoom", "child"),
        }
        assert impact["membership_decision_count"] == 1
        assert impact["workspace_revision"] == workspace["revision"]
        assert impact["membership_revision"] == linked.json()["revision"]
        assert impact["impact_sha256"]
        changed_row = {**linked.json()["rows"][0], "state": "absent"}
        changed_memberships = client.put(
            f"{API}/memberships",
            headers=mutation_headers(client, linked.json()["revision"]),
            json={"rows": [changed_row]},
        )
        assert changed_memberships.status_code == 200, changed_memberships.text
        stale_delete = client.post(
            f"{API}/workspace/deletions",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "kind": "selection",
                "id": "base",
                "confirmed_removed": impact["removed"],
                "expected_membership_revision": impact["membership_revision"],
                "impact_sha256": impact["impact_sha256"],
            },
        )
        assert stale_delete.status_code == 409
        impact = client.post(
            f"{API}/workspace/deletions/preview",
            headers=mutation_headers(client, workspace["revision"]),
            json={"kind": "selection", "id": "base"},
        ).json()
        deleted = client.post(
            f"{API}/workspace/deletions",
            headers=mutation_headers(client, workspace["revision"]),
            json={
                "kind": "selection",
                "id": "base",
                "confirmed_removed": impact["removed"],
                "expected_membership_revision": impact["membership_revision"],
                "impact_sha256": impact["impact_sha256"],
            },
        )
        assert deleted.status_code == 200, deleted.text
        result = deleted.json()
        assert {item["id"] for item in result["zooms"]} == {"root"}
        assert {item["id"] for item in result["selections"]} == {"other"}
        assert client.get(f"{API}/memberships").json()["rows"] == []
