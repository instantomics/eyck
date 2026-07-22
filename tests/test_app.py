from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from eyck.app import create_app
from eyck.cli import supervisor_loop
from eyck.discovery import discover_projects
from eyck.project import EyckProject, ProjectError

from conftest import sha256, write_project


@pytest.fixture
def client(project_root: Path):
    with TestClient(create_app([project_root])) as test_client:
        yield test_client


def test_typed_index_detail_points_and_features(client: TestClient):
    assert client.get("/healthz").json() == {"status": "ok"}
    index = client.get("/api/v1/annotations").json()
    assert index["projects"][0]["project_id"] == "cells"
    detail = client.get("/api/v1/annotations/cells").json()
    assert detail["identities"]["source_sha256"]
    assert detail["identities"]["observation_index_sha256"]
    assert detail["identities"]["feature_index_sha256"]
    assert detail["hierarchy_summary"] == {
        "explicit_decisions": 9,
        "materialized_decisions": 12,
        "supported_observations": 3,
        "entities": 3,
        "derived_ancestors": 0,
        "derived_intersections": 3,
    }
    assert detail["metadata_columns"] == ["sample", "score"]
    assert detail["modalities"] == [{"id": "crispr", "obsm": "X_crispr", "dimensions": 1}]

    points = client.get(detail["points_url"]).json()
    assert points == {
        "observation_ids": ["d1", "d2", "d3"],
        "coordinates": [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
        "metadata": {"sample": ["s1", "s1", "s2"], "score": [0.1, 0.2, 0.3]},
        "modalities": {"crispr": [0.20000000298023224, 0.4000000059604645, 0.6000000238418579]},
    }
    hits = client.get(detail["features_url"], params={"q": "gene"}).json()
    assert [hit["feature_id"] for hit in hits["features"]] == ["GeneA", "GeneB"]
    values = client.get(f'{detail["features_url"]}/1/values').json()
    assert values["values"] == [0.0, 2.0, 1.0]


def test_generated_initial_wide_and_long_entity_import(tmp_path: Path):
    wide_manifest = write_project(tmp_path / "wide", initial="wide")
    wide = EyckProject(discover_projects([wide_manifest])["cells"]).current_memberships()
    assert wide.source == "generated_initial"
    assert len(wide.rows) == 9
    assert {row.state for row in wide.rows} == {"present", "absent"}

    long_manifest = write_project(tmp_path / "long", initial="long")
    long = EyckProject(discover_projects([long_manifest])["cells"]).current_memberships()
    assert {(row.observation_id, row.entity_id) for row in long.rows} == {("d1", "left"), ("d1", "right")}
    assert all(row.state == "present" for row in long.rows)

    indexed_manifest = write_project(tmp_path / "indexed", initial="long-index")
    indexed = EyckProject(discover_projects([indexed_manifest])["cells"]).current_memberships()
    assert {(row.observation_id, row.entity_id) for row in indexed.rows} == {("d1", "left"), ("d1", "right")}


def test_entity_ids_allow_biological_separators(project_root: Path):
    project = EyckProject(discover_projects([project_root])["cells"])
    current = project.current_memberships()
    row = current.rows[0].model_copy(update={"entity_id": "hep:0/subunit"})
    project.validate_rows([row])

    invalid = row.model_copy(update={"entity_id": "hep:0\ninvalid"})
    with pytest.raises(ProjectError, match="control characters"):
        project.validate_rows([invalid])


def test_draft_save_revision_export_and_immutable_h5ad(project_root: Path, mutation_headers):
    h5ad = project_root / "nested" / "source" / "data" / "cells.h5ad"
    initial = project_root / "nested" / "source" / "output" / "annotation_projects" / "cells" / "generated_initial.parquet"
    source_before = sha256(h5ad)
    initial_before = sha256(initial)
    with TestClient(create_app([project_root])) as client:
        headers = mutation_headers(client)
        current = client.get("/api/v1/annotations/cells/memberships").json()
        rows = [
            {"support_id": "review", "observation_id": "d1", "entity_id": "left", "label_id": "a", "state": "present", "decision_view_id": "manual", "provenance": "explicit"},
            {"support_id": "review", "observation_id": "d1", "entity_id": "left", "label_id": "b", "state": "absent", "decision_view_id": "manual", "provenance": "explicit"},
            {"support_id": "review", "observation_id": "d1", "entity_id": "right", "label_id": "b", "state": "present", "decision_view_id": "manual", "provenance": "explicit"},
            {"support_id": "review", "observation_id": "d2", "entity_id": "0", "label_id": "cell", "state": "unreviewed", "decision_view_id": "manual", "provenance": "explicit"},
        ]
        saved = client.put(
            "/api/v1/annotations/cells/memberships",
            headers={**headers, "If-Match": current["revision"]},
            json={"rows": rows, "support_observation_ids": ["d1", "d2"]},
        )
        assert saved.status_code == 200
        document = saved.json()
        assert document["origin"] == "draft"
        assert len({(row["observation_id"], row["entity_id"]) for row in document["rows"]}) == 3

        stale = client.put(
            "/api/v1/annotations/cells/memberships",
            headers={**headers, "If-Match": current["revision"]},
            json={"rows": rows, "support_observation_ids": ["d1", "d2"]},
        )
        assert stale.status_code == 409
        exported = client.post(
            "/api/v1/annotations/cells/export",
            headers={**headers, "If-Match": document["revision"]},
        )
        assert exported.status_code == 200

    output = project_root / "nested" / "source" / "output" / "annotation_projects" / "cells"
    memberships = pd.read_parquet(output / "memberships.parquet")
    support = pd.read_parquet(output / "support.parquet")
    assert "unreviewed" not in set(memberships["state"])
    assert ((memberships["entity_id"] == "left") & (memberships["label_id"] == "cell") & (memberships["provenance"] == "derived_ancestor")).any()
    assert ((memberships["entity_id"] == "left") & (memberships["label_id"] == "a-b") & (memberships["state"] == "absent") & (memberships["provenance"] == "derived_intersection")).any()
    assert not support.duplicated(["support_id", "observation_id", "label_id"]).any()
    report = json.loads((output / "export_report.json").read_text(encoding="utf-8"))
    assert report["validation"] == {"errors": [], "valid": True}
    assert sha256(h5ad) == source_before
    assert sha256(initial) == initial_before


def test_failed_export_preserves_previous_files(project_root: Path, monkeypatch: pytest.MonkeyPatch):
    project = EyckProject(discover_projects([project_root])["cells"])
    revision = project.current_memberships().revision
    project.export(revision)
    targets = [project.spec.output_path / name for name in ("memberships.parquet", "support.parquet", "export_report.json")]
    before = [path.read_bytes() for path in targets]
    project.export(revision)
    assert [path.read_bytes() for path in targets] == before

    import eyck.project as project_module

    original = project_module.pq.write_table
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("writer failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(project_module.pq, "write_table", fail_second)
    with pytest.raises(RuntimeError, match="writer failed"):
        project.export(revision)
    assert [path.read_bytes() for path in targets] == before


def test_draft_write_rechecks_child_symlink_confinement(project_root: Path):
    project = EyckProject(discover_projects([project_root])["cells"])
    current = project.current_memberships()
    outside = project_root / "outside-drafts"
    outside.mkdir()
    (project.spec.output_path / "drafts").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProjectError, match="generated path"):
        project.put_memberships(current.revision, current.rows)


def test_security_restart_request_limit_and_assets(project_root: Path, mutation_headers):
    restarted: list[bool] = []
    with TestClient(create_app([project_root], restart_callback=lambda: restarted.append(True), max_request_bytes=256)) as client:
        current = client.get("/api/v1/annotations/cells/memberships").json()
        body = {"rows": [], "support_observation_ids": []}
        assert client.put("/api/v1/annotations/cells/memberships", json=body).status_code == 403
        assert client.put("/api/v1/annotations/cells/memberships", headers={"origin": "https://evil.example", "x-eyck-csrf": "x"}, json=body).status_code == 403
        headers = mutation_headers(client)
        assert client.post("/api/v1/restart", headers=headers).json() == {"restarting": True}
        assert restarted == [True]
        assert client.put("/api/v1/annotations/cells/memberships", headers={**headers, "If-Match": current["revision"]}, content=b"x" * 257).status_code == 413
        assert client.get("/").status_code == 200
        css = client.get("/assets/polyptich-ui.css")
        assert css.status_code == 200
        assert ".pt-button" in css.text
        assert client.get("/projects/cells").status_code == 200

    with TestClient(create_app([project_root])) as client:
        assert client.get("/api/v1/annotations", headers={"host": "evil.example"}).status_code == 400


def test_proxy_prefix_is_used_in_public_urls(project_root: Path, mutation_headers):
    with TestClient(create_app([project_root], proxy_prefix="/eyck/")) as client:
        detail = client.get("/api/v1/annotations/cells").json()
        assert detail["points_url"].startswith("/eyck/api/")
        assert detail["features_url"].startswith("/eyck/api/")
        assert detail["memberships_url"].startswith("/eyck/api/")
        assert detail["export_url"].startswith("/eyck/api/")

        frontend = client.get("/annotations/cells")
        assert 'content="/eyck"' in frontend.text
        assert 'src="/eyck/assets/' in frontend.text

        current = client.get("/api/v1/annotations/cells/memberships").json()
        exported = client.post(
            "/api/v1/annotations/cells/export",
            headers={**mutation_headers(client), "If-Match": current["revision"]},
        )
        assert exported.json()["report_url"] == "/eyck/api/v1/annotations/cells/export/report"


def test_supervisor_loop_restarts_only_when_requested():
    generations = 0

    def serve_once(event):
        nonlocal generations
        generations += 1
        if generations == 1:
            event.set()
            return True
        return False

    supervisor_loop(serve_once)
    assert generations == 2
