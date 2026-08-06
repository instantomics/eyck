from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import LABELS, write_project
from eyck.discovery import DiscoveryError, discover_projects, load_labels
from eyck.project import EyckProject


def test_recursive_discovery_resolves_confined_real_paths(project_root: Path):
    projects = discover_projects([project_root])
    project = projects["cells"]
    assert project.manifest_path == project_root / "nested" / "source" / "eyck.toml"
    assert project.h5ad_path.is_file()
    assert project.h5ad_path.is_relative_to(project.source_root)
    assert project.output_path.is_relative_to(project.source_root)
    assert [item.id for item in project.embeddings] == ["umap"]


def test_output_uri_remains_runtime_confined(tmp_path: Path, monkeypatch) -> None:
    manifest = write_project(tmp_path)
    source_output = manifest.parent / "output/annotation_projects/cells"
    output_root = tmp_path / "managed-output"
    managed_output = output_root / "sources/test-source/annotation_projects/cells"
    managed_output.parent.mkdir(parents=True)
    source_output.rename(managed_output)
    text = manifest.read_text(encoding="utf-8")
    text = text.replace(
        'output = "output/annotation_projects/cells"',
        'output = "output://sources/test-source/annotation_projects/cells"',
    ).replace(
        'initial_memberships = "output/annotation_projects/cells/generated_initial.parquet"',
        'initial_memberships = "output://sources/test-source/annotation_projects/cells/generated_initial.parquet"',
    )
    manifest.write_text(text, encoding="utf-8")
    monkeypatch.setenv("IOMIX_OUTPUT_ROOT", str(output_root))

    spec = discover_projects([tmp_path])["cells"]
    project = EyckProject(spec)
    with project.writer_lock():
        assert project.spec.output_path == managed_output


def test_discovery_rejects_unknown_fields_and_escaping_paths(tmp_path: Path):
    manifest = write_project(tmp_path)
    original = manifest.read_text(encoding="utf-8")
    manifest.write_text(original + 'mystery = "no"\n', encoding="utf-8")
    with pytest.raises(DiscoveryError, match="unknown project fields"):
        discover_projects([tmp_path])

    manifest.write_text(
        original.replace(
            'h5ad = "data/cells.h5ad"', 'h5ad = "../../../../outside.h5ad"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(DiscoveryError, match="escapes source project"):
        discover_projects([tmp_path])


def test_discovery_rejects_symlink_escape(tmp_path: Path):
    manifest = write_project(tmp_path)
    source = manifest.parent
    h5ad = source / "data" / "cells.h5ad"
    outside = tmp_path / "outside.h5ad"
    h5ad.rename(outside)
    h5ad.symlink_to(outside)
    with pytest.raises(DiscoveryError, match="escapes source project"):
        discover_projects([tmp_path])


def test_discovery_rejects_duplicate_project_ids(tmp_path: Path):
    first = write_project(tmp_path / "one")
    second = write_project(tmp_path / "two")
    assert first != second
    with pytest.raises(DiscoveryError, match="duplicate project ID"):
        discover_projects([tmp_path])


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (
            {"schema_version": 1, "labels": [{"id": "../bad", "name": "bad"}]},
            "unsafe label ID",
        ),
        (
            {
                "schema_version": 1,
                "labels": [{"id": "a", "name": "a", "parent_ids": ["missing"]}],
            },
            "missing parents",
        ),
        (
            {
                "schema_version": 1,
                "labels": [
                    {"id": "a", "name": "a", "parent_ids": ["b"]},
                    {"id": "b", "name": "b", "parent_ids": ["a"]},
                ],
            },
            "cycle",
        ),
    ],
)
def test_label_dag_rejects_unsafe_missing_and_cyclic(
    labels: dict, message: str, tmp_path: Path
):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels), encoding="utf-8")
    with pytest.raises(DiscoveryError, match=message):
        load_labels(path)


def test_label_dag_accepts_multiple_parents(tmp_path: Path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(LABELS), encoding="utf-8")
    labels = load_labels(path)
    intersection = next(label for label in labels.labels if label.id == "a-b")
    assert intersection.parent_ids == ["a", "b"]
