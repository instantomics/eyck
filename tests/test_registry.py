from __future__ import annotations

from pathlib import Path

from conftest import write_project
from eyck.registry import ProjectRegistry


def _rename(manifest: Path, project_id: str) -> None:
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text.replace('id = "cells"', f'id = "{project_id}"', 1), encoding="utf-8"
    )


def test_rescan_loads_new_projects_and_isolates_broken_manifests(tmp_path: Path):
    write_project(tmp_path / "one")
    registry = ProjectRegistry([tmp_path])
    original = registry["cells"]

    _rename(write_project(tmp_path / "two"), "later")
    broken = write_project(tmp_path / "three")
    _rename(broken, "broken")
    broken.write_text(broken.read_text(encoding="utf-8") + "unknown = 1\n", encoding="utf-8")
    registry.refresh()

    assert sorted(registry) == ["cells", "later"]
    assert registry["cells"] is original
    assert list(registry.errors) == [str(broken.resolve())]

    # A broken edit of a loaded project keeps serving its last good state.
    first = tmp_path / "one" / "nested" / "source" / "eyck.toml"
    first.write_text(first.read_text(encoding="utf-8") + "unknown = 1\n", encoding="utf-8")
    registry.refresh()
    assert registry["cells"] is original
