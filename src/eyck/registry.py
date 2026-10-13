"""Loaded projects, rescanned so new or changed declarations load without a restart."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from .discovery import ProjectSpec, discover_projects
from .process_drawings import ProcessAnnotations
from .project import EyckProject

# Uvicorn configures this logger, so rescans appear in the server log.
logger = logging.getLogger("uvicorn.error")


class ProjectRegistry(Mapping[str, EyckProject]):
    """Live mapping of project IDs to loaded projects.

    A rescan loads new projects and reloads a project when its declaration or
    h5ad changes. A project whose manifest or data fails to load keeps its last
    loaded state, and the failure is reported instead of hiding other projects.
    """

    def __init__(
        self,
        roots: Sequence[str | Path] | None = None,
        *,
        only: set[str] | None = None,
        specs: dict[str, ProjectSpec] | None = None,
    ):
        self._roots = list(roots or [])
        self._only = only
        self._refresh_lock = threading.Lock()
        self.errors: dict[str, str] = {}
        if specs is None:
            specs = discover_projects(self._roots, only=only)
        self._loaded = {project_id: _load(spec) for project_id, spec in specs.items()}

    def __getitem__(self, project_id: str) -> EyckProject:
        return self._loaded[project_id][1]

    def __iter__(self) -> Iterator[str]:
        return iter(dict(self._loaded))

    def __len__(self) -> int:
        return len(self._loaded)

    def drawings(self, project_id: str) -> ProcessAnnotations:
        return self._loaded[project_id][2]

    def refresh(self) -> None:
        if not self._roots:
            return
        with self._refresh_lock:
            manifest_errors: dict[Path, str] = {}
            specs = discover_projects(self._roots, only=self._only, errors=manifest_errors)
            errors = {str(path): message for path, message in manifest_errors.items()}
            loaded: dict[str, tuple[tuple, EyckProject, ProcessAnnotations]] = {}
            for project_id, spec in specs.items():
                current = self._loaded.get(project_id)
                try:
                    key = _reload_key(spec)
                    loaded[project_id] = (
                        current if current is not None and current[0] == key else _load(spec)
                    )
                except Exception as exc:
                    errors[str(spec.manifest_path)] = f"{project_id}: {exc}"
                    if current is not None:
                        loaded[project_id] = current
            for project_id, current in self._loaded.items():
                if project_id not in loaded and str(current[1].spec.manifest_path) in errors:
                    loaded[project_id] = current
            added = sorted(set(loaded) - set(self._loaded))
            removed = sorted(set(self._loaded) - set(loaded))
            reloaded = sorted(
                project_id
                for project_id in set(loaded) & set(self._loaded)
                if loaded[project_id] is not self._loaded[project_id]
            )
            self._loaded = loaded
            self.errors = errors
        for verb, ids in (("added", added), ("removed", removed), ("reloaded", reloaded)):
            if ids:
                logger.info("Eyck projects %s: %s", verb, ", ".join(ids))

    def watch(self, interval_seconds: float) -> threading.Thread:
        def loop() -> None:
            while True:
                time.sleep(interval_seconds)
                try:
                    self.refresh()
                except Exception:
                    logger.exception("Eyck project rescan failed")

        thread = threading.Thread(target=loop, name="eyck-project-rescan", daemon=True)
        thread.start()
        return thread


def _reload_key(spec: ProjectSpec) -> tuple:
    stat = spec.h5ad_path.stat()
    return (spec, stat.st_mtime_ns, stat.st_size)


def _load(spec: ProjectSpec) -> tuple[tuple, EyckProject, ProcessAnnotations]:
    key = _reload_key(spec)
    project = EyckProject(spec)
    return key, project, ProcessAnnotations(project)
