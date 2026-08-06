"""Strict recursive project discovery and declarative contract validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import LabelDocument

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ROOT_KEYS = {"schema_version", "projects"}
PROJECT_KEYS = {
    "id",
    "title",
    "description",
    "dataset_id",
    "source_owner",
    "h5ad",
    "source_digest",
    "labels",
    "output",
    "expression_layer",
    "embedding",
    "embeddings",
    "metadata_columns",
    "modalities",
    "initial_memberships",
    "reviewed_memberships",
}
NAMED_INPUT_KEYS = {"id", "obsm"}


class DiscoveryError(ValueError):
    """A project declaration is unsafe, ambiguous, or invalid."""


@dataclass(frozen=True)
class NamedInputSpec:
    id: str
    obsm: str


@dataclass(frozen=True)
class ProjectSpec:
    id: str
    title: str
    description: str
    dataset_id: str
    source_owner: str
    manifest_path: Path
    source_root: Path
    labels_root: Path
    output_root: Path
    h5ad_path: Path
    source_digest: str | None
    labels_path: Path
    output_path: Path
    expression_layer: str
    embeddings: tuple[NamedInputSpec, ...]
    metadata_columns: tuple[str, ...]
    modalities: tuple[NamedInputSpec, ...]
    initial_memberships_path: Path | None
    reviewed_memberships_path: Path | None
    manifest_sha256: str
    labels: LabelDocument


def canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def validate_safe_id(value: str, kind: str) -> None:
    if not SAFE_ID.fullmatch(value):
        raise DiscoveryError(f"unsafe {kind} ID: {value!r}")


def _confined_path(root: Path, raw: Any, field: str, *, required: bool) -> Path:
    if not isinstance(raw, str) or not raw:
        raise DiscoveryError(f"{field} must be a non-empty path")
    if raw.startswith("output://"):
        resolved = _output_uri_path(raw, field)
        if required and (not resolved.exists() or not resolved.is_file()):
            raise DiscoveryError(f"{field} does not name a real file: {raw}")
        return resolved
    relative = Path(raw)
    if relative.is_absolute():
        raise DiscoveryError(f"{field} must be relative to {root}")
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DiscoveryError(f"{field} escapes source project: {raw}") from exc
    if required and (not resolved.exists() or not resolved.is_file()):
        raise DiscoveryError(f"{field} does not name a real file: {raw}")
    return resolved


def _output_uri_path(raw: str, field: str) -> Path:
    relative = Path(raw.removeprefix("output://"))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise DiscoveryError(f"{field} contains an invalid output URI: {raw}")
    root = _configured_output_root(field)
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DiscoveryError(f"{field} escapes the output root: {raw}") from exc
    return resolved


def _configured_output_root(field: str) -> Path:
    output_root = os.environ.get("IOMIX_OUTPUT_ROOT")
    if not output_root:
        raise DiscoveryError(
            f"{field} uses an output URI but IOMIX_OUTPUT_ROOT is not configured"
        )
    return Path(output_root).expanduser().resolve()


def _confinement_root(source_root: Path, raw: Any, field: str) -> Path:
    return (
        _configured_output_root(field)
        if isinstance(raw, str) and raw.startswith("output://")
        else source_root
    )


def _named_inputs(raw: Any, field: str) -> tuple[NamedInputSpec, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise DiscoveryError(f"{field} must be an array")
    result: list[NamedInputSpec] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            item = {"id": item, "obsm": item}
        if not isinstance(item, dict) or set(item) - NAMED_INPUT_KEYS:
            raise DiscoveryError(f"invalid {field} entry: {item!r}")
        if set(item) != NAMED_INPUT_KEYS:
            raise DiscoveryError(f"{field} entries require id and obsm")
        identifier, obsm = item["id"], item["obsm"]
        if not isinstance(identifier, str) or not isinstance(obsm, str) or not obsm:
            raise DiscoveryError(f"invalid {field} entry: {item!r}")
        validate_safe_id(identifier, field)
        if identifier in seen:
            raise DiscoveryError(f"duplicate {field} ID: {identifier}")
        seen.add(identifier)
        result.append(NamedInputSpec(identifier, obsm))
    return tuple(result)


def load_labels(path: Path) -> LabelDocument:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        document = LabelDocument.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise DiscoveryError(f"invalid labels document {path}: {exc}") from exc

    return validate_label_document(document)


def validate_label_document(document: LabelDocument) -> LabelDocument:
    """Validate label identities and DAG structure independent of storage."""
    by_id = {}
    for label in document.labels:
        validate_safe_id(label.id, "label")
        if label.id in by_id:
            raise DiscoveryError(f"duplicate label ID: {label.id}")
        if len(label.parent_ids) != len(set(label.parent_ids)):
            raise DiscoveryError(f"duplicate parent on label {label.id}")
        by_id[label.id] = label
    for label in document.labels:
        missing = set(label.parent_ids) - set(by_id)
        if missing:
            raise DiscoveryError(
                f"label {label.id} has missing parents: {sorted(missing)}"
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(label_id: str) -> None:
        if label_id in visiting:
            raise DiscoveryError(f"label DAG contains a cycle at {label_id}")
        if label_id in visited:
            return
        visiting.add(label_id)
        for parent in by_id[label_id].parent_ids:
            visit(parent)
        visiting.remove(label_id)
        visited.add(label_id)

    for label_id in by_id:
        visit(label_id)
    return document


def _parse_project(manifest: Path, raw: Any) -> ProjectSpec:
    if not isinstance(raw, dict):
        raise DiscoveryError(f"project in {manifest} must be a table")
    unknown = set(raw) - PROJECT_KEYS
    if unknown:
        raise DiscoveryError(f"unknown project fields in {manifest}: {sorted(unknown)}")
    required = {"id", "h5ad", "labels", "output"}
    missing = required - set(raw)
    if missing:
        raise DiscoveryError(f"missing project fields in {manifest}: {sorted(missing)}")

    project_id = raw["id"]
    if not isinstance(project_id, str):
        raise DiscoveryError("project id must be a string")
    validate_safe_id(project_id, "project")
    for field in (
        "title",
        "description",
        "dataset_id",
        "source_owner",
        "source_digest",
        "expression_layer",
    ):
        if field in raw and not isinstance(raw[field], str):
            raise DiscoveryError(f"project field {field} must be a string")
    source_root = manifest.parent.resolve(strict=True)
    h5ad = _confined_path(source_root, raw["h5ad"], "h5ad", required=True)
    labels_path = _confined_path(source_root, raw["labels"], "labels", required=True)
    output = _confined_path(source_root, raw["output"], "output", required=False)
    labels_root = _confinement_root(source_root, raw["labels"], "labels")
    output_root = _confinement_root(source_root, raw["output"], "output")
    if output == source_root:
        raise DiscoveryError("output must not be the source project root")

    embeddings_raw = raw.get("embeddings")
    if "embedding" in raw:
        if embeddings_raw is not None:
            raise DiscoveryError("use either embedding or embeddings, not both")
        embeddings_raw = [raw["embedding"]]
    embeddings = _named_inputs(embeddings_raw, "embeddings")
    if not embeddings:
        raise DiscoveryError(f"project {project_id} declares no inherited embedding")
    modalities = _named_inputs(raw.get("modalities", []), "modalities")

    metadata = raw.get("metadata_columns", [])
    if not isinstance(metadata, list) or not all(
        isinstance(x, str) and x for x in metadata
    ):
        raise DiscoveryError("metadata_columns must be an array of non-empty strings")
    if len(metadata) != len(set(metadata)):
        raise DiscoveryError("metadata_columns contains duplicates")

    initial = raw.get("initial_memberships")
    reviewed = raw.get("reviewed_memberships")
    initial_path = (
        _confined_path(source_root, initial, "initial_memberships", required=True)
        if initial
        else None
    )
    reviewed_path = (
        _confined_path(source_root, reviewed, "reviewed_memberships", required=True)
        if reviewed
        else None
    )
    if (
        initial_path is not None
        and initial_path != output
        and output not in initial_path.parents
    ):
        raise DiscoveryError(
            "initial_memberships must be beneath the declared output directory"
        )
    labels = load_labels(labels_path)
    manifest_identity = canonical_sha256(raw)
    return ProjectSpec(
        id=project_id,
        title=str(raw.get("title", project_id)),
        description=str(raw.get("description", "")),
        dataset_id=str(raw.get("dataset_id", project_id)),
        source_owner=str(raw.get("source_owner", "source")),
        manifest_path=manifest,
        source_root=source_root,
        labels_root=labels_root,
        output_root=output_root,
        h5ad_path=h5ad,
        source_digest=raw.get("source_digest"),
        labels_path=labels_path,
        output_path=output,
        expression_layer=str(raw.get("expression_layer", "X")),
        embeddings=embeddings,
        metadata_columns=tuple(metadata),
        modalities=modalities,
        initial_memberships_path=initial_path,
        reviewed_memberships_path=reviewed_path,
        manifest_sha256=manifest_identity,
        labels=labels,
    )


def discover_projects(roots: list[str | Path]) -> dict[str, ProjectSpec]:
    """Recursively discover every eyck.toml or fail the complete discovery."""
    manifests: set[Path] = set()
    if not roots:
        raise DiscoveryError("at least one discovery root is required")
    for supplied in roots:
        root = Path(supplied).expanduser().resolve(strict=True)
        if root.is_file():
            if root.name != "eyck.toml":
                raise DiscoveryError(f"discovery file is not eyck.toml: {root}")
            manifests.add(root)
        elif root.is_dir():
            manifests.update(
                path.resolve(strict=True) for path in root.rglob("eyck.toml")
            )
        else:
            raise DiscoveryError(f"discovery root is not a file or directory: {root}")
    if not manifests:
        raise DiscoveryError("no eyck.toml manifests discovered")

    projects: dict[str, ProjectSpec] = {}
    outputs: list[tuple[str, Path]] = []
    for manifest in sorted(manifests):
        try:
            raw = tomllib.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise DiscoveryError(f"invalid manifest {manifest}: {exc}") from exc
        unknown = set(raw) - ROOT_KEYS
        if unknown:
            raise DiscoveryError(
                f"unknown manifest fields in {manifest}: {sorted(unknown)}"
            )
        if raw.get("schema_version") != 1:
            raise DiscoveryError(f"unsupported schema_version in {manifest}")
        declarations = raw.get("projects")
        if not isinstance(declarations, list) or not declarations:
            raise DiscoveryError(f"{manifest} must contain one or more [[projects]]")
        for declaration in declarations:
            project = _parse_project(manifest, declaration)
            if project.id in projects:
                raise DiscoveryError(
                    f"duplicate project ID across manifests: {project.id}"
                )
            for other_id, other_output in outputs:
                if (
                    project.output_path == other_output
                    or project.output_path in other_output.parents
                    or other_output in project.output_path.parents
                ):
                    raise DiscoveryError(
                        f"conflicting output roots for {other_id} and {project.id}"
                    )
            projects[project.id] = project
            outputs.append((project.id, project.output_path))
    return projects
