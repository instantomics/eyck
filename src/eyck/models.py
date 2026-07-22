"""Typed public API and tracked document models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Label(StrictModel):
    id: str
    name: str
    description: str = ""
    ontology_ids: list[str] = Field(default_factory=list)
    parent_ids: list[str] = Field(default_factory=list)


class LabelDocument(StrictModel):
    schema_version: Literal[1]
    labels: list[Label]


class IdentitySet(StrictModel):
    source_sha256: str
    observation_index_sha256: str
    feature_index_sha256: str
    manifest_sha256: str
    labels_revision: str
    labels_semantic_sha256: str


class NamedInput(StrictModel):
    id: str
    obsm: str
    dimensions: int


class ProjectSummary(StrictModel):
    id: str
    title: str
    description: str
    dataset_id: str
    observation_count: int
    feature_count: int
    url: str


class ProjectIndex(StrictModel):
    projects: list[ProjectSummary]


class ProjectUrls(StrictModel):
    points: str
    feature_search: str
    feature_values: str
    memberships: str
    export: str
    restart: str


class ProjectDetail(StrictModel):
    id: str
    title: str
    description: str
    dataset_id: str
    source_owner: str
    observation_count: int
    feature_count: int
    expression_layer: str
    metadata_columns: list[str]
    embeddings: list[NamedInput]
    modalities: list[NamedInput]
    labels: LabelDocument
    identities: IdentitySet
    csrf_token: str
    urls: ProjectUrls


class PointPayload(StrictModel):
    embedding_id: str
    observation_ids: list[str]
    x: list[float]
    y: list[float]
    metadata: dict[str, list[Any]]


class FeatureHit(StrictModel):
    id: str
    index: int


class FeatureSearch(StrictModel):
    query: str
    features: list[FeatureHit]


class FeatureValues(StrictModel):
    feature_id: str
    observation_ids: list[str]
    values: list[float | None]


MembershipState = Literal["unreviewed", "absent", "present"]


class DecisionRow(StrictModel):
    support_id: str = "default"
    observation_id: str
    entity_id: str = "0"
    label_id: str
    state: MembershipState
    decision_view_id: str = "manual"
    provenance: str = "explicit"


class MembershipDocument(StrictModel):
    schema_version: Literal[1] = 1
    project_id: str
    revision: str
    source: Literal["generated_initial", "draft", "reviewed", "empty"]
    rows: list[DecisionRow]


class MembershipPut(StrictModel):
    expected_revision: str
    rows: list[DecisionRow]


class ExportRequest(StrictModel):
    expected_revision: str


class ExportResponse(StrictModel):
    revision: str
    report_url: str
    membership_rows: int
    support_rows: int


class RestartResponse(StrictModel):
    restarting: bool


class HierarchySummary(StrictModel):
    explicit_decisions: int
    materialized_decisions: int
    supported_observations: int
    entities: int
    derived_ancestors: int
    derived_intersections: int


class AnnotationSummary(StrictModel):
    project_id: str
    title: str
    description: str
    n_observations: int
    n_features: int


class AnnotationIndex(StrictModel):
    projects: list[AnnotationSummary]
    csrf_token: str
    restart_available: bool


class LabelDescriptor(StrictModel):
    label_id: str
    display_name: str
    parents: list[str]
    description: str = ""
    ontology_ids: list[str] = Field(default_factory=list)


class AnnotationDetail(StrictModel):
    project_id: str
    title: str
    description: str
    dataset_id: str
    source_owner: str
    n_observations: int
    n_features: int
    expression_layer: str
    labels: list[LabelDescriptor]
    metadata_columns: list[str]
    embeddings: list[NamedInput]
    modalities: list[NamedInput]
    identities: IdentitySet
    hierarchy_summary: HierarchySummary
    points_url: str
    features_url: str
    memberships_url: str
    export_url: str
    csrf_token: str
    restart_available: bool


class AnnotationPoints(StrictModel):
    observation_ids: list[str]
    coordinates: list[tuple[float, float]]
    metadata: dict[str, list[Any]]
    modalities: dict[str, list[Any]]


class FeatureDescriptor(StrictModel):
    feature_index: int
    feature_id: str
    feature_symbol: str | None = None


class FeaturesPayload(StrictModel):
    features: list[FeatureDescriptor]


class MembershipPayload(StrictModel):
    revision: str
    origin: Literal["generated_initial", "draft", "reviewed", "empty"]
    rows: list[DecisionRow]
    support_observation_ids: list[str]


class MembershipDraftPut(StrictModel):
    rows: list[DecisionRow]
    support_observation_ids: list[str] = Field(default_factory=list)
