"""Typed public API and tracked document models."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


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


class SelectionProvenance(StrictModel):
    method: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    input_ids: list[str] = Field(default_factory=list)


class ObservationSetDefinition(StrictModel):
    kind: Literal["manual", "lasso"]
    observation_ids: list[str]
    provenance: SelectionProvenance
    embedding_id: str | None = None
    polygon: list[tuple[float, float]] = Field(default_factory=list)


class ClusterSelectionDefinition(StrictModel):
    kind: Literal["clusters"]
    clustering_id: str
    cluster_ids: list[str]


class MarkerCutoffDefinition(StrictModel):
    kind: Literal["marker_cutoff"]
    marker_program_id: str
    comparator: Literal[">", ">=", "<", "<="]
    cutoff: float


class BooleanSelectionDefinition(StrictModel):
    kind: Literal["boolean"]
    operator: Literal["union", "intersection", "exclusion"]
    selection_ids: list[str]


SelectionDefinition = Annotated[
    ObservationSetDefinition
    | ClusterSelectionDefinition
    | MarkerCutoffDefinition
    | BooleanSelectionDefinition,
    Field(discriminator="kind"),
]


class SavedSelection(StrictModel):
    id: str
    name: str
    zoom_id: str
    definition: SelectionDefinition
    observation_ids: list[str]
    observation_sha256: str
    observation_count: int


class ZoomRecipe(StrictModel):
    operator: Literal["union", "intersection", "exclusion"] = "union"
    selection_ids: list[str]


class Zoom(StrictModel):
    id: str
    name: str
    parent_id: str | None
    recipe: ZoomRecipe | None
    observation_ids: list[str]
    observation_sha256: str
    observation_count: int
    fraction_of_root: float
    fraction_of_parent: float
    initial_embedding_id: str


class ClusteringDescriptor(StrictModel):
    id: str
    name: str
    zoom_id: str
    identity: str
    implementation: str
    parameters: dict[str, Any]
    provenance: dict[str, Any]
    cache_path: str
    observation_sha256: str
    cluster_count: int


class EmbeddingDescriptor(StrictModel):
    id: str
    name: str
    zoom_id: str
    identity: str
    implementation: str
    parameters: dict[str, Any]
    provenance: dict[str, Any]
    cache_path: str | None = None
    inherited_from_embedding_id: str | None = None


class MarkerClusterSummary(StrictModel):
    cluster_id: str
    cell_count: int
    mean_score: float


class MarkerProgramDescriptor(StrictModel):
    id: str
    name: str
    zoom_id: str
    identity: str
    marker_ids: list[str]
    requested_markers: list[str]
    clustering_id: str
    clustering_identity: str
    implementation: str
    parameters: dict[str, Any]
    provenance: dict[str, Any]
    cache_path: str
    cluster_table: list[MarkerClusterSummary]


class WorkspaceDocument(StrictModel):
    schema_version: Literal[1] = 1
    project_id: str
    revision: str
    source_identity: str
    root_zoom_id: str = "root"
    zooms: list[Zoom]
    selections: list[SavedSelection] = Field(default_factory=list)
    clusterings: list[ClusteringDescriptor] = Field(default_factory=list)
    embeddings: list[EmbeddingDescriptor] = Field(default_factory=list)
    marker_programs: list[MarkerProgramDescriptor] = Field(default_factory=list)


class SelectionCreate(StrictModel):
    id: str
    name: str
    zoom_id: str
    definition: SelectionDefinition


class ZoomCreate(StrictModel):
    id: str
    name: str
    parent_id: str
    parent_embedding_id: str
    recipe: ZoomRecipe


class ClusteringImport(StrictModel):
    id: str
    name: str
    zoom_id: str
    metadata_column: str


class ExternalClusteringImport(StrictModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str
    name: str
    zoom_id: str
    observation_ids: list[str]
    cluster_ids: list[str]
    implementation: str = Field(min_length=1)
    parameters: dict[str, JsonValue]
    provenance: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_assignments(self) -> "ExternalClusteringImport":
        if not self.implementation.strip():
            raise ValueError("implementation must not be blank")
        if len(self.observation_ids) != len(self.cluster_ids):
            raise ValueError("observation_ids and cluster_ids must have equal lengths")
        return self


class LocalClusteringOutput(StrictModel):
    id: str
    name: str
    resolution: float = Field(gt=0)


class LocalAnalysisCreate(StrictModel):
    embedding_id: str
    embedding_name: str
    zoom_id: str
    clustering_outputs: list[LocalClusteringOutput] = Field(min_length=1)
    profile: Literal["scanpy_default", "scanpy_standard", "scanpy_fast"]
    use_counts: bool = False
    n_neighbors: int = Field(default=15, ge=1)
    n_pcs: int | None = Field(default=None, ge=1)
    n_top_genes: int | None = Field(default=None, ge=1)
    n_comps: int | None = Field(default=None, ge=1)
    random_state: int = 0

    @model_validator(mode="after")
    def validate_profile(self) -> "LocalAnalysisCreate":
        clustering_ids = [item.id for item in self.clustering_outputs]
        if len(clustering_ids) != len(set(clustering_ids)):
            raise ValueError("clustering output IDs must be unique")
        if self.embedding_id in clustering_ids:
            raise ValueError("embedding and clustering output IDs must be unique")
        if self.profile in {"scanpy_default", "scanpy_standard"} and (
            self.n_top_genes is not None or self.n_comps is not None
        ):
            raise ValueError(
                f"{self.profile} does not accept n_top_genes or n_comps overrides"
            )
        if self.profile == "scanpy_fast":
            resolved_n_pcs = self.n_pcs or 50
            resolved_n_comps = self.n_comps or 50
            if resolved_n_pcs > resolved_n_comps:
                raise ValueError("scanpy_fast n_pcs must not exceed n_comps")
        return self


class MarkerProgramCreate(StrictModel):
    id: str
    name: str
    zoom_id: str
    clustering_id: str
    markers: list[str] = Field(min_length=1)
    ctrl_size: int = Field(default=50, ge=1)
    n_bins: int = Field(default=25, ge=2)
    random_state: int = 0


class MarkerProgramResult(StrictModel):
    program: MarkerProgramDescriptor
    observation_ids: list[str]
    per_cell_scores: list[float]
    cluster_mean_by_cell: list[float]
    cluster_ids: list[str]
    cluster_table: list[MarkerClusterSummary]


class ClusteringResult(StrictModel):
    clustering: ClusteringDescriptor
    observation_ids: list[str]
    cluster_ids: list[str]


WorkspaceObjectKind = Literal["zoom", "selection", "clustering", "embedding", "marker_program"]


class WorkspaceObjectRef(StrictModel):
    kind: WorkspaceObjectKind
    id: str


class DeletionRequest(StrictModel):
    kind: WorkspaceObjectKind
    id: str


class DeletionImpact(StrictModel):
    workspace_revision: str
    membership_revision: str
    requested: WorkspaceObjectRef
    removed: list[WorkspaceObjectRef]
    counts: dict[WorkspaceObjectKind, int]
    membership_decision_count: int = 0
    membership_observation_ids: list[str] = Field(default_factory=list)
    impact_sha256: str


class DeletionConfirm(DeletionRequest):
    confirmed_removed: list[WorkspaceObjectRef]
    expected_membership_revision: str
    impact_sha256: str


class LabelState(StrictModel):
    schema_version: Literal[1] = 1
    revision: str
    labels: list[Label]


class LabelMutation(StrictModel):
    labels: list[Label]
    delete_label_ids: list[str] = Field(default_factory=list)
    confirm_cascade: bool = False
    expected_membership_revision: str | None = None
    expected_impact_sha256: str | None = None


class LabelImpact(StrictModel):
    labels_revision: str
    proposed_revision: str
    membership_revision: str
    removed_label_ids: list[str]
    changed_label_ids: list[str]
    added_label_ids: list[str]
    child_edge_ids: list[str]
    membership_decision_count: int
    membership_observation_ids: list[str]
    selection_ids: list[str]
    impact_sha256: str


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
    selection_id: str | None = None
    zoom_id: str | None = None


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
    workspace_url: str
    labels_url: str
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
