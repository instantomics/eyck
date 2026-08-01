export type Scalar = string | number | boolean | null;
export type MembershipState = "present" | "absent" | "unreviewed";
export type BooleanOperator = "union" | "intersection" | "exclusion";
export type Comparator = ">" | ">=" | "<" | "<=";
export type WorkspaceObjectKind = "zoom" | "selection" | "clustering" | "embedding" | "marker_program";

export interface NamedInput {
  id: string;
  obsm: string;
  dimensions: number;
}

export interface ProjectSummary {
  project_id: string;
  title: string;
  description: string;
  n_observations: number;
  n_features: number;
}

export interface ProjectIndex {
  projects: ProjectSummary[];
  csrf_token: string;
  restart_available: boolean;
}

export interface LabelDescriptor {
  label_id: string;
  display_name: string;
  parents: string[];
  description: string;
  ontology_ids: string[];
}

export interface Label {
  id: string;
  name: string;
  description: string;
  ontology_ids: string[];
  parent_ids: string[];
}

export interface LabelState {
  schema_version: 1;
  revision: string;
  labels: Label[];
}

export interface LabelMutation {
  labels: Label[];
  delete_label_ids?: string[];
  confirm_cascade?: boolean;
  expected_membership_revision?: string | null;
  expected_impact_sha256?: string | null;
}

export interface LabelImpact {
  labels_revision: string;
  proposed_revision: string;
  membership_revision: string;
  removed_label_ids: string[];
  changed_label_ids: string[];
  added_label_ids: string[];
  child_edge_ids: string[];
  membership_decision_count: number;
  membership_observation_ids: string[];
  selection_ids: string[];
  impact_sha256: string;
}

export interface HierarchySummary {
  explicit_decisions: number;
  materialized_decisions: number;
  supported_observations: number;
  entities: number;
  derived_ancestors: number;
  derived_intersections: number;
}

export interface IdentitySet {
  source_sha256: string;
  observation_index_sha256: string;
  feature_index_sha256: string;
  manifest_sha256: string;
  labels_revision: string;
  labels_semantic_sha256: string;
}

export interface ProjectDetail {
  project_id: string;
  title: string;
  description: string;
  dataset_id: string;
  source_owner: string;
  n_observations: number;
  n_features: number;
  expression_layer: string;
  labels: LabelDescriptor[];
  metadata_columns: string[];
  embeddings: NamedInput[];
  modalities: NamedInput[];
  identities: IdentitySet;
  hierarchy_summary: HierarchySummary;
  points_url: string;
  features_url: string;
  memberships_url: string;
  export_url: string;
  workspace_url: string;
  labels_url: string;
  csrf_token: string;
  restart_available: boolean;
}

export interface PointsPayload {
  observation_ids: string[];
  coordinates: [number, number][];
  metadata: Record<string, Scalar[]>;
  modalities: Record<string, Scalar[]>;
}

export interface FeatureDescriptor {
  feature_index: number;
  feature_id: string;
  feature_symbol?: string | null;
}

export interface FeaturesPayload {
  features: FeatureDescriptor[];
}

export interface FeatureValuesPayload {
  feature_id: string;
  observation_ids: string[];
  values: Array<number | null>;
}

export interface SelectionProvenance {
  method: string;
  parameters: Record<string, unknown>;
  input_ids: string[];
}

export interface ObservationSelectionDefinition {
  kind: "manual" | "lasso";
  observation_ids: string[];
  provenance: SelectionProvenance;
  embedding_id?: string | null;
  polygon?: [number, number][];
}

export interface ClusterSelectionDefinition {
  kind: "clusters";
  clustering_id: string;
  cluster_ids: string[];
}

export interface MarkerCutoffDefinition {
  kind: "marker_cutoff";
  marker_program_id: string;
  comparator: Comparator;
  cutoff: number;
}

export interface BooleanSelectionDefinition {
  kind: "boolean";
  operator: BooleanOperator;
  selection_ids: string[];
}

export type SelectionDefinition = ObservationSelectionDefinition
  | ClusterSelectionDefinition
  | MarkerCutoffDefinition
  | BooleanSelectionDefinition;

export interface SavedSelection {
  id: string;
  name: string;
  zoom_id: string;
  definition: SelectionDefinition;
  observation_ids: string[];
  observation_sha256: string;
  observation_count: number;
}

export interface ZoomRecipe {
  operator: BooleanOperator;
  selection_ids: string[];
}

export interface Zoom {
  id: string;
  name: string;
  parent_id: string | null;
  recipe: ZoomRecipe | null;
  observation_ids: string[];
  observation_sha256: string;
  observation_count: number;
  fraction_of_root: number;
  fraction_of_parent: number;
  initial_embedding_id: string;
}

export interface ClusteringDescriptor {
  id: string;
  name: string;
  zoom_id: string;
  identity: string;
  implementation: string;
  parameters: Record<string, unknown>;
  provenance: Record<string, unknown>;
  cache_path: string;
  observation_sha256: string;
  cluster_count: number;
}

export interface EmbeddingDescriptor {
  id: string;
  name: string;
  zoom_id: string;
  identity: string;
  implementation: string;
  parameters: Record<string, unknown>;
  provenance: Record<string, unknown>;
  cache_path: string | null;
  inherited_from_embedding_id: string | null;
}

export interface MarkerClusterSummary {
  cluster_id: string;
  cell_count: number;
  mean_score: number;
}

export interface MarkerProgramDescriptor {
  id: string;
  name: string;
  zoom_id: string;
  identity: string;
  marker_ids: string[];
  requested_markers: string[];
  clustering_id: string;
  clustering_identity: string;
  implementation: string;
  parameters: Record<string, unknown>;
  provenance: Record<string, unknown>;
  cache_path: string;
  cluster_table: MarkerClusterSummary[];
}

export interface WorkspaceDocument {
  schema_version: 1;
  project_id: string;
  revision: string;
  source_identity: string;
  root_zoom_id: string;
  zooms: Zoom[];
  selections: SavedSelection[];
  clusterings: ClusteringDescriptor[];
  embeddings: EmbeddingDescriptor[];
  marker_programs: MarkerProgramDescriptor[];
}

export interface SelectionCreate {
  id: string;
  name: string;
  zoom_id: string;
  definition: SelectionDefinition;
}

export interface ZoomCreate {
  id: string;
  name: string;
  parent_id: string;
  parent_embedding_id: string;
  recipe: ZoomRecipe;
}

export interface ClusteringImport {
  id: string;
  name: string;
  zoom_id: string;
  metadata_column: string;
}

export interface LocalAnalysisCreate {
  embedding_id: string;
  embedding_name: string;
  zoom_id: string;
  clustering_outputs: Array<{
    id: string;
    name: string;
    resolution: number;
  }>;
  profile: "scanpy_default" | "scanpy_standard" | "scanpy_fast";
  use_counts?: boolean;
  n_neighbors: number;
  n_pcs: number | null;
  n_top_genes?: number | null;
  n_comps?: number | null;
  random_state: number;
}

export interface MarkerProgramCreate {
  id: string;
  name: string;
  zoom_id: string;
  clustering_id: string;
  markers: string[];
  ctrl_size: number;
  n_bins: number;
  random_state: number;
}

export interface MarkerProgramResult {
  program: MarkerProgramDescriptor;
  observation_ids: string[];
  per_cell_scores: number[];
  cluster_mean_by_cell: number[];
  cluster_ids: string[];
  cluster_table: MarkerClusterSummary[];
}

export interface ClusteringResult {
  clustering: ClusteringDescriptor;
  observation_ids: string[];
  cluster_ids: string[];
}

export interface WorkspaceObjectRef {
  kind: WorkspaceObjectKind;
  id: string;
}

export interface DeletionRequest extends WorkspaceObjectRef {}

export interface DeletionImpact {
  workspace_revision: string;
  membership_revision: string;
  requested: WorkspaceObjectRef;
  removed: WorkspaceObjectRef[];
  counts: Record<WorkspaceObjectKind, number>;
  membership_decision_count: number;
  membership_observation_ids: string[];
  impact_sha256: string;
}

export interface DeletionConfirm extends DeletionRequest {
  confirmed_removed: WorkspaceObjectRef[];
  expected_membership_revision: string;
  impact_sha256: string;
}

export interface MembershipRow {
  support_id: string;
  observation_id: string;
  entity_id: string;
  label_id: string;
  state: MembershipState;
  decision_view_id: string;
  provenance: string;
  selection_id: string | null;
  zoom_id: string | null;
}

export interface MembershipPayload {
  revision: string;
  origin: "generated_initial" | "draft" | "reviewed" | "empty";
  rows: MembershipRow[];
  support_observation_ids: string[];
}

export interface MembershipDraft {
  rows: MembershipRow[];
  support_observation_ids: string[];
}

export interface ExportResponse {
  revision: string;
  report_url: string;
  membership_rows: number;
  support_rows: number;
}

export function labelDescriptors(state: LabelState): LabelDescriptor[] {
  return state.labels.map((label) => ({
    label_id: label.id,
    display_name: label.name,
    parents: label.parent_ids,
    description: label.description,
    ontology_ids: label.ontology_ids
  }));
}
