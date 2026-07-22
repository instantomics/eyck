export type Scalar = string | number | boolean | null;
export type MembershipState = "present" | "absent" | "unreviewed";

export interface NamedDescriptor {
  id?: string;
  name?: string;
  key?: string;
  column_id?: string;
  modality_id?: string;
  display_name?: string;
}

export type Descriptor = string | NamedDescriptor;

export interface ProjectSummary {
  project_id: string;
  title: string;
  description?: string;
  n_observations?: number;
  n_features?: number;
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
}

export interface ProjectDetail {
  project_id: string;
  title: string;
  description?: string;
  n_observations: number;
  n_features: number;
  labels: LabelDescriptor[];
  metadata_columns: Descriptor[];
  modalities: Descriptor[];
  points_url: string;
  features_url: string;
  memberships_url: string;
  export_url: string;
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
  feature_symbol?: string;
}

export interface FeaturesPayload {
  features: FeatureDescriptor[];
}

export interface MembershipRow {
  observation_id: string;
  entity_id: string;
  label_id: string;
  state: MembershipState;
}

export interface MembershipPayload {
  revision: string;
  origin: string;
  rows: MembershipRow[];
  support_observation_ids: string[];
}

export interface MembershipDraft {
  rows: MembershipRow[];
  support_observation_ids: string[];
}

export function descriptorKey(descriptor: Descriptor): string {
  if (typeof descriptor === "string") return descriptor;
  return descriptor.id
    ?? descriptor.key
    ?? descriptor.column_id
    ?? descriptor.modality_id
    ?? descriptor.name
    ?? descriptor.display_name
    ?? "";
}

export function descriptorName(descriptor: Descriptor): string {
  return typeof descriptor === "string"
    ? descriptor
    : descriptor.display_name ?? descriptor.name ?? descriptorKey(descriptor);
}
