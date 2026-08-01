import type {
  ClusteringImport,
  ClusteringResult,
  DeletionConfirm,
  DeletionImpact,
  DeletionRequest,
  ExportResponse,
  FeatureDescriptor,
  FeatureValuesPayload,
  FeaturesPayload,
  LabelImpact,
  LabelMutation,
  LabelState,
  LocalAnalysisCreate,
  MarkerProgramCreate,
  MarkerProgramResult,
  MembershipDraft,
  MembershipPayload,
  PointsPayload,
  ProjectDetail,
  ProjectIndex,
  SelectionCreate,
  WorkspaceDocument,
  ZoomCreate
} from "./types";

export const basePath =
  document.querySelector<HTMLMetaElement>('meta[name="eyck-base-path"]')?.content ?? "";

export function publicUrl(path: string): string {
  return `${basePath}${path}`;
}

export class ApiError extends Error {
  status: number;
  payload: unknown;

  constructor(status: number, message: string, payload: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

async function readResponse<T>(response: Response): Promise<T> {
  let payload: unknown = null;
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
  } else {
    const text = await response.text();
    payload = text || null;
  }
  if (!response.ok) {
    let message = response.statusText || `Request failed with status ${response.status}`;
    if (payload && typeof payload === "object" && "detail" in payload) {
      const detail = (payload as { detail: unknown }).detail;
      message = Array.isArray(detail)
        ? detail.map((item) => typeof item === "object" && item && "msg" in item
          ? String((item as { msg: unknown }).msg) : String(item)).join("; ")
        : String(detail);
    }
    throw new ApiError(response.status, message, payload);
  }
  return payload as T;
}

function get<T>(url: string, signal?: AbortSignal): Promise<T> {
  return fetch(url, { headers: { Accept: "application/json" }, signal }).then(readResponse<T>);
}

async function mutate<T>(
  url: string,
  method: "POST" | "PUT",
  csrfToken: string,
  body?: unknown,
  revision?: string
): Promise<T> {
  const headers: Record<string, string> = {
    Accept: "application/json",
    "X-Eyck-CSRF": csrfToken
  };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (revision !== undefined) headers["If-Match"] = revision;
  const response = await fetch(url, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body)
  });
  return readResponse<T>(response);
}

function workspaceEndpoint(workspaceUrl: string, path: string): string {
  return `${workspaceUrl.replace(/\/$/, "")}${path}`;
}

export function getProjects(signal?: AbortSignal): Promise<ProjectIndex> {
  return get<ProjectIndex>(publicUrl("/api/v1/annotations"), signal);
}

export function getProject(projectId: string, signal?: AbortSignal): Promise<ProjectDetail> {
  return get<ProjectDetail>(
    publicUrl(`/api/v1/annotations/${encodeURIComponent(projectId)}`),
    signal
  );
}

export function getPoints(url: string, signal?: AbortSignal): Promise<PointsPayload> {
  return get<PointsPayload>(url, signal);
}

export function getZoomPoints(
  workspaceUrl: string,
  zoomId: string,
  embeddingId: string,
  signal?: AbortSignal
): Promise<PointsPayload> {
  const url = new URL(
    workspaceEndpoint(workspaceUrl, `/zooms/${encodeURIComponent(zoomId)}/points`),
    window.location.href
  );
  url.searchParams.set("embedding", embeddingId);
  return get<PointsPayload>(url.toString(), signal);
}

export function getWorkspace(url: string, signal?: AbortSignal): Promise<WorkspaceDocument> {
  return get<WorkspaceDocument>(url, signal);
}

export function getMemberships(url: string, signal?: AbortSignal): Promise<MembershipPayload> {
  return get<MembershipPayload>(url, signal);
}

export function getLabels(url: string, signal?: AbortSignal): Promise<LabelState> {
  return get<LabelState>(url, signal);
}

export function searchFeatures(
  url: string,
  query: string,
  signal?: AbortSignal
): Promise<FeatureDescriptor[]> {
  const requestUrl = new URL(url, window.location.href);
  requestUrl.searchParams.set("q", query);
  return get<FeaturesPayload>(requestUrl.toString(), signal).then((payload) => payload.features);
}

export async function getFeatureValues(
  url: string,
  featureIndex: number,
  signal?: AbortSignal
): Promise<FeatureValuesPayload> {
  return get<FeatureValuesPayload>(
    `${url.replace(/\/$/, "")}/${featureIndex}/values`,
    signal
  );
}

export function putMemberships(
  url: string,
  revision: string,
  csrfToken: string,
  draft: MembershipDraft
): Promise<MembershipPayload> {
  return mutate<MembershipPayload>(url, "PUT", csrfToken, draft, revision);
}

export function exportAnnotations(
  url: string,
  revision: string,
  csrfToken: string
): Promise<ExportResponse> {
  return mutate<ExportResponse>(url, "POST", csrfToken, undefined, revision);
}

export function createSelection(
  workspaceUrl: string,
  revision: string,
  csrfToken: string,
  request: SelectionCreate
): Promise<WorkspaceDocument> {
  return mutate(workspaceEndpoint(workspaceUrl, "/selections"), "POST", csrfToken, request, revision);
}

export function createZoom(
  workspaceUrl: string,
  revision: string,
  csrfToken: string,
  request: ZoomCreate
): Promise<WorkspaceDocument> {
  return mutate(workspaceEndpoint(workspaceUrl, "/zooms"), "POST", csrfToken, request, revision);
}

export function importClustering(
  workspaceUrl: string,
  revision: string,
  csrfToken: string,
  request: ClusteringImport
): Promise<WorkspaceDocument> {
  return mutate(workspaceEndpoint(workspaceUrl, "/clusterings/import"), "POST", csrfToken, request, revision);
}

export function computeLocalAnalysis(
  workspaceUrl: string,
  revision: string,
  csrfToken: string,
  request: LocalAnalysisCreate
): Promise<WorkspaceDocument> {
  return mutate(workspaceEndpoint(workspaceUrl, "/analyses/local"), "POST", csrfToken, request, revision);
}

export function createMarkerProgram(
  workspaceUrl: string,
  revision: string,
  csrfToken: string,
  request: MarkerProgramCreate
): Promise<MarkerProgramResult> {
  return mutate(workspaceEndpoint(workspaceUrl, "/marker-programs"), "POST", csrfToken, request, revision);
}

export function getMarkerProgramResult(
  workspaceUrl: string,
  programId: string,
  signal?: AbortSignal
): Promise<MarkerProgramResult> {
  return get(workspaceEndpoint(
    workspaceUrl,
    `/marker-programs/${encodeURIComponent(programId)}/result`
  ), signal);
}

export function getClusteringResult(
  workspaceUrl: string,
  clusteringId: string,
  signal?: AbortSignal
): Promise<ClusteringResult> {
  return get(workspaceEndpoint(
    workspaceUrl,
    `/clusterings/${encodeURIComponent(clusteringId)}/result`
  ), signal);
}

export function previewDeletion(
  workspaceUrl: string,
  csrfToken: string,
  request: DeletionRequest
): Promise<DeletionImpact> {
  return mutate(workspaceEndpoint(workspaceUrl, "/deletions/preview"), "POST", csrfToken, request);
}

export function confirmDeletion(
  workspaceUrl: string,
  revision: string,
  csrfToken: string,
  request: DeletionConfirm
): Promise<WorkspaceDocument> {
  return mutate(workspaceEndpoint(workspaceUrl, "/deletions"), "POST", csrfToken, request, revision);
}

export function previewLabelMutation(
  labelsUrl: string,
  csrfToken: string,
  request: LabelMutation
): Promise<LabelImpact> {
  return mutate(`${labelsUrl.replace(/\/$/, "")}/impact`, "POST", csrfToken, request);
}

export function putLabels(
  labelsUrl: string,
  revision: string,
  csrfToken: string,
  request: LabelMutation
): Promise<LabelState> {
  return mutate(labelsUrl, "PUT", csrfToken, request, revision);
}

export async function restartServer(csrfToken: string): Promise<void> {
  try {
    await mutate(publicUrl("/api/v1/restart"), "POST", csrfToken);
  } catch (error) {
    if (error instanceof ApiError) throw error;
  }
}

export async function waitForRestart(timeoutMs = 60_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  await new Promise((resolve) => window.setTimeout(resolve, 700));
  while (Date.now() < deadline) {
    try {
      const response = await fetch(publicUrl(`/api/v1/annotations?restart=${Date.now()}`), {
        cache: "no-store",
        headers: { Accept: "application/json" }
      });
      if (response.ok) return;
    } catch {
      // The expected state while the process is restarting.
    }
    await new Promise((resolve) => window.setTimeout(resolve, 900));
  }
  throw new Error("The server did not become available within one minute");
}
