import type {
  FeatureDescriptor,
  FeaturesPayload,
  MembershipDraft,
  MembershipPayload,
  PointsPayload,
  ProjectDetail,
  ProjectIndex,
  Scalar
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
  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }
  if (!response.ok) {
    let message = response.statusText || `Request failed with status ${response.status}`;
    if (payload && typeof payload === "object" && "detail" in payload) {
      message = String((payload as { detail: unknown }).detail);
    } else if (payload && typeof payload === "object" && "error" in payload) {
      const error = (payload as { error: unknown }).error;
      message = error && typeof error === "object" && "message" in error
        ? String((error as { message: unknown }).message)
        : String(error);
    }
    throw new ApiError(response.status, message, payload);
  }
  return payload as T;
}

function get<T>(url: string, signal?: AbortSignal): Promise<T> {
  return fetch(url, {
    headers: { Accept: "application/json" },
    signal
  }).then(readResponse<T>);
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

export function getMemberships(url: string, signal?: AbortSignal): Promise<MembershipPayload> {
  return get<MembershipPayload>(url, signal);
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
): Promise<Scalar[]> {
  const base = url.replace(/\/$/, "");
  const payload = await get<{ values: Scalar[] } | Scalar[]>(
    `${base}/${featureIndex}/values`,
    signal
  );
  return Array.isArray(payload) ? payload : payload.values;
}

function responseRevision(response: Response, payload: unknown, fallback: string): string {
  if (payload && typeof payload === "object" && "revision" in payload) {
    return String((payload as { revision: unknown }).revision);
  }
  return response.headers.get("ETag")?.replace(/^W\//, "").replace(/^"|"$/g, "") ?? fallback;
}

export async function putMemberships(
  url: string,
  revision: string,
  csrfToken: string,
  draft: MembershipDraft
): Promise<string> {
  const response = await fetch(url, {
    method: "PUT",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      "If-Match": revision,
      "X-Eyck-CSRF": csrfToken
    },
    body: JSON.stringify(draft)
  });
  const payload = await readResponse<unknown>(response);
  return responseRevision(response, payload, revision);
}

export async function exportAnnotations(
  url: string,
  revision: string,
  csrfToken: string
): Promise<unknown> {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "If-Match": revision,
      "X-Eyck-CSRF": csrfToken
    }
  });
  return readResponse<unknown>(response);
}

export async function restartServer(csrfToken: string): Promise<void> {
  try {
    const response = await fetch(publicUrl("/api/v1/restart"), {
      method: "POST",
      headers: { Accept: "application/json", "X-Eyck-CSRF": csrfToken }
    });
    await readResponse<unknown>(response);
  } catch (error) {
    // A server that exits immediately can terminate the request before a response arrives.
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
      // The expected state while the process is coming back up.
    }
    await new Promise((resolve) => window.setTimeout(resolve, 900));
  }
  throw new Error("The server did not become available within one minute");
}
