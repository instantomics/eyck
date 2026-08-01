import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  basePath,
  computeLocalAnalysis,
  confirmDeletion,
  createMarkerProgram,
  createSelection,
  createZoom,
  exportAnnotations,
  getMarkerProgramResult,
  getProject,
  getProjects,
  getWorkspace,
  getZoomPoints,
  importClustering,
  previewDeletion,
  restartServer,
  waitForRestart
} from "./api";
import { AnalysisPanel } from "./AnalysisPanel";
import { AnnotationPanel } from "./AnnotationPanel";
import { colorValues, uniformColors, type ColorResult } from "./color";
import { ColoringControls } from "./ColoringControls";
import { DeleteDialog } from "./DeleteDialog";
import { HierarchyNavigator } from "./HierarchyNavigator";
import { LabelDagEditor } from "./LabelDagEditor";
import { MarkerPanel } from "./MarkerPanel";
import { RecipePanel } from "./RecipePanel";
import { Scatterplot, type SelectionMode } from "./Scatterplot";
import { SelectionPanel } from "./SelectionPanel";
import type {
  ClusteringImport,
  DeletionImpact,
  LabelState,
  LocalAnalysisCreate,
  MarkerProgramCreate,
  MarkerProgramResult,
  PointsPayload,
  ProjectDetail,
  ProjectIndex,
  SavedSelection,
  Scalar,
  SelectionCreate,
  WorkspaceDocument,
  WorkspaceObjectRef,
  ZoomCreate
} from "./types";
import { labelDescriptors } from "./types";
import { Badge, Button, Loading, Notice, Select } from "./ui";
import { useMembershipDraft } from "./useMembershipDraft";
import { formatCount, formatFraction, zoomPath } from "./workspace";

const SELECTION_COLORS = [
  "#d3543c", "#2a7f62", "#7650a8", "#d0922d", "#1f75a8", "#b34472",
  "#568f36", "#a05b32", "#397d86", "#835c44", "#5267b2", "#a56b9e"
];

function projectIdFromPath(): string | null {
  const route = window.location.pathname.slice(basePath.length);
  const match = route.match(/\/annotations\/([^/]+)\/?$/);
  return match ? decodeURIComponent(match[1]) : null;
}

function RestartButton({ csrfToken, available }: { csrfToken: string; available: boolean }) {
  const [restarting, setRestarting] = useState(false);
  const [error, setError] = useState("");
  if (!available) return null;
  return <div className="restart-control"><Button disabled={restarting} onClick={() => {
    setRestarting(true); setError("");
    void restartServer(csrfToken).then(() => waitForRestart()).then(() => window.location.reload()).catch((caught: unknown) => {
      setError(caught instanceof Error ? caught.message : "Restart failed"); setRestarting(false);
    });
  }}>{restarting ? "Restarting..." : "Restart server"}</Button>{error && <span>{error}</span>}</div>;
}

function Dashboard() {
  const [index, setIndex] = useState<ProjectIndex | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    getProjects(controller.signal).then(setIndex).catch((caught: unknown) => {
      if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : "Could not load projects");
    });
    return () => controller.abort();
  }, []);
  return <main className="dashboard"><header className="dashboard-header"><strong>Eyck</strong><span>Single-cell annotation</span></header><section className="project-section"><div className="project-list-header"><div><h1>Annotation projects</h1><p>Select a dataset to explore from all loaded observations.</p></div>{index && <RestartButton csrfToken={index.csrf_token} available={index.restart_available} />}</div>{error && <Notice tone="danger">{error}</Notice>}{!index && !error && <Loading label="Loading projects" />}{index?.projects.length === 0 && <div className="empty-state">No Eyck annotation projects were discovered.</div>}{index && index.projects.length > 0 && <div className="project-list">{index.projects.map((project) => <a className="project-row" href={`${basePath}/annotations/${encodeURIComponent(project.project_id)}`} key={project.project_id}><div className="project-row-main"><strong>{project.title}</strong><span>{project.description || project.project_id}</span></div><div className="project-row-meta"><span>{formatCount(project.n_observations)} observations</span><span>{formatCount(project.n_features)} features</span><code>{project.project_id}</code></div><span className="project-open">Open root</span></a>)}</div>}</section></main>;
}

function SaveIndicator({ status }: { status: ReturnType<typeof useMembershipDraft>["status"] }) {
  const labels = { loading: "Loading draft", saved: "Saved", dirty: "Unsaved", saving: "Saving", conflict: "Conflict", error: "Save failed" };
  const className = status === "saved" ? "saved" : status === "dirty" || status === "saving" || status === "loading" ? "working" : "problem";
  return <span className={`save-indicator ${className}`}><i />{labels[status]}</span>;
}

function Workbench({ projectId }: { projectId: string }) {
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [workspace, setWorkspace] = useState<WorkspaceDocument | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    getProject(projectId, controller.signal).then((detail) => Promise.all([detail, getWorkspace(detail.workspace_url, controller.signal)] as const))
      .then(([detail, loadedWorkspace]) => { setProject(detail); setWorkspace(loadedWorkspace); })
      .catch((caught: unknown) => { if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : "Could not load project workspace"); });
    return () => controller.abort();
  }, [projectId]);
  if (error) return <main className="fatal-page"><a className="back-link" href={`${basePath}/annotations`}>&lt; All projects</a><Notice tone="danger">{error}</Notice></main>;
  if (!project || !workspace) return <Loading fullPage label="Loading root workspace" />;
  return <LoadedWorkbench project={project} workspace={workspace} onProject={setProject} />;
}

function hexRgb(hex: string): [number, number, number] {
  return [Number.parseInt(hex.slice(1, 3), 16) / 255, Number.parseInt(hex.slice(3, 5), 16) / 255, Number.parseInt(hex.slice(5, 7), 16) / 255];
}

function LoadedWorkbench({ project, workspace: initialWorkspace, onProject }: { project: ProjectDetail; workspace: WorkspaceDocument; onProject: (project: ProjectDetail) => void }) {
  const [workspace, setWorkspace] = useState(initialWorkspace);
  const workspaceRef = useRef(workspace);
  workspaceRef.current = workspace;
  const [currentZoomId, setCurrentZoomId] = useState(workspace.root_zoom_id);
  const root = workspace.zooms.find((zoom) => zoom.id === workspace.root_zoom_id)!;
  const [embeddingId, setEmbeddingId] = useState(root.initial_embedding_id);
  const [points, setPoints] = useState<PointsPayload | null>(null);
  const pointsRef = useRef<PointsPayload | null>(null);
  pointsRef.current = points;
  const [pointsError, setPointsError] = useState("");
  const [pointsLoading, setPointsLoading] = useState(true);
  const [draftIds, setDraftIds] = useState<Set<string>>(new Set());
  const [draftKind, setDraftKind] = useState<"manual" | "lasso">("manual");
  const [draftPolygon, setDraftPolygon] = useState<[number, number][]>([]);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [focusedSelectionId, setFocusedSelectionId] = useState<string | null>(null);
  const [visibleSelectionIds, setVisibleSelectionIds] = useState<Set<string>>(new Set());
  const [colorTitle, setColorTitle] = useState("Observations");
  const [baseColors, setBaseColors] = useState<ColorResult>(() => uniformColors(0));
  const [activeTool, setActiveTool] = useState<"recipe" | "selection" | "analysis" | "marker">("recipe");
  const [openMarkerId, setOpenMarkerId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<WorkspaceObjectRef | null>(null);
  const [labelsOpen, setLabelsOpen] = useState(false);
  const [labels, setLabels] = useState(project.labels);
  const [exportState, setExportState] = useState<"idle" | "working" | "done" | "error">("idle");
  const [exportMessage, setExportMessage] = useState("");
  const [refreshMessage, setRefreshMessage] = useState("");
  const memberships = useMembershipDraft(project.memberships_url, project.csrf_token);
  const currentZoomIdRef = useRef(currentZoomId);
  const embeddingIdRef = useRef(embeddingId);
  const focusedSelectionIdRef = useRef(focusedSelectionId);
  const visibleSelectionIdsRef = useRef(visibleSelectionIds);
  const membershipStatusRef = useRef(memberships.status);
  const membershipHasLocalChangesRef = useRef(memberships.hasLocalChanges);
  currentZoomIdRef.current = currentZoomId;
  embeddingIdRef.current = embeddingId;
  focusedSelectionIdRef.current = focusedSelectionId;
  visibleSelectionIdsRef.current = visibleSelectionIds;
  membershipStatusRef.current = memberships.status;
  membershipHasLocalChangesRef.current = memberships.hasLocalChanges;
  const currentZoom = workspace.zooms.find((zoom) => zoom.id === currentZoomId) ?? root;
  const path = useMemo(() => zoomPath(workspace, currentZoom.id), [currentZoom.id, workspace]);
  const pathIds = useMemo(() => new Set(path.map((zoom) => zoom.id)), [path]);
  const usableEmbeddings = workspace.embeddings.filter((embedding) => pathIds.has(embedding.zoom_id));
  const focusedSelection = workspace.selections.find((selection) => selection.id === focusedSelectionId) ?? null;
  const selectionColors = useMemo(() => new Map(workspace.selections.map((selection, index) => [selection.id, SELECTION_COLORS[index % SELECTION_COLORS.length]])), [workspace.selections]);
  const destructiveDisabled = memberships.status !== "saved" || memberships.hasLocalChanges;
  const destructiveDisabledReason = destructiveDisabled
    ? memberships.hasLocalChanges
      ? "membership rows have unresolved local changes; wait for autosave or resolve the save error"
      : `membership state is ${memberships.status}; wait for a fully saved draft before changing dependencies`
    : "";

  const installWorkspace = useCallback((saved: WorkspaceDocument, preferred?: { zoomId?: string; embeddingId?: string }) => {
    const requestedZoomId = preferred?.zoomId ?? currentZoomIdRef.current;
    const nextZoom = saved.zooms.find((zoom) => zoom.id === requestedZoomId)
      ?? saved.zooms.find((zoom) => zoom.id === saved.root_zoom_id)
      ?? saved.zooms[0];
    if (!nextZoom) throw new Error("Workspace replacement has no root zoom");
    const nextPathIds = new Set(zoomPath(saved, nextZoom.id).map((zoom) => zoom.id));
    const nextEmbeddings = saved.embeddings.filter((embedding) => nextPathIds.has(embedding.zoom_id));
    const requestedEmbeddingId = preferred?.embeddingId ?? embeddingIdRef.current;
    const nextEmbedding = nextEmbeddings.find((embedding) => embedding.id === requestedEmbeddingId)
      ?? nextEmbeddings.find((embedding) => embedding.id === nextZoom.initial_embedding_id)
      ?? nextEmbeddings[0];
    if (!nextEmbedding) throw new Error(`Workspace replacement has no usable embedding for zoom ${nextZoom.id}`);
    const viewChanged = nextZoom.id !== currentZoomIdRef.current || nextEmbedding.id !== embeddingIdRef.current;
    const nextFocused = saved.selections.find((selection) => selection.id === focusedSelectionIdRef.current && selection.zoom_id === nextZoom.id)?.id ?? null;
    const knownSelections = new Set(saved.selections.map((selection) => selection.id));
    const nextVisible = new Set([...visibleSelectionIdsRef.current].filter((id) => knownSelections.has(id)));
    workspaceRef.current = saved;
    currentZoomIdRef.current = nextZoom.id;
    embeddingIdRef.current = nextEmbedding.id;
    focusedSelectionIdRef.current = nextFocused;
    visibleSelectionIdsRef.current = nextVisible;
    setWorkspace(saved);
    setCurrentZoomId(nextZoom.id);
    setEmbeddingId(nextEmbedding.id);
    setFocusedSelectionId(nextFocused);
    setVisibleSelectionIds(nextVisible);
    setOpenMarkerId((current) => saved.marker_programs.some((program) => program.id === current) ? current : null);
    if (viewChanged) {
      setPoints(null);
      setPointsLoading(true);
      setActiveIndex(null);
      setDraftIds(new Set());
      setDraftPolygon([]);
      setDraftKind("manual");
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setPointsLoading(true); setPointsError(""); setActiveIndex(null);
    getZoomPoints(project.workspace_url, currentZoom.id, embeddingId, controller.signal).then((loaded) => {
      if (loaded.observation_ids.length !== loaded.coordinates.length) throw new Error("Point coordinates do not match the observation index");
      setPoints(loaded); setBaseColors(uniformColors(loaded.observation_ids.length)); setColorTitle("Observations");
    }).catch((caught: unknown) => { if (!controller.signal.aborted) { setPoints(null); setPointsError(caught instanceof Error ? caught.message : "Could not load zoom points"); } }).finally(() => { if (!controller.signal.aborted) setPointsLoading(false); });
    return () => controller.abort();
  }, [currentZoom.id, embeddingId, project.workspace_url]);

  const navigate = useCallback((zoomId: string) => {
    const next = workspaceRef.current.zooms.find((zoom) => zoom.id === zoomId);
    if (!next) return;
    currentZoomIdRef.current = next.id;
    embeddingIdRef.current = next.initial_embedding_id;
    setCurrentZoomId(next.id); setEmbeddingId(next.initial_embedding_id); setPoints(null); setPointsLoading(true); setDraftIds(new Set()); setDraftPolygon([]); setDraftKind("manual"); setActiveTool("recipe");
    setFocusedSelectionId((current) => {
      const reconciled = workspaceRef.current.selections.some((selection) => selection.id === current && selection.zoom_id === next.id) ? current : null;
      focusedSelectionIdRef.current = reconciled;
      return reconciled;
    });
  }, []);

  const focusSelection = useCallback((selection: SavedSelection) => {
    if (selection.zoom_id !== currentZoomId) navigate(selection.zoom_id);
    setFocusedSelectionId(selection.id);
    setVisibleSelectionIds((current) => new Set(current).add(selection.id));
  }, [currentZoomId, navigate]);

  const createSelectionAction = useCallback(async (request: SelectionCreate) => {
    const saved = await createSelection(project.workspace_url, workspaceRef.current.revision, project.csrf_token, request);
    installWorkspace(saved);
  }, [installWorkspace, project.csrf_token, project.workspace_url]);
  const createZoomAction = useCallback(async (request: ZoomCreate) => {
    const saved = await createZoom(project.workspace_url, workspaceRef.current.revision, project.csrf_token, request);
    const created = saved.zooms.find((zoom) => zoom.id === request.id);
    installWorkspace(saved, { zoomId: created?.id, embeddingId: created?.initial_embedding_id });
    setFocusedSelectionId(null);
    if (created) setActiveTool("recipe");
  }, [installWorkspace, project.csrf_token, project.workspace_url]);
  const importAction = useCallback(async (request: ClusteringImport) => {
    installWorkspace(await importClustering(project.workspace_url, workspaceRef.current.revision, project.csrf_token, request));
  }, [installWorkspace, project.csrf_token, project.workspace_url]);
  const computeAction = useCallback(async (request: LocalAnalysisCreate) => {
    installWorkspace(await computeLocalAnalysis(project.workspace_url, workspaceRef.current.revision, project.csrf_token, request), { embeddingId: request.embedding_id });
  }, [installWorkspace, project.csrf_token, project.workspace_url]);
  const createMarkerAction = useCallback(async (request: MarkerProgramCreate): Promise<MarkerProgramResult> => {
    const result = await createMarkerProgram(project.workspace_url, workspaceRef.current.revision, project.csrf_token, request);
    try {
      installWorkspace(await getWorkspace(project.workspace_url));
    } catch (caught) {
      setRefreshMessage(`Marker program ${result.program.id} was committed, but workspace refresh failed: ${caught instanceof Error ? caught.message : "refresh required"}. Do not rerun the marker mutation.`);
    }
    return result;
  }, [installWorkspace, project.csrf_token, project.workspace_url]);
  const loadMarkerAction = useCallback((programId: string) => getMarkerProgramResult(project.workspace_url, programId), [project.workspace_url]);
  const deletionPreviewAction = useCallback((target: WorkspaceObjectRef) => {
    if (membershipStatusRef.current !== "saved" || membershipHasLocalChangesRef.current) throw new Error(`Deletion is locked while membership state is ${membershipStatusRef.current}`);
    return previewDeletion(project.workspace_url, project.csrf_token, target);
  }, [project.csrf_token, project.workspace_url]);
  const deletionConfirmAction = useCallback(async (impact: DeletionImpact) => {
    if (membershipStatusRef.current !== "saved" || membershipHasLocalChangesRef.current) throw new Error(`Deletion is locked while membership state is ${membershipStatusRef.current}`);
    const saved = await confirmDeletion(project.workspace_url, impact.workspace_revision, project.csrf_token, {
      ...impact.requested,
      confirmed_removed: impact.removed,
      expected_membership_revision: impact.membership_revision,
      impact_sha256: impact.impact_sha256
    });
    try {
      installWorkspace(saved);
    } catch (caught) {
      setRefreshMessage(`Deletion was committed, but workspace reconciliation failed: ${caught instanceof Error ? caught.message : "refresh required"}. Do not repeat the deletion.`);
    }
    try {
      await memberships.reload();
    } catch (caught) {
      setRefreshMessage(`Deletion was committed, but membership refresh failed: ${caught instanceof Error ? caught.message : "refresh required"}. Do not repeat the deletion.`);
    }
  }, [installWorkspace, memberships, project.csrf_token, project.workspace_url]);

  const indexById = useMemo(() => new Map((points?.observation_ids ?? []).map((id, index) => [id, index])), [points]);
  const draftIndices = useMemo(() => new Set([...draftIds].flatMap((id) => { const index = indexById.get(id); return index === undefined ? [] : [index]; })), [draftIds, indexById]);
  const focusedIndices = useMemo(() => new Set((focusedSelection?.observation_ids ?? []).flatMap((id) => { const index = indexById.get(id); return index === undefined ? [] : [index]; })), [focusedSelection, indexById]);
  const blendedColors = useMemo(() => {
    if (!points) return baseColors.colors;
    const colors = new Float32Array(baseColors.colors);
    const sums = new Float32Array(points.observation_ids.length * 3);
    const counts = new Uint16Array(points.observation_ids.length);
    workspace.selections.forEach((selection) => {
      if (!visibleSelectionIds.has(selection.id)) return;
      const rgb = hexRgb(selectionColors.get(selection.id) ?? "#2463a8");
      const weight = selection.id === focusedSelectionId ? 2 : 1;
      selection.observation_ids.forEach((id) => {
        const index = indexById.get(id);
        if (index === undefined) return;
        sums[index * 3] += rgb[0] * weight; sums[index * 3 + 1] += rgb[1] * weight; sums[index * 3 + 2] += rgb[2] * weight; counts[index] += weight;
      });
    });
    for (let index = 0; index < counts.length; index += 1) {
      if (counts[index] === 0) { colors[index * 4 + 3] = 0.34; continue; }
      colors[index * 4] = sums[index * 3] / counts[index]; colors[index * 4 + 1] = sums[index * 3 + 1] / counts[index]; colors[index * 4 + 2] = sums[index * 3 + 2] / counts[index]; colors[index * 4 + 3] = 0.9;
    }
    return colors;
  }, [baseColors.colors, focusedSelectionId, indexById, points, selectionColors, visibleSelectionIds, workspace.selections]);

  const applyOverlay = useCallback((result: MarkerProgramResult, mode: "per_cell" | "cluster_mean") => {
    const currentPoints = pointsRef.current;
    if (!currentPoints) return;
    const valuesById = new Map(result.observation_ids.map((id, index) => [id, mode === "per_cell" ? result.per_cell_scores[index] : result.cluster_mean_by_cell[index]]));
    setBaseColors(colorValues(currentPoints.observation_ids.map((id) => valuesById.get(id) ?? null), currentPoints.observation_ids.length));
    setColorTitle(`${result.program.name} / ${mode === "per_cell" ? "per-cell score" : "cluster mean"}`);
  }, []);
  const updateBaseColor = useCallback((title: string, values: Scalar[] | undefined) => {
    const currentPoints = pointsRef.current;
    if (!currentPoints) return;
    setColorTitle(title);
    setBaseColors(colorValues(values, currentPoints.observation_ids.length));
  }, []);

  useEffect(() => {
    if (!destructiveDisabled) return;
    setDeleteTarget(null);
    setLabelsOpen(false);
  }, [destructiveDisabled]);

  const refreshCommittedState = useCallback(async () => {
    if (memberships.hasLocalChanges || ["dirty", "saving", "conflict"].includes(memberships.status)) {
      setRefreshMessage(`Server refresh remains blocked while membership state is ${memberships.status}; resolve local changes first.`);
      return;
    }
    try {
      const [savedWorkspace, savedProject] = await Promise.all([
        getWorkspace(project.workspace_url),
        getProject(project.project_id)
      ]);
      installWorkspace(savedWorkspace);
      onProject(savedProject);
      setLabels(savedProject.labels);
      await memberships.reloadAfterCommit();
      setRefreshMessage("");
    } catch (caught) {
      setRefreshMessage(`A committed mutation still needs refresh: ${caught instanceof Error ? caught.message : "refresh failed"}. Do not repeat the mutation.`);
    }
  }, [installWorkspace, memberships, onProject, project.project_id, project.workspace_url]);

  if (!points && pointsLoading) return <Loading fullPage label="Loading all root observations" />;
  return <main className="workbench redesign-workbench">
    <header className="workbench-header"><a className="compact-brand" href={`${basePath}/annotations`}>Eyck</a><div className="project-title"><h1>{project.title}</h1><code>{project.project_id}</code><Badge>{formatCount(project.n_observations)} source cells</Badge></div><div className="header-actions"><SaveIndicator status={memberships.status} /><Button disabled={destructiveDisabled} title={destructiveDisabled ? destructiveDisabledReason : "Edit label identities"} onClick={() => { if (!destructiveDisabled) setLabelsOpen(true); }}>Label DAG</Button><Button tone="primary" disabled={memberships.status !== "saved" || exportState === "working"} onClick={() => {
      setExportState("working"); setExportMessage("");
      void exportAnnotations(project.export_url, memberships.revision, project.csrf_token).then((result) => { setExportState("done"); setExportMessage(`Exported ${formatCount(result.membership_rows)} membership rows.`); }).catch((caught: unknown) => { setExportState("error"); setExportMessage(caught instanceof Error ? caught.message : "Export failed"); });
    }}>{exportState === "working" ? "Exporting..." : "Export"}</Button><RestartButton csrfToken={project.csrf_token} available={project.restart_available} /></div></header>

    <nav className="zoom-breadcrumb" aria-label="Zoom path"><span>Population path</span>{path.map((zoom, index) => <button type="button" className={zoom.id === currentZoom.id ? "current" : ""} key={zoom.id} onClick={() => navigate(zoom.id)}><b>{index === 0 ? "Root" : zoom.name}</b><small>{formatCount(zoom.observation_count)} / {formatFraction(zoom.fraction_of_root)} root{zoom.parent_id ? ` / ${formatFraction(zoom.fraction_of_parent)} parent` : ""}</small></button>)}</nav>

    {(memberships.status === "conflict" || memberships.status === "error" || exportMessage || refreshMessage) && <div className="issue-strip"><div className={`issue ${memberships.status === "conflict" || memberships.status === "error" || exportState === "error" || refreshMessage ? "issue-error" : ""}`}><span>{refreshMessage || memberships.message || exportMessage}</span><div className="issue-actions">{memberships.status === "conflict" && <><Button onClick={() => void memberships.mergeLatest()}>Merge local edits</Button><Button onClick={() => void memberships.discardLocal()}>Discard local</Button></>}{memberships.status === "error" && !refreshMessage && <Button onClick={memberships.retry}>Retry save</Button>}{refreshMessage && <Button disabled={memberships.hasLocalChanges || ["dirty", "saving", "conflict"].includes(memberships.status)} onClick={() => void refreshCommittedState()}>Refresh committed state</Button>}{exportMessage && !refreshMessage && <Button onClick={() => setExportMessage("")}>Dismiss</Button>}</div></div></div>}

    <div className="workbench-grid redesign-grid">
      <HierarchyNavigator workspace={workspace} currentZoomId={currentZoom.id} focusedSelectionId={focusedSelectionId} visibleSelectionIds={visibleSelectionIds} selectionColors={selectionColors} draftCount={draftIds.size} memberships={memberships.draft} destructiveDisabled={destructiveDisabled} destructiveDisabledReason={destructiveDisabledReason} onNavigate={navigate} onFocusSelection={focusSelection} onToggleSelection={(id) => setVisibleSelectionIds((current) => { const next = new Set(current); if (next.has(id)) next.delete(id); else next.add(id); return next; })} onDelete={(kind, id) => { if (!destructiveDisabled) setDeleteTarget({ kind, id }); }} onOpenMarker={(id) => { const program = workspace.marker_programs.find((item) => item.id === id); if (program && program.zoom_id !== currentZoom.id) navigate(program.zoom_id); setOpenMarkerId(id); setActiveTool("marker"); }} />
      <section className="workspace-center">
        <header className="population-header"><div><span>{currentZoom.parent_id ? "Current zoom" : "Immutable source root"}</span><h2>{currentZoom.name}</h2><p>{currentZoom.parent_id ? `${formatFraction(currentZoom.fraction_of_parent)} of parent, ${formatFraction(currentZoom.fraction_of_root)} of root.` : `All ${formatCount(currentZoom.observation_count)} loaded observations. No implicit filter.`}</p></div><div className="population-stats"><strong>{formatCount(currentZoom.observation_count)}</strong><span>cells</span><code>{currentZoom.id}</code></div></header>
        <div className="plot-composition">
          <aside className="display-drawer"><div className="section-heading"><span>Embedding</span></div><Select value={embeddingId} onChange={(event) => { embeddingIdRef.current = event.target.value; setEmbeddingId(event.target.value); setPoints(null); setPointsLoading(true); setDraftIds(new Set()); setDraftPolygon([]); }} aria-label="Active embedding">{usableEmbeddings.map((embedding) => <option value={embedding.id} key={embedding.id}>{embedding.name} [{embedding.id}]{embedding.id === currentZoom.initial_embedding_id ? " (initial)" : ""}</option>)}</Select>{points && <ColoringControls metadataColumns={project.metadata_columns} modalities={project.modalities} points={points} featuresUrl={project.features_url} onColor={updateBaseColor} />}<div className="compact-legend"><span>{colorTitle}</span>{baseColors.legend.slice(0, 8).map((item) => <div key={item.label}><i style={{ background: item.color }} />{item.label}</div>)}</div><div className="selection-layer-key"><span>Visible selection layers</span>{workspace.selections.filter((item) => visibleSelectionIds.has(item.id)).map((item) => <button type="button" key={item.id} onClick={() => focusSelection(item)}><i style={{ background: selectionColors.get(item.id) }} />{item.name}</button>)}</div></aside>
          <div className="plot-panel">{pointsLoading && <div className="plot-loading"><Loading label="Loading embedding coordinates" /></div>}{pointsError && <Notice tone="danger">{pointsError}</Notice>}{points && <Scatterplot coordinates={points.coordinates} colors={blendedColors} selectedIndices={draftIndices} focusedIndices={focusedIndices} onSingleSelect={(index) => { setDraftIds(new Set([points.observation_ids[index]])); setDraftKind("manual"); setDraftPolygon([]); setActiveIndex(index); }} onLasso={(indices, mode: SelectionMode, polygon) => {
            const incoming = indices.map((index) => points.observation_ids[index]); const next = mode === "replace" ? new Set<string>() : new Set(draftIds); incoming.forEach((id) => { if (mode === "subtract") next.delete(id); else next.add(id); }); setDraftIds(next); setDraftKind(mode === "replace" ? "lasso" : "manual"); setDraftPolygon(mode === "replace" ? polygon : []); const first = next.values().next().value; setActiveIndex(first === undefined ? null : points.observation_ids.indexOf(first)); setActiveTool("selection");
          }} onClearSelection={() => { setDraftIds(new Set()); setDraftPolygon([]); setDraftKind("manual"); setActiveIndex(null); }} />}</div>
        </div>
        <nav className="tool-tabs" aria-label="Workspace tools">{([ ["recipe", "Path recipe"], ["selection", "Selections / zoom"], ["analysis", "Analysis"], ["marker", "Markers"] ] as const).map(([id, label]) => <button type="button" className={activeTool === id ? "active" : ""} key={id} onClick={() => setActiveTool(id)}>{label}</button>)}</nav>
        <div className="tool-workspace">{activeTool === "recipe" && <RecipePanel workspace={workspace} zoomId={currentZoom.id} />}{activeTool === "selection" && points && <SelectionPanel workspace={workspace} workspaceUrl={project.workspace_url} zoomId={currentZoom.id} embeddingId={embeddingId} points={points} draftIds={[...draftIds]} draftKind={draftKind} draftPolygon={draftPolygon} onCreateSelection={createSelectionAction} onCreateZoom={createZoomAction} onDraftSaved={(id) => { setFocusedSelectionId(id); setVisibleSelectionIds((current) => new Set(current).add(id)); setDraftIds(new Set()); setDraftPolygon([]); }} onClearDraft={() => { setDraftIds(new Set()); setDraftPolygon([]); setDraftKind("manual"); }} />}{activeTool === "analysis" && <AnalysisPanel project={project} workspace={workspace} zoomId={currentZoom.id} onImport={importAction} onCompute={computeAction} onEmbeddingCreated={(id) => { embeddingIdRef.current = id; setEmbeddingId(id); }} />}{activeTool === "marker" && <MarkerPanel project={project} workspace={workspace} zoomId={currentZoom.id} openProgramId={openMarkerId} onCreateProgram={createMarkerAction} onLoadProgram={loadMarkerAction} onCreateSelection={createSelectionAction} onOverlay={applyOverlay} />}</div>
      </section>
      {points && <AnnotationPanel labels={labels} points={points} activeIndex={activeIndex} focusedSelection={focusedSelection} zoom={currentZoom} draft={memberships.draft} origin={memberships.origin} destructiveDisabled={destructiveDisabled} destructiveDisabledReason={destructiveDisabledReason} onOpenLabels={() => { if (!destructiveDisabled) setLabelsOpen(true); }} onApply={(entityId, labelId, state) => { if (!focusedSelection) return; memberships.applyState(focusedSelection.observation_ids, entityId, labelId, state, { selectionId: focusedSelection.id, zoomId: currentZoom.id }); }} />}
    </div>
    <DeleteDialog target={deleteTarget} onPreview={deletionPreviewAction} onConfirm={deletionConfirmAction} onClose={() => setDeleteTarget(null)} />
    <LabelDagEditor open={labelsOpen} labelsUrl={project.labels_url} csrfToken={project.csrf_token} onClose={() => setLabelsOpen(false)} onSaved={async (saved: LabelState) => {
      setLabels(labelDescriptors(saved));
      const [membershipRefresh, projectRefresh] = await Promise.allSettled([memberships.reload(), getProject(project.project_id)]);
      if (projectRefresh.status === "fulfilled") onProject(projectRefresh.value);
      const failures = [membershipRefresh, projectRefresh].filter((result) => result.status === "rejected") as PromiseRejectedResult[];
      if (failures.length > 0) setRefreshMessage(`Labels were committed, but follow-up refresh failed: ${failures.map((result) => result.reason instanceof Error ? result.reason.message : String(result.reason)).join("; ")}. Do not repeat the label mutation.`);
    }} />
  </main>;
}

export default function App() {
  const projectId = projectIdFromPath();
  return projectId ? <Workbench projectId={projectId} /> : <Dashboard />;
}
