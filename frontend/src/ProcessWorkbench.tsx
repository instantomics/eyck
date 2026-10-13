import { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import {
  ApiError,
  basePath,
  exportProcessSet,
  getFeatureValues,
  getObservationColumn,
  getPoints,
  getProcessIndex,
  getProcessSet,
  getProject,
  previewProcess,
  putProcessSet,
  reviewProcessSet,
  searchFeatures
} from "./api";
import { displayColors, PALETTES, type DisplayOptions, type Palette } from "./color";
import { Scatterplot, type DrawMode, type Projection, type SelectionMode, type ToScreen, type ViewState } from "./Scatterplot";
import { dragWeights, insertVertex, isClosed, minVertices, moved, openVertices, smoothed, withVertices, type XY } from "./shapes";
import type {
  FeatureDescriptor,
  PointsPayload,
  ProcessDocument,
  ProcessFeature,
  ProcessIndex,
  ProjectDetail,
  Scalar,
  ViewDescriptor
} from "./types";
import { Badge, Button, Field, Loading, Notice, Select } from "./ui";

type Tool = "pan" | "curve" | "include" | "exclude" | "edit" | "highlight";
/** A process colour source with no ID follows the process being drawn. */
type ColorSource =
  | { kind: "process"; process: string | null }
  | { kind: "column"; column: string }
  | { kind: "gene"; feature: FeatureDescriptor };
type SaveStatus = "saved" | "dirty" | "saving" | "conflict" | "error";
type Layout = "single" | "pair" | "grid" | "focus";
type Display = DisplayOptions & { pointSize: number };
interface PanelState {
  embeddingId: string;
  color: ColorSource;
  blend: ColorSource | null;
}
interface Drag {
  pointerId: number;
  featureId: string;
  original: XY[];
  weights: number[];
  start: XY;
  fromScreen: Projection["fromScreen"];
  svg: SVGSVGElement;
  recorded: boolean;
}

const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const NO_INDICES = new Set<number>();
const DEFAULT_DISPLAY: Display = { palette: "grey_blue", upperPercentile: 99, opacity: 0.82, pointSize: 4 };
const LAYOUTS: { id: Layout; label: string; panels: number; title: string }[] = [
  { id: "single", label: "1", panels: 1, title: "One plot" },
  { id: "pair", label: "2", panels: 2, title: "Two plots side by side" },
  { id: "grid", label: "2x2", panels: 4, title: "Four plots in a grid" },
  { id: "focus", label: "1+4", panels: 5, title: "One large plot above four small ones" }
];
const MAX_PANELS = 5;
/** Smaller plots draw smaller points, so that a dense embedding does not saturate. */
function pointScale(layout: Layout, position: number): number {
  if (layout === "focus" && position > 0) return 0.55;
  return { single: 1, pair: 0.85, grid: 0.7, focus: 1 }[layout];
}
const TOOLS: { id: Tool; label: string; key: string; title: string; hint: string }[] = [
  { id: "pan", label: "Pan", key: "p", title: "Pan and zoom (P)", hint: "Drag to pan, scroll to zoom" },
  { id: "curve", label: "Curve", key: "c", title: "Draw a curve from its 0 end to its 1 end (C)", hint: "Draw from the 0 end towards the 1 end" },
  { id: "include", label: "Region", key: "r", title: "Draw a region of cells that receive this process (R)", hint: "Draw around cells that receive this process" },
  { id: "exclude", label: "Exclude", key: "x", title: "Draw a region of cells that do not receive this process (X)", hint: "Draw around cells to mark not applicable" },
  { id: "edit", label: "Edit", key: "e", title: "Select, move and reshape drawings (E)", hint: "Click a drawing; drag its points or lines" },
  { id: "highlight", label: "Highlight", key: "h", title: "Lasso cells to highlight them in every plot (H)", hint: "Lasso cells; Shift adds, Alt removes" }
];
const HISTORY_LIMIT = 200;

function readDisplay(): Display {
  try {
    return { ...DEFAULT_DISPLAY, ...JSON.parse(readSetting("eyck-process-display", "{}")) };
  } catch {
    return DEFAULT_DISPLAY;
  }
}

function readSetting(key: string, fallback: string): string {
  try {
    return window.localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function writeSetting(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Settings are a convenience only.
  }
}

/** The remembered layout and plots of a project; plots on unknown embeddings fall back to the first. */
function readLayout(projectId: string, embeddings: ViewDescriptor[]): { layout: Layout; panels: PanelState[]; linked: boolean } {
  const known = new Set(embeddings.map((item) => item.id));
  const fallback: PanelState = { embeddingId: embeddings[0].id, color: { kind: "process", process: null }, blend: null };
  try {
    const stored = JSON.parse(readSetting(`eyck-process-layout:${projectId}`, "{}")) as { layout?: Layout; panels?: PanelState[]; linked?: boolean };
    const panels = Array.from({ length: MAX_PANELS }, (_, position) => {
      const panel = stored.panels?.[position];
      return panel && known.has(panel.embeddingId) ? panel : fallback;
    });
    const layout = LAYOUTS.find((item) => item.id === stored.layout)?.id ?? "single";
    return { layout, panels, linked: stored.linked ?? true };
  } catch {
    return { layout: "single", panels: Array.from({ length: MAX_PANELS }, () => fallback), linked: true };
  }
}

function newFeatureId(): string {
  return `f-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function simplify(points: [number, number][], toScreen: ToScreen, minPixels = 4): [number, number][] {
  const kept: [number, number][] = [];
  let last: [number, number] | null = null;
  points.forEach((point, index) => {
    const screen = toScreen(point[0], point[1]);
    if (last === null || index === points.length - 1 || Math.hypot(screen[0] - last[0], screen[1] - last[1]) >= minPixels) {
      kept.push(point);
      last = screen;
    }
  });
  return kept;
}

function drawMode(tool: Tool): DrawMode | null {
  if (tool === "curve") return "curve";
  if (tool === "include" || tool === "exclude") return "region";
  if (tool === "highlight") return "select";
  return null;
}

function svgPoint(event: ReactMouseEvent<SVGElement>, svg: SVGSVGElement): XY {
  const rect = svg.getBoundingClientRect();
  return [event.clientX - rect.left, event.clientY - rect.top];
}

export function ProcessWorkbench({ projectId }: { projectId: string }) {
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [index, setIndex] = useState<ProcessIndex | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    Promise.all([getProject(projectId, controller.signal), getProcessIndex(projectId, controller.signal)])
      .then(([detail, processes]) => { setProject(detail); setIndex(processes); })
      .catch((caught: unknown) => { if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : "Could not load project"); });
    return () => controller.abort();
  }, [projectId]);
  if (error) return <main className="fatal-page"><a className="back-link" href={`${basePath}/annotations`}>&lt; All projects</a><Notice tone="danger">{error}</Notice></main>;
  if (!project || !index) return <Loading fullPage label="Loading process drawing" />;
  return <ProcessEditor project={project} index={index} onIndex={setIndex} />;
}

/** Points of every requested embedding, each fetched once. */
function usePointsCache(projectId: string, embeddingIds: string[], onError: (message: string) => void): Record<string, PointsPayload> {
  const [cache, setCache] = useState<Record<string, PointsPayload>>({});
  const requested = useRef(new Set<string>());
  const key = [...new Set(embeddingIds)].sort().join("\n");
  useEffect(() => {
    key.split("\n").filter(Boolean).forEach((id) => {
      if (requested.current.has(id)) return;
      requested.current.add(id);
      const url = new URL(publicPoints(projectId, id), window.location.href);
      getPoints(url.toString()).then((loaded) => setCache((current) => ({ ...current, [id]: loaded }))).catch((caught: unknown) => {
        requested.current.delete(id);
        onError(caught instanceof Error ? caught.message : `Could not load embedding ${id}`);
      });
    });
  }, [key, onError, projectId]);
  return cache;
}

function ProcessEditor({ project, index, onIndex }: { project: ProjectDetail; index: ProcessIndex; onIndex: (index: ProcessIndex) => void }) {
  const projectId = project.project_id;
  const initialSet = new URLSearchParams(window.location.search).get("set") ?? readSetting(`eyck-process-set:${projectId}`, "a");
  const [setId, setSetId] = useState(initialSet);
  const [setInput, setSetInput] = useState(initialSet);
  const [annotator, setAnnotator] = useState(() => readSetting("eyck-annotator", ""));
  const [document, setDocument] = useState<ProcessDocument | null>(null);
  const [features, setFeatures] = useState<ProcessFeature[]>([]);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("saved");
  const [message, setMessage] = useState("");
  const [extraProcesses, setExtraProcesses] = useState<string[]>([]);
  const [newProcess, setNewProcess] = useState("");
  const [processId, setProcessId] = useState(index.processes[0]?.id ?? "");
  const [tool, setTool] = useState<Tool>("pan");
  const [display, setDisplay] = useState(readDisplay);
  const [processValues, setProcessValues] = useState<(number | null)[] | null>(null);
  const [assigned, setAssigned] = useState<number | null>(null);
  const [previewError, setPreviewError] = useState("");
  const [busy, setBusy] = useState(false);
  const [initialLayout] = useState(() => readLayout(projectId, index.embeddings));
  const [layout, setLayout] = useState<Layout>(initialLayout.layout);
  const [panels, setPanels] = useState<PanelState[]>(initialLayout.panels);
  const [linked, setLinked] = useState(initialLayout.linked);
  const [activePanel, setActivePanel] = useState(0);
  const [views, setViews] = useState<Record<string, ViewState>>({});
  const [highlight, setHighlight] = useState<Set<number>>(NO_INDICES);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [editRadius, setEditRadius] = useState(() => Number(readSetting("eyck-edit-radius", "40")) || 0);
  const [geneList, setGeneList] = useState("");
  const [historySize, setHistorySize] = useState({ past: 0, future: 0 });
  const revisionRef = useRef("");
  const versionRef = useRef(0);
  const savedVersionRef = useRef(0);
  const savingRef = useRef(false);
  const featuresRef = useRef(features);
  featuresRef.current = features;
  const historyRef = useRef<{ past: ProcessFeature[][]; future: ProcessFeature[][] }>({ past: [], future: [] });
  const dragRef = useRef<Drag | null>(null);
  const [dragging, setDragging] = useState(false);

  const nPanels = LAYOUTS.find((item) => item.id === layout)?.panels ?? 1;
  const visiblePanels = panels.slice(0, nPanels);
  const active = Math.min(activePanel, nPanels - 1);
  const activeState = panels[active];
  const embeddingById = useMemo(() => new Map(index.embeddings.map((item) => [item.id, item])), [index.embeddings]);

  useEffect(() => {
    writeSetting(`eyck-process-layout:${projectId}`, JSON.stringify({ layout, panels, linked }));
  }, [layout, linked, panels, projectId]);

  const onError = useCallback((text: string) => setMessage(text), []);
  const pointsCache = usePointsCache(projectId, visiblePanels.map((panel) => panel.embeddingId), onError);
  const nObservations = Object.values(pointsCache)[0]?.observation_ids.length ?? project.n_observations;

  const resetHistory = () => {
    historyRef.current = { past: [], future: [] };
    setHistorySize({ past: 0, future: 0 });
  };

  useEffect(() => {
    const controller = new AbortController();
    setDocument(null);
    getProcessSet(projectId, setId, controller.signal).then((loaded) => {
      revisionRef.current = loaded.revision;
      versionRef.current = 0;
      savedVersionRef.current = 0;
      setDocument(loaded);
      setFeatures(loaded.features);
      setSaveStatus("saved");
      setMessage("");
      setSelectedId(null);
      resetHistory();
    }).catch((caught: unknown) => { if (!controller.signal.aborted) setMessage(caught instanceof Error ? caught.message : "Could not load drawings"); });
    writeSetting(`eyck-process-set:${projectId}`, setId);
    const url = new URL(window.location.href);
    url.searchParams.set("set", setId);
    window.history.replaceState(null, "", url.toString());
    return () => controller.abort();
  }, [projectId, setId]);

  const save = useCallback(async () => {
    if (savingRef.current) return;
    const version = versionRef.current;
    if (version === savedVersionRef.current) return;
    savingRef.current = true;
    setSaveStatus("saving");
    try {
      const saved = await putProcessSet(projectId, setId, revisionRef.current, project.csrf_token, featuresRef.current);
      revisionRef.current = saved.revision;
      savedVersionRef.current = version;
      setDocument(saved);
      if (versionRef.current === version) {
        setFeatures(saved.features);
        setSaveStatus("saved");
      } else {
        setSaveStatus("dirty");
      }
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 409) {
        setSaveStatus("conflict");
        setMessage("These drawings were changed elsewhere. Reload to continue from the saved version.");
      } else {
        setSaveStatus("error");
        setMessage(caught instanceof Error ? caught.message : "Save failed");
      }
    } finally {
      savingRef.current = false;
    }
  }, [project.csrf_token, projectId, setId]);

  // Saving waits until a drag ends, so one drag is one saved revision.
  useEffect(() => {
    if (saveStatus !== "dirty" || dragging) return;
    const timer = window.setTimeout(() => void save(), 500);
    return () => window.clearTimeout(timer);
  }, [dragging, features, save, saveStatus]);

  /** Change the drawings; `checkpoint` false merges the change into the last undo step. */
  const edit = useCallback((update: (current: ProcessFeature[]) => ProcessFeature[], checkpoint = true) => {
    if (saveStatus === "conflict") return;
    if (checkpoint) {
      const history = historyRef.current;
      history.past.push(featuresRef.current);
      if (history.past.length > HISTORY_LIMIT) history.past.shift();
      history.future = [];
      setHistorySize({ past: history.past.length, future: 0 });
    }
    versionRef.current += 1;
    setFeatures(update);
    setSaveStatus("dirty");
  }, [saveStatus]);

  const travel = useCallback((direction: "undo" | "redo") => {
    if (saveStatus === "conflict") return;
    const history = historyRef.current;
    const target = direction === "undo" ? history.past.pop() : history.future.pop();
    if (!target) return;
    (direction === "undo" ? history.future : history.past).push(featuresRef.current);
    setHistorySize({ past: history.past.length, future: history.future.length });
    versionRef.current += 1;
    setFeatures(target);
    setSaveStatus("dirty");
    setSelectedId((current) => (current && target.some((item) => item.id === current) ? current : null));
  }, [saveStatus]);

  const processFeatures = useMemo(() => features.filter((item) => item.properties.process_id === processId), [features, processId]);
  const deferredProcessFeatures = useDeferredValue(processFeatures);
  const regionOnly = processFeatures.length > 0 && !processFeatures.some((item) => item.properties.role === "curve");

  useEffect(() => {
    if (!processId) return;
    if (deferredProcessFeatures.length === 0) {
      setProcessValues(null); setAssigned(0); setPreviewError(""); return;
    }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      previewProcess(projectId, project.csrf_token, processId, deferredProcessFeatures).then((result) => {
        if (cancelled) return;
        setProcessValues(result.values); setAssigned(result.assigned); setPreviewError("");
      }).catch((caught: unknown) => { if (!cancelled) setPreviewError(caught instanceof Error ? caught.message : "Projection failed"); });
    }, 200);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [deferredProcessFeatures, processId, project.csrf_token, projectId]);

  const loadValues = useCallback((source: ColorSource, signal: AbortSignal): Promise<Scalar[]> => {
    if (source.kind === "column") return getObservationColumn(projectId, source.column, signal).then((result) => result.values);
    if (source.kind === "gene") return getFeatureValues(project.features_url, source.feature.feature_index, signal).then((result) => result.values);
    const id = source.process ?? processId;
    const drawn = featuresRef.current.filter((item) => item.properties.process_id === id);
    if (drawn.length === 0) return Promise.resolve([]);
    return previewProcess(projectId, project.csrf_token, id, drawn).then((result) => result.values);
  }, [processId, project.csrf_token, project.features_url, projectId]);

  const updateDisplay = (patch: Partial<Display>) => setDisplay((current) => {
    const next = { ...current, ...patch };
    writeSetting("eyck-process-display", JSON.stringify(next));
    return next;
  });
  const updatePanel = useCallback((position: number, patch: Partial<PanelState>) => setPanels((current) => current.map((panel, at) => (at === position ? { ...panel, ...patch } : panel))), []);

  const processIds = useMemo(() => {
    const declared = index.processes.map((item) => item.id);
    const drawn = features.map((item) => item.properties.process_id);
    return [...new Set([...declared, ...extraProcesses, ...drawn])];
  }, [extraProcesses, features, index.processes]);
  const descriptor = index.processes.find((item) => item.id === processId);

  const onDraw = useCallback((embeddingId: string, drawn: [number, number][], toScreen: ToScreen) => {
    if (tool !== "curve" && tool !== "include" && tool !== "exclude") return;
    if (!annotator.trim()) { setMessage("Enter your annotator name before drawing."); return; }
    if (!processId) { setMessage("Choose a process before drawing."); return; }
    const simplified = simplify(drawn, toScreen);
    const base = { process_id: processId, embedding_id: embeddingId, annotator: annotator.trim(), annotation_set: setId };
    let feature: ProcessFeature;
    if (tool === "curve") {
      if (simplified.length < 2) return;
      const otherEmbedding = processFeatures.find((item) => item.properties.role === "curve" && item.properties.embedding_id !== embeddingId);
      if (otherEmbedding) { setMessage(`Curves of ${processId} are drawn on ${otherEmbedding.properties.embedding_id}; switch embedding or delete them first.`); return; }
      feature = { type: "Feature", id: newFeatureId(), geometry: { type: "LineString", coordinates: simplified }, properties: { ...base, role: "curve", value_start: 0, value_end: 1 } };
    } else {
      if (simplified.length < 3) return;
      const ring = [...simplified, simplified[0]];
      feature = { type: "Feature", id: newFeatureId(), geometry: { type: "Polygon", coordinates: [ring] }, properties: { ...base, role: "region", region_mode: tool } };
    }
    setMessage("");
    setSelectedId(feature.id);
    edit((current) => [...current, feature]);
  }, [annotator, edit, processFeatures, processId, setId, tool]);

  const onHighlight = useCallback((indices: number[], mode: SelectionMode) => setHighlight((current) => {
    if (mode === "replace") return new Set(indices);
    const next = new Set(current);
    indices.forEach((item) => (mode === "add" ? next.add(item) : next.delete(item)));
    return next;
  }), []);

  const updateFeature = (id: string, patch: Partial<ProcessFeature["properties"]>) => edit((current) => current.map((item) => (item.id === id ? { ...item, properties: { ...item.properties, ...patch } } : item)));
  const reverseCurve = (id: string) => edit((current) => current.map((item) => (item.id === id && item.geometry.type === "LineString" ? { ...item, geometry: { type: "LineString", coordinates: [...item.geometry.coordinates].reverse() } } : item)));
  const smoothFeature = (id: string) => edit((current) => current.map((item) => (item.id === id ? withVertices(item, smoothed(openVertices(item), isClosed(item))) : item)));
  const removeFeature = useCallback((id: string) => {
    edit((current) => current.filter((item) => item.id !== id));
    setSelectedId((current) => (current === id ? null : current));
  }, [edit]);

  const selectFeature = (feature: ProcessFeature) => {
    setSelectedId(feature.id);
    setProcessId(feature.properties.process_id);
    if (!visiblePanels.some((panel) => panel.embeddingId === feature.properties.embedding_id)) updatePanel(active, { embeddingId: feature.properties.embedding_id });
  };

  // Shape editing: dragging a point moves it with its neighbours along the line
  // (alone with Shift); dragging a line or edge moves the whole drawing.
  const startDrag = (event: ReactPointerEvent<SVGElement>, feature: ProcessFeature, projection: Projection, vertex: number | null) => {
    if (event.button !== 0) return;
    event.stopPropagation();
    const svg = event.currentTarget.ownerSVGElement;
    if (!svg) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setSelectedId(feature.id);
    setProcessId(feature.properties.process_id);
    const original = openVertices(feature);
    const screen = original.map(([x, y]) => projection.toScreen(x, y));
    const weights = vertex === null ? original.map(() => 1) : dragWeights(screen, vertex, isClosed(feature), event.shiftKey ? 0 : editRadius);
    const [x, y] = svgPoint(event, svg);
    dragRef.current = { pointerId: event.pointerId, featureId: feature.id, original, weights, start: projection.fromScreen(x, y), fromScreen: projection.fromScreen, svg, recorded: false };
    setDragging(true);
  };
  const moveDrag = (event: ReactPointerEvent<SVGElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const [x, y] = svgPoint(event, drag.svg);
    const [ex, ey] = drag.fromScreen(x, y);
    const delta: XY = [ex - drag.start[0], ey - drag.start[1]];
    edit((current) => current.map((item) => (item.id === drag.featureId ? withVertices(item, moved(drag.original, drag.weights, delta)) : item)), !drag.recorded);
    drag.recorded = true;
  };
  const endDrag = (event: ReactPointerEvent<SVGElement>) => {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    setDragging(false);
  };
  const addVertex = (event: ReactMouseEvent<SVGElement>, feature: ProcessFeature, projection: Projection) => {
    event.stopPropagation();
    const svg = event.currentTarget.ownerSVGElement;
    if (!svg) return;
    const at = svgPoint(event, svg);
    const vertices = openVertices(feature);
    const screen = vertices.map(([x, y]) => projection.toScreen(x, y));
    const next = insertVertex(vertices, screen, at, projection.fromScreen(...at), isClosed(feature));
    edit((current) => current.map((item) => (item.id === feature.id ? withVertices(item, next) : item)));
  };
  const deleteVertex = (event: ReactMouseEvent<SVGElement>, feature: ProcessFeature, vertex: number) => {
    event.stopPropagation();
    const vertices = openVertices(feature);
    if (vertices.length <= minVertices(feature)) { setMessage("A curve needs two points and a region three."); return; }
    edit((current) => current.map((item) => (item.id === feature.id ? withVertices(item, vertices.filter((_point, at) => at !== vertex)) : item)));
  };
  // The overlay is re-rendered on every view change; handlers always see the latest state through this ref.
  const handlersRef = useRef({ startDrag, moveDrag, endDrag, addVertex, deleteVertex });
  handlersRef.current = { startDrag, moveDrag, endDrag, addVertex, deleteVertex };

  const renderOverlay = useCallback((embeddingId: string, projection: Projection) => {
    const handlers = handlersRef.current;
    const visible = features.filter((item) => item.properties.embedding_id === embeddingId);
    const rank = (item: ProcessFeature) => (item.id === selectedId ? 2 : item.properties.process_id === processId ? 1 : 0);
    const ordered = [...visible].sort((a, b) => rank(a) - rank(b));
    const editing = tool === "edit";
    const drag = { onPointerMove: handlers.moveDrag, onPointerUp: handlers.endDrag, onPointerCancel: handlers.endDrag };
    return <>
      <defs><marker id="process-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" /></marker></defs>
      {ordered.map((item) => {
        const current = item.properties.process_id === processId;
        const selected = item.id === selectedId;
        const screen = openVertices(item).map(([x, y]) => projection.toScreen(x, y));
        const path = screen.map((point) => point.join(",")).join(" ");
        const state = `${current ? "current" : "other"}${selected ? " selected" : ""}`;
        const hit = editing && <polyline
          className="edit-hit"
          points={isClosed(item) ? `${path} ${screen[0].join(",")}` : path}
          onPointerDown={(event) => handlers.startDrag(event, item, projection, null)}
          onDoubleClick={(event) => handlers.addVertex(event, item, projection)}
          {...drag}
        />;
        const handles = editing && selected && screen.map(([x, y], vertex) => <circle
          key={vertex}
          className="vertex-handle"
          cx={x}
          cy={y}
          r={4}
          onPointerDown={(event) => handlers.startDrag(event, item, projection, vertex)}
          onDoubleClick={(event) => handlers.deleteVertex(event, item, vertex)}
          {...drag}
        />);
        if (item.properties.role === "curve") {
          return <g key={item.id} className={`process-curve ${state}`}><polyline points={path} markerEnd={current ? "url(#process-arrow)" : undefined} />{current && <circle className="curve-start" cx={screen[0][0]} cy={screen[0][1]} r={4} />}{hit}{handles}</g>;
        }
        return <g key={item.id} className={`process-region ${item.properties.region_mode ?? "include"} ${state}`}><polygon points={path} />{hit}{handles}</g>;
      })}
    </>;
  }, [features, processId, selectedId, tool]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select")) return;
      const command = event.ctrlKey || event.metaKey;
      if (command && event.key.toLowerCase() === "z") { event.preventDefault(); travel(event.shiftKey ? "redo" : "undo"); return; }
      if (command && event.key.toLowerCase() === "y") { event.preventDefault(); travel("redo"); return; }
      if (command || event.altKey) return;
      if ((event.key === "Delete" || event.key === "Backspace") && selectedId) { event.preventDefault(); removeFeature(selectedId); return; }
      if (event.key === "Escape") { setSelectedId(null); setHighlight(NO_INDICES); return; }
      const shortcut = TOOLS.find((item) => item.key === event.key.toLowerCase());
      if (shortcut) setTool(shortcut.id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [removeFeature, selectedId, travel]);

  const refreshIndex = () => getProcessIndex(projectId).then(onIndex).catch(() => undefined);
  const exportValues = async () => {
    setBusy(true); setMessage("");
    try {
      const result = await exportProcessSet(projectId, setId, revisionRef.current, project.csrf_token);
      setMessage(`Exported ${Object.keys(result.report.processes).length} processes to ${result.values_path}`);
      void refreshIndex();
    } catch (caught) { setMessage(caught instanceof Error ? caught.message : "Export failed"); } finally { setBusy(false); }
  };
  const review = async (only: string[] | null) => {
    if (!annotator.trim()) { setMessage("Enter your name before reviewing."); return; }
    setBusy(true); setMessage("");
    try {
      const result = await reviewProcessSet(projectId, setId, revisionRef.current, project.csrf_token, annotator.trim(), only);
      const reloaded = await getProcessSet(projectId, setId);
      revisionRef.current = reloaded.revision; versionRef.current = 0; savedVersionRef.current = 0;
      setDocument(reloaded); setFeatures(reloaded.features); setSaveStatus("saved");
      setMessage(`Marked reviewed and exported to ${result.values_path}`);
      void refreshIndex();
    } catch (caught) { setMessage(caught instanceof Error ? caught.message : "Review failed"); } finally { setBusy(false); }
  };
  const reload = async () => {
    const loaded = await getProcessSet(projectId, setId);
    revisionRef.current = loaded.revision; versionRef.current = 0; savedVersionRef.current = 0;
    setDocument(loaded); setFeatures(loaded.features); setSaveStatus("saved"); setMessage("");
    resetHistory();
  };

  const showGenes = async () => {
    const tokens = geneList.split(/[\s,;]+/).filter(Boolean);
    if (tokens.length === 0) return;
    const found: FeatureDescriptor[] = [];
    const missing: string[] = [];
    for (const token of tokens) {
      const hits = await searchFeatures(project.features_url, token).catch(() => [] as FeatureDescriptor[]);
      const folded = token.toLowerCase();
      const hit = hits.find((item) => (item.feature_symbol ?? "").toLowerCase() === folded || item.feature_id.toLowerCase() === folded);
      if (hit) found.push(hit); else missing.push(token);
    }
    const room = nPanels - active;
    setPanels((current) => current.map((panel, position) => {
      const offset = position - active;
      return offset >= 0 && offset < Math.min(room, found.length) ? { ...panel, color: { kind: "gene", feature: found[offset] }, blend: null } : panel;
    }));
    setMessage([
      missing.length > 0 ? `Not found: ${missing.join(", ")}.` : "",
      found.length > room ? `Only ${room} plot(s) from the active one onward; choose a larger layout or an earlier plot for the rest.` : ""
    ].filter(Boolean).join(" "));
  };

  const settled = saveStatus === "saved";
  const shownEmbeddings = new Set(visiblePanels.map((panel) => panel.embeddingId));
  const hiddenCount = processFeatures.filter((item) => !shownEmbeddings.has(item.properties.embedding_id)).length;
  const hint = TOOLS.find((item) => item.id === tool)?.hint ?? "";
  const activeView = embeddingById.get(activeState.embeddingId);
  const clearPlot = useCallback(() => {
    if (tool === "edit") setSelectedId(null);
    else if (tool === "highlight") setHighlight(NO_INDICES);
  }, [tool]);
  const setView = useCallback((embeddingId: string, next: ViewState) => setViews((current) => ({ ...current, [embeddingId]: next })), []);

  return <main className="workbench process-workbench">
    <header className="workbench-header">
      <a className="compact-brand" href={`${basePath}/annotations`}>Eyck</a>
      <div className="project-title"><h1>{project.title}</h1><code>{projectId}</code><Badge>{nObservations} cells</Badge><Badge>process drawing</Badge></div>
      <div className="header-actions">
        <span className={`save-indicator ${settled ? "saved" : saveStatus === "dirty" || saveStatus === "saving" ? "working" : "problem"}`}><i />{{ saved: "Saved", dirty: "Unsaved", saving: "Saving", conflict: "Conflict", error: "Save failed" }[saveStatus]}</span>
        <a className="pt-button" href={`${basePath}/annotations/${encodeURIComponent(projectId)}`}>Annotation workbench</a>
      </div>
    </header>
    {message && <div className="issue-strip"><div className={`issue ${saveStatus === "conflict" || saveStatus === "error" ? "issue-error" : ""}`}><span>{message}</span><div className="issue-actions">{saveStatus === "conflict" && <Button onClick={() => void reload()}>Reload saved drawings</Button>}{saveStatus === "error" && <Button onClick={() => { setSaveStatus("dirty"); }}>Retry save</Button>}<Button onClick={() => setMessage("")}>Dismiss</Button></div></div></div>}
    <div className="process-grid">
      <aside className="process-panel">
        <div className="control-section">
          <Field label="Annotation set" hint="Independent sets (for example a and b) measure agreement.">
            <div className="inline-controls"><input className="pt-input" aria-label="Annotation set" value={setInput} onChange={(event) => setSetInput(event.target.value)} /><Button disabled={!SAFE_ID.test(setInput) || setInput === setId || !settled} onClick={() => setSetId(setInput)}>Open</Button></div>
          </Field>
          {index.annotation_sets.length > 0 && <div className="set-chips">{index.annotation_sets.map((item) => <button type="button" key={item.id} className={item.id === setId ? "active" : ""} disabled={!settled} onClick={() => { setSetInput(item.id); setSetId(item.id); }}>{item.id}<small>{item.feature_count}</small></button>)}</div>}
          <Field label="Annotator"><input className="pt-input" aria-label="Annotator" value={annotator} placeholder="Your name" onChange={(event) => { setAnnotator(event.target.value); writeSetting("eyck-annotator", event.target.value); }} /></Field>
        </div>
        <div className="control-section">
          <div className="section-heading"><span>Processes</span></div>
          <div className="process-list">{processIds.map((id) => {
            const drawn = features.filter((item) => item.properties.process_id === id);
            const title = index.processes.find((item) => item.id === id)?.title ?? id;
            const subset = drawn.length > 0 && !drawn.some((item) => item.properties.role === "curve");
            return <button type="button" key={id} className={id === processId ? "active" : ""} onClick={() => setProcessId(id)}><strong>{title}</strong><small>{id} / {drawn.length} drawings{subset ? " / subset" : ""}</small></button>;
          })}</div>
          <div className="inline-controls"><input className="pt-input" aria-label="New process" placeholder="new_process_id" value={newProcess} onChange={(event) => setNewProcess(event.target.value)} /><Button disabled={!SAFE_ID.test(newProcess) || processIds.includes(newProcess)} onClick={() => { setExtraProcesses((current) => [...current, newProcess]); setProcessId(newProcess); setNewProcess(""); }}>Add</Button></div>
          <p className="process-description">A process drawn with regions only marks a subset of cells (value 1) that any plot can colour.</p>
        </div>
        <div className="control-section">
          <div className="section-heading"><span>{descriptor?.title ?? processId}</span></div>
          {descriptor?.description && <p className="process-description">{descriptor.description}</p>}
          <div className="tool-buttons">{TOOLS.map((item) => <Button key={item.id} title={item.title} active={tool === item.id} aria-pressed={tool === item.id} onClick={() => setTool(item.id)}>{item.label}</Button>)}</div>
          <div className="inline-controls">
            <Button title="Undo (Ctrl+Z)" disabled={historySize.past === 0 || saveStatus === "conflict"} onClick={() => travel("undo")}>Undo</Button>
            <Button title="Redo (Ctrl+Shift+Z)" disabled={historySize.future === 0 || saveStatus === "conflict"} onClick={() => travel("redo")}>Redo</Button>
          </div>
          {tool === "edit" ? <>
            <Field label={`Bend radius ${editRadius}px`} hint="Dragging a point also moves its neighbours within this distance along the line; hold Shift to move only that point."><input type="range" min={0} max={200} step={5} value={editRadius} onChange={(event) => { setEditRadius(Number(event.target.value)); writeSetting("eyck-edit-radius", event.target.value); }} /></Field>
            <p className="process-description">Click a drawing to select it. Drag a line or edge to move the drawing, double-click it to add a point, double-click a point to remove it, and press Delete to remove the drawing.</p>
          </> : <p className="process-description">Cells inside any region (all cells when there is none) and outside every exclusion take the position of their projection onto the nearest curve.</p>}
          {hiddenCount > 0 && <Notice>{hiddenCount} drawing(s) of this process are on an embedding not shown; click one below to show it.</Notice>}
          <div className="feature-list">{processFeatures.map((item) => <div className={`feature-row ${item.id === selectedId ? "selected" : ""}`} key={item.id} onClick={() => selectFeature(item)}>
            <div><strong>{item.properties.role === "curve" ? "Curve" : item.properties.region_mode === "exclude" ? "Exclusion" : "Region"}</strong><small>{embeddingById.get(item.properties.embedding_id)?.title ?? item.properties.embedding_id} / {openVertices(item).length} pts / {item.properties.annotator}</small></div>
            <Badge tone={item.properties.review_status === "reviewed" ? "success" : "default"}>{item.properties.review_status ?? "draft"}</Badge>
            {item.properties.role === "curve" && <div className="curve-values" onClick={(event) => event.stopPropagation()}><input className="pt-input" type="number" step="0.05" aria-label="Start value" value={item.properties.value_start ?? 0} onChange={(event) => updateFeature(item.id, { value_start: Number(event.target.value) })} /><span>to</span><input className="pt-input" type="number" step="0.05" aria-label="End value" value={item.properties.value_end ?? 1} onChange={(event) => updateFeature(item.id, { value_end: Number(event.target.value) })} /><Button onClick={() => reverseCurve(item.id)}>Reverse</Button></div>}
            <div className="feature-actions" onClick={(event) => event.stopPropagation()}>
              <Button title="Average each point with its neighbours" onClick={() => smoothFeature(item.id)}>Smooth</Button>
              <Button tone="danger" onClick={() => removeFeature(item.id)}>Delete</Button>
            </div>
          </div>)}</div>
          {previewError && <p className="inline-error">{previewError}</p>}
          <small className="process-assigned">{assigned === null ? "" : regionOnly ? `${assigned} cells in this subset` : `${assigned} cells receive a value`}</small>
        </div>
        <div className="control-section">
          <Button disabled={!settled || busy || features.length === 0} onClick={() => void exportValues()}>Export values</Button>
          <Button disabled={!settled || busy || processFeatures.length === 0} onClick={() => void review([processId])}>Mark process reviewed</Button>
          <Button tone="primary" disabled={!settled || busy || features.length === 0} onClick={() => void review(null)}>Mark set reviewed</Button>
          <small className="process-assigned">Set {setId}: {document?.origin ?? "loading"}</small>
        </div>
      </aside>
      <section className="process-plot">
        <div className="layout-bar">
          <span>Plots</span>
          {LAYOUTS.map((item) => <Button key={item.id} title={item.title} active={layout === item.id} aria-pressed={layout === item.id} onClick={() => setLayout(item.id)}>{item.label}</Button>)}
          <label className="layout-toggle"><input type="checkbox" checked={linked} onChange={(event) => setLinked(event.target.checked)} />Link zoom</label>
          <span className="layout-hint">{hint}</span>
          {highlight.size > 0 && <span className="highlight-count">{highlight.size} highlighted <Button onClick={() => setHighlight(NO_INDICES)}>Clear</Button></span>}
        </div>
        <div className={`plot-layout ${layout}`}>
          {visiblePanels.map((panel, position) => <PlotPanel
            key={position}
            position={position}
            pointScale={pointScale(layout, position)}
            panel={panel}
            view={embeddingById.get(panel.embeddingId)}
            points={pointsCache[panel.embeddingId]}
            active={position === active && nPanels > 1}
            processId={processId}
            processValues={processValues}
            features={features}
            load={loadValues}
            display={display}
            onError={onError}
            highlight={highlight}
            tool={tool}
            onActivate={setActivePanel}
            onDraw={onDraw}
            onHighlight={onHighlight}
            onClear={clearPlot}
            renderOverlay={renderOverlay}
            linkedView={linked ? views[panel.embeddingId] : undefined}
            onViewChange={linked ? setView : undefined}
          />)}
        </div>
      </section>
      <aside className="process-panel">
        <div className="control-section">
          <div className="section-heading"><span>{nPanels > 1 ? `Plot ${active + 1} of ${nPanels}` : "Plot"}</span></div>
          {nPanels > 1 && <p className="process-description">Click a plot to choose which one these settings change.</p>}
          <Field label="Embedding" hint={activeView && (activeView.description || activeView.observations < nObservations) ? <>{activeView.description}{activeView.observations < nObservations && ` ${activeView.observations} of ${nObservations} cells.`}</> : undefined}>
            <Select aria-label="Embedding" value={activeState.embeddingId} onChange={(event) => updatePanel(active, { embeddingId: event.target.value })}>{index.embeddings.map((item) => <option key={item.id} value={item.id}>{item.title === item.id ? item.id : `${item.title} (${item.id})`}</option>)}</Select>
          </Field>
        </div>
        <ColorControls label="Colour by" index={index} processIds={processIds} featuresUrl={project.features_url} source={activeState.color} onSource={(source) => { if (source) updatePanel(active, { color: source }); }} />
        <ColorControls label="Blend with (red)" index={index} processIds={processIds} featuresUrl={project.features_url} source={activeState.blend} onSource={(source) => updatePanel(active, { blend: source })} />
        <div className="control-section">
          <Field label="Genes across plots" hint="Colours this plot and the following ones by one gene each.">
            <div className="inline-controls"><input className="pt-input" aria-label="Genes across plots" placeholder="Fos Egr1 Cxcl9 Mki67" value={geneList} onChange={(event) => setGeneList(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void showGenes(); }} /><Button disabled={!geneList.trim()} onClick={() => void showGenes()}>Show</Button></div>
          </Field>
        </div>
        <DisplayControls display={display} blended={activeState.blend !== null} onChange={updateDisplay} />
      </aside>
    </div>
  </main>;
}

interface PlotPanelProps {
  position: number;
  pointScale: number;
  panel: PanelState;
  view: ViewDescriptor | undefined;
  points: PointsPayload | undefined;
  active: boolean;
  processId: string;
  processValues: (number | null)[] | null;
  features: ProcessFeature[];
  load: (source: ColorSource, signal: AbortSignal) => Promise<Scalar[]>;
  display: Display;
  onError: (message: string) => void;
  highlight: Set<number>;
  tool: Tool;
  onActivate: (position: number) => void;
  onDraw: (embeddingId: string, drawn: [number, number][], toScreen: ToScreen) => void;
  onHighlight: (indices: number[], mode: SelectionMode) => void;
  onClear: () => void;
  renderOverlay: (embeddingId: string, projection: Projection) => ReactNode;
  linkedView: ViewState | undefined;
  onViewChange: ((embeddingId: string, view: ViewState) => void) | undefined;
}

function PlotPanel({ position, pointScale, panel, view, points, active, processId, processValues, features, load, display, onError, highlight, tool, onActivate, onDraw, onHighlight, onClear, renderOverlay, linkedView, onViewChange }: PlotPanelProps) {
  const followsCurrent = panel.color.kind === "process" && panel.color.process === null;
  const firstValues = useSourceValues(panel.color, followsCurrent ? processValues : undefined, load, features, onError);
  const secondValues = useSourceValues(panel.blend, undefined, load, features, onError);
  const nPoints = points?.observation_ids.length ?? 0;
  const colors = useMemo(() => displayColors(firstValues ?? undefined, panel.blend ? secondValues ?? undefined : undefined, nPoints, display), [display, firstValues, nPoints, panel.blend, secondValues]);
  const toScreenRef = useRef<ToScreen | null>(null);
  const colorTitle = [panel.color, panel.blend].filter((item): item is ColorSource => item !== null).map((item) => sourceTitle(item, processId)).join(" + ");
  return <section className={`plot-cell ${active ? "active" : ""}`}>
    {!points ? <Loading label="Loading embedding coordinates" /> : <Scatterplot
      coordinates={points.coordinates}
      colors={colors.colors}
      selectedIndices={highlight}
      focusedIndices={NO_INDICES}
      onSingleSelect={() => undefined}
      onLasso={onHighlight}
      onClearSelection={onClear}
      pointSize={Math.max(1, display.pointSize * pointScale)}
      drawOrder={colors.order}
      title={<><b>{view?.title ?? panel.embeddingId}</b> / {colorTitle}</>}
      onActivate={() => onActivate(position)}
      view={linkedView}
      onViewChange={onViewChange && ((next) => onViewChange(panel.embeddingId, next))}
      drawing={{
        mode: drawMode(tool),
        onDraw: (drawn) => { if (toScreenRef.current) onDraw(panel.embeddingId, drawn, toScreenRef.current); },
        renderOverlay: (projection) => { toScreenRef.current = projection.toScreen; return renderOverlay(panel.embeddingId, projection); },
        hint: "",
        interactive: tool === "edit"
      }}
    />}
    <div className="panel-legend">{colors.legend.slice(0, 8).map((item) => <div key={item.label}><i style={{ background: item.color }} />{item.label}</div>)}<div><i style={{ background: "rgb(190, 194, 198)" }} />not applicable / missing</div></div>
  </section>;
}

function publicPoints(projectId: string, embeddingId: string): string {
  return `${basePath}/api/v1/annotations/${encodeURIComponent(projectId)}/points?embedding=${encodeURIComponent(embeddingId)}`;
}

function sourceTitle(source: ColorSource, currentProcess: string): string {
  if (source.kind === "process") return `Process / ${source.process ?? currentProcess}`;
  if (source.kind === "column") return source.column;
  return `Gene / ${source.feature.feature_symbol ?? source.feature.feature_id}`;
}

/** Values of one colour source, reloaded when it or the drawings change. */
function useSourceValues(
  source: ColorSource | null,
  given: Scalar[] | null | undefined,
  load: (source: ColorSource, signal: AbortSignal) => Promise<Scalar[]>,
  features: ProcessFeature[],
  onError: (message: string) => void
): Scalar[] | null {
  const [values, setValues] = useState<Scalar[] | null>(null);
  const dependsOnDrawings = source?.kind === "process";
  const drawings = useDeferredValue(dependsOnDrawings ? features : null);
  // Panel settings are stored as plain objects, so a source is identified by its key.
  const key = sourceKey(source) + (source?.kind === "gene" ? `:${source.feature.feature_index}` : "");
  const sourceRef = useRef(source);
  sourceRef.current = source;
  useEffect(() => {
    const current = sourceRef.current;
    if (!current || given !== undefined) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      load(current, controller.signal).then((loaded) => { if (!controller.signal.aborted) setValues(loaded); }).catch((caught: unknown) => {
        if (!controller.signal.aborted) onError(caught instanceof Error ? caught.message : "Could not load colour values");
      });
    }, dependsOnDrawings ? 200 : 0);
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [dependsOnDrawings, drawings, given, key, load, onError]);
  if (!source) return null;
  return given !== undefined ? given : values;
}

function sourceKey(source: ColorSource | null): string {
  if (source === null) return "none";
  if (source.kind === "process") return source.process === null ? "process" : `process:${source.process}`;
  return source.kind === "column" ? `column:${source.column}` : "gene";
}

function ColorControls({ label, index, processIds, featuresUrl, source, onSource }: { label: string; index: ProcessIndex; processIds: string[]; featuresUrl: string; source: ColorSource | null; onSource: (source: ColorSource | null) => void }) {
  const columns = [...index.metadata_columns, ...index.overlay_columns];
  const optional = label !== "Colour by";
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);
  const [hits, setHits] = useState<FeatureDescriptor[]>([]);
  useEffect(() => {
    if (!deferredQuery.trim()) { setHits([]); return; }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      searchFeatures(featuresUrl, deferredQuery.trim(), controller.signal).then((result) => setHits(result.slice(0, 12))).catch(() => undefined);
    }, 180);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [deferredQuery, featuresUrl]);
  return <div className="control-section">
    <Field label={label}>
      <Select aria-label={label} value={sourceKey(source)} onChange={(event) => {
        const value = event.target.value;
        if (value === "none") onSource(null);
        else if (value === "process") onSource({ kind: "process", process: null });
        else if (value.startsWith("process:")) onSource({ kind: "process", process: value.slice(8) });
        else if (value.startsWith("column:")) onSource({ kind: "column", column: value.slice(7) });
      }}>
        {optional && <option value="none">Nothing</option>}
        <option value="process">Current process values</option>
        {source?.kind === "gene" && <option value="gene">Gene / {source.feature.feature_symbol ?? source.feature.feature_id}</option>}
        <optgroup label="Drawn processes">{processIds.map((id) => <option key={id} value={`process:${id}`}>{id}</option>)}</optgroup>
        <optgroup label="Observation columns">{columns.map((column) => <option key={column} value={`column:${column}`}>{column}</option>)}</optgroup>
      </Select>
    </Field>
    <Field label="Gene"><input className="pt-input" aria-label={`${label} gene search`} placeholder="Symbol" value={query} onChange={(event) => setQuery(event.target.value)} /></Field>
    {hits.length > 0 && <div className="feature-results">{hits.map((hit) => <button type="button" key={hit.feature_index} onClick={() => { onSource({ kind: "gene", feature: hit }); setQuery(""); setHits([]); }}><strong>{hit.feature_symbol ?? hit.feature_id}</strong>{hit.feature_symbol && <span>{hit.feature_id}</span>}</button>)}</div>}
  </div>;
}

function DisplayControls({ display, blended, onChange }: { display: Display; blended: boolean; onChange: (patch: Partial<Display>) => void }) {
  return <div className="control-section">
    <Field label="Palette" hint={blended ? "Blending: first channel blue, second red, both purple." : undefined}>
      <Select aria-label="Palette" disabled={blended} value={display.palette} onChange={(event) => onChange({ palette: event.target.value as Palette })}>{PALETTES.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</Select>
    </Field>
    <Field label={`Point size ${display.pointSize}px`}><input type="range" min={1} max={16} step={0.5} value={display.pointSize} onChange={(event) => onChange({ pointSize: Number(event.target.value) })} /></Field>
    <Field label={`Opacity ${Math.round(display.opacity * 100)}%`}><input type="range" min={0.1} max={1} step={0.05} value={display.opacity} onChange={(event) => onChange({ opacity: Number(event.target.value) })} /></Field>
    <Field label={`Colour cap at percentile ${display.upperPercentile}`} hint="Lower caps saturate more cells."><input type="range" min={80} max={100} step={0.5} value={display.upperPercentile} onChange={(event) => onChange({ upperPercentile: Number(event.target.value) })} /></Field>
  </div>;
}
