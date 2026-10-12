import { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
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
import { colorValues } from "./color";
import { Scatterplot, type DrawMode, type ToScreen } from "./Scatterplot";
import type {
  FeatureDescriptor,
  PointsPayload,
  ProcessDocument,
  ProcessFeature,
  ProcessIndex,
  ProjectDetail,
  Scalar
} from "./types";
import { Badge, Button, Field, Loading, Notice, Select } from "./ui";

type Tool = "pan" | "curve" | "include" | "exclude";
type ColorSource =
  | { kind: "process" }
  | { kind: "column"; column: string }
  | { kind: "gene"; feature: FeatureDescriptor };
type SaveStatus = "saved" | "dirty" | "saving" | "conflict" | "error";

const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const NO_INDICES = new Set<number>();

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

function vertices(feature: ProcessFeature): [number, number][] {
  return feature.geometry.type === "LineString" ? feature.geometry.coordinates : feature.geometry.coordinates[0];
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
  const [embeddingId, setEmbeddingId] = useState(index.embeddings[0]);
  const [points, setPoints] = useState<PointsPayload | null>(null);
  const [extraProcesses, setExtraProcesses] = useState<string[]>([]);
  const [newProcess, setNewProcess] = useState("");
  const [processId, setProcessId] = useState(index.processes[0]?.id ?? "");
  const [tool, setTool] = useState<Tool>("pan");
  const [colorSource, setColorSource] = useState<ColorSource>({ kind: "process" });
  const [colorValuesState, setColorValuesState] = useState<Scalar[] | null>(null);
  const [processValues, setProcessValues] = useState<(number | null)[] | null>(null);
  const [assigned, setAssigned] = useState<number | null>(null);
  const [previewError, setPreviewError] = useState("");
  const [busy, setBusy] = useState(false);
  const revisionRef = useRef("");
  const versionRef = useRef(0);
  const savedVersionRef = useRef(0);
  const savingRef = useRef(false);
  const featuresRef = useRef(features);
  featuresRef.current = features;
  const toScreenRef = useRef<ToScreen | null>(null);

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
    }).catch((caught: unknown) => { if (!controller.signal.aborted) setMessage(caught instanceof Error ? caught.message : "Could not load drawings"); });
    writeSetting(`eyck-process-set:${projectId}`, setId);
    const url = new URL(window.location.href);
    url.searchParams.set("set", setId);
    window.history.replaceState(null, "", url.toString());
    return () => controller.abort();
  }, [projectId, setId]);

  useEffect(() => {
    const controller = new AbortController();
    setPoints(null);
    const url = new URL(publicPoints(projectId, embeddingId), window.location.href);
    getPoints(url.toString(), controller.signal).then(setPoints).catch((caught: unknown) => {
      if (!controller.signal.aborted) setMessage(caught instanceof Error ? caught.message : "Could not load embedding");
    });
    return () => controller.abort();
  }, [projectId, embeddingId]);

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

  useEffect(() => {
    if (saveStatus !== "dirty") return;
    const timer = window.setTimeout(() => void save(), 500);
    return () => window.clearTimeout(timer);
  }, [features, save, saveStatus]);

  const edit = useCallback((update: (current: ProcessFeature[]) => ProcessFeature[]) => {
    if (saveStatus === "conflict") return;
    versionRef.current += 1;
    setFeatures(update);
    setSaveStatus("dirty");
  }, [saveStatus]);

  const processFeatures = useMemo(() => features.filter((item) => item.properties.process_id === processId), [features, processId]);
  const deferredProcessFeatures = useDeferredValue(processFeatures);

  useEffect(() => {
    if (!processId) return;
    if (!deferredProcessFeatures.some((item) => item.properties.role === "curve")) {
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

  useEffect(() => {
    if (colorSource.kind === "process") return;
    const controller = new AbortController();
    setColorValuesState(null);
    const load = colorSource.kind === "column"
      ? getObservationColumn(projectId, colorSource.column, controller.signal).then((result) => result.values)
      : getFeatureValues(project.features_url, colorSource.feature.feature_index, controller.signal).then((result) => result.values);
    load.then(setColorValuesState).catch((caught: unknown) => {
      if (!controller.signal.aborted) setMessage(caught instanceof Error ? caught.message : "Could not load colour values");
    });
    return () => controller.abort();
  }, [colorSource, project.features_url, projectId]);

  const shownValues: Scalar[] | null = colorSource.kind === "process" ? processValues : colorValuesState;
  const colors = useMemo(() => colorValues(shownValues ?? undefined, points?.observation_ids.length ?? 0, { contrast: true }), [points, shownValues]);
  const colorTitle = colorSource.kind === "process" ? `Process / ${processId}` : colorSource.kind === "column" ? colorSource.column : `Gene / ${colorSource.feature.feature_symbol ?? colorSource.feature.feature_id}`;

  const processIds = useMemo(() => {
    const declared = index.processes.map((item) => item.id);
    const drawn = features.map((item) => item.properties.process_id);
    return [...new Set([...declared, ...extraProcesses, ...drawn])];
  }, [extraProcesses, features, index.processes]);
  const descriptor = index.processes.find((item) => item.id === processId);

  const onDraw = useCallback((drawn: [number, number][]) => {
    const toScreen = toScreenRef.current;
    if (!toScreen || tool === "pan") return;
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
    edit((current) => [...current, feature]);
  }, [annotator, edit, embeddingId, processFeatures, processId, setId, tool]);

  const renderOverlay = useCallback((toScreen: ToScreen) => {
    toScreenRef.current = toScreen;
    const visible = features.filter((item) => item.properties.embedding_id === embeddingId);
    const ordered = [...visible.filter((item) => item.properties.process_id !== processId), ...visible.filter((item) => item.properties.process_id === processId)];
    return <>
      <defs><marker id="process-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" /></marker></defs>
      {ordered.map((item) => {
        const current = item.properties.process_id === processId;
        const path = vertices(item).map(([x, y]) => toScreen(x, y).join(",")).join(" ");
        if (item.properties.role === "curve") {
          return <g key={item.id} className={`process-curve ${current ? "current" : "other"}`}><polyline points={path} markerEnd={current ? "url(#process-arrow)" : undefined} />{current && <circle cx={toScreen(...vertices(item)[0])[0]} cy={toScreen(...vertices(item)[0])[1]} r={4} />}</g>;
        }
        return <polygon key={item.id} className={`process-region ${item.properties.region_mode ?? "include"} ${current ? "current" : "other"}`} points={path} />;
      })}
    </>;
  }, [embeddingId, features, processId]);

  const updateFeature = (id: string, patch: Partial<ProcessFeature["properties"]>) => edit((current) => current.map((item) => item.id === id ? { ...item, properties: { ...item.properties, ...patch } } : item));
  const reverseCurve = (id: string) => edit((current) => current.map((item) => item.id === id && item.geometry.type === "LineString" ? { ...item, geometry: { type: "LineString", coordinates: [...item.geometry.coordinates].reverse() } } : item));
  const removeFeature = (id: string) => edit((current) => current.filter((item) => item.id !== id));

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
  };

  const settled = saveStatus === "saved";
  const hiddenCount = processFeatures.filter((item) => item.properties.embedding_id !== embeddingId).length;
  const hint = tool === "pan" ? "Drag to pan, scroll to zoom" : tool === "curve" ? "Draw from the 0 end towards the 1 end" : tool === "include" ? "Draw around cells that receive this process" : "Draw around cells to mark not applicable";

  return <main className="workbench process-workbench">
    <header className="workbench-header">
      <a className="compact-brand" href={`${basePath}/annotations`}>Eyck</a>
      <div className="project-title"><h1>{project.title}</h1><code>{projectId}</code><Badge>{points?.observation_ids.length ?? project.n_observations} cells</Badge><Badge>process drawing</Badge></div>
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
            const count = features.filter((item) => item.properties.process_id === id).length;
            const title = index.processes.find((item) => item.id === id)?.title ?? id;
            return <button type="button" key={id} className={id === processId ? "active" : ""} onClick={() => setProcessId(id)}><strong>{title}</strong><small>{id} / {count} drawings</small></button>;
          })}</div>
          <div className="inline-controls"><input className="pt-input" aria-label="New process" placeholder="new_process_id" value={newProcess} onChange={(event) => setNewProcess(event.target.value)} /><Button disabled={!SAFE_ID.test(newProcess) || processIds.includes(newProcess)} onClick={() => { setExtraProcesses((current) => [...current, newProcess]); setProcessId(newProcess); setNewProcess(""); }}>Add</Button></div>
        </div>
        <div className="control-section">
          <div className="section-heading"><span>{descriptor?.title ?? processId}</span></div>
          {descriptor?.description && <p className="process-description">{descriptor.description}</p>}
          <div className="tool-buttons">{([["pan", "Pan"], ["curve", "Curve"], ["include", "Region"], ["exclude", "Exclude"]] as const).map(([id, label]) => <Button key={id} active={tool === id} aria-pressed={tool === id} onClick={() => setTool(id)}>{label}</Button>)}</div>
          <p className="process-description">Cells inside any region (all cells when there is none) and outside every exclusion take the position of their projection onto the nearest curve.</p>
          {hiddenCount > 0 && <Notice>{hiddenCount} drawing(s) of this process are on another embedding.</Notice>}
          <div className="feature-list">{processFeatures.map((item) => <div className="feature-row" key={item.id}>
            <div><strong>{item.properties.role === "curve" ? "Curve" : item.properties.region_mode === "exclude" ? "Exclusion" : "Region"}</strong><small>{item.properties.embedding_id} / {vertices(item).length} pts / {item.properties.annotator}</small></div>
            <Badge tone={item.properties.review_status === "reviewed" ? "success" : "default"}>{item.properties.review_status ?? "draft"}</Badge>
            {item.properties.role === "curve" && <div className="curve-values"><input className="pt-input" type="number" step="0.05" aria-label="Start value" value={item.properties.value_start ?? 0} onChange={(event) => updateFeature(item.id, { value_start: Number(event.target.value) })} /><span>to</span><input className="pt-input" type="number" step="0.05" aria-label="End value" value={item.properties.value_end ?? 1} onChange={(event) => updateFeature(item.id, { value_end: Number(event.target.value) })} /><Button onClick={() => reverseCurve(item.id)}>Reverse</Button></div>}
            <Button tone="danger" onClick={() => removeFeature(item.id)}>Delete</Button>
          </div>)}</div>
          {previewError && <p className="inline-error">{previewError}</p>}
          <small className="process-assigned">{assigned === null ? "" : `${assigned} cells receive a value`}</small>
        </div>
        <div className="control-section">
          <Button disabled={!settled || busy || features.length === 0} onClick={() => void exportValues()}>Export values</Button>
          <Button disabled={!settled || busy || processFeatures.length === 0} onClick={() => void review([processId])}>Mark process reviewed</Button>
          <Button tone="primary" disabled={!settled || busy || features.length === 0} onClick={() => void review(null)}>Mark set reviewed</Button>
          <small className="process-assigned">Set {setId}: {document?.origin ?? "loading"}</small>
        </div>
      </aside>
      <section className="process-plot">
        {!points ? <Loading label="Loading embedding coordinates" /> : <Scatterplot
          coordinates={points.coordinates}
          colors={colors.colors}
          selectedIndices={NO_INDICES}
          focusedIndices={NO_INDICES}
          onSingleSelect={() => undefined}
          onLasso={() => undefined}
          onClearSelection={() => undefined}
          drawing={{ mode: tool === "pan" ? null : (tool === "curve" ? "curve" : "region") as DrawMode, onDraw, renderOverlay, hint }}
        />}
      </section>
      <aside className="process-panel">
        <div className="control-section">
          <Field label="Embedding"><Select value={embeddingId} onChange={(event) => setEmbeddingId(event.target.value)}>{index.embeddings.map((id) => <option key={id} value={id}>{id}</option>)}</Select></Field>
        </div>
        <ColorControls index={index} featuresUrl={project.features_url} source={colorSource} onSource={setColorSource} />
        <div className="compact-legend"><span>{colorTitle}</span>{colors.legend.slice(0, 10).map((item) => <div key={item.label}><i style={{ background: item.color }} />{item.label}</div>)}<div><i style={{ background: "rgb(190, 194, 198)" }} />not applicable / missing</div></div>
      </aside>
    </div>
  </main>;
}

function publicPoints(projectId: string, embeddingId: string): string {
  return `${basePath}/api/v1/annotations/${encodeURIComponent(projectId)}/points?embedding=${encodeURIComponent(embeddingId)}`;
}

function ColorControls({ index, featuresUrl, source, onSource }: { index: ProcessIndex; featuresUrl: string; source: ColorSource; onSource: (source: ColorSource) => void }) {
  const columns = [...index.metadata_columns, ...index.overlay_columns];
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
    <Field label="Colour by">
      <Select aria-label="Colour by" value={source.kind === "process" ? "process" : source.kind === "column" ? `column:${source.column}` : "gene"} onChange={(event) => {
        const value = event.target.value;
        if (value === "process") onSource({ kind: "process" });
        else if (value.startsWith("column:")) onSource({ kind: "column", column: value.slice(7) });
      }}>
        <option value="process">Current process values</option>
        {source.kind === "gene" && <option value="gene">Gene / {source.feature.feature_symbol ?? source.feature.feature_id}</option>}
        <optgroup label="Observation columns">{columns.map((column) => <option key={column} value={`column:${column}`}>{column}</option>)}</optgroup>
      </Select>
    </Field>
    <Field label="Gene"><input className="pt-input" aria-label="Gene search" placeholder="Symbol" value={query} onChange={(event) => setQuery(event.target.value)} /></Field>
    {hits.length > 0 && <div className="feature-results">{hits.map((hit) => <button type="button" key={hit.feature_index} onClick={() => { onSource({ kind: "gene", feature: hit }); setQuery(""); setHits([]); }}><strong>{hit.feature_symbol ?? hit.feature_id}</strong>{hit.feature_symbol && <span>{hit.feature_id}</span>}</button>)}</div>}
  </div>;
}
