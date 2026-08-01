import { useDeferredValue, useEffect, useState } from "react";
import { searchFeatures } from "./api";
import type {
  Comparator,
  FeatureDescriptor,
  MarkerProgramCreate,
  MarkerProgramResult,
  ProjectDetail,
  SelectionCreate,
  WorkspaceDocument
} from "./types";
import { Badge, Button, Field, Notice, Select } from "./ui";

interface MarkerPanelProps {
  project: ProjectDetail;
  workspace: WorkspaceDocument;
  zoomId: string;
  openProgramId: string | null;
  onCreateProgram: (request: MarkerProgramCreate) => Promise<MarkerProgramResult>;
  onLoadProgram: (programId: string) => Promise<MarkerProgramResult>;
  onCreateSelection: (request: SelectionCreate) => Promise<void>;
  onOverlay: (result: MarkerProgramResult, mode: "per_cell" | "cluster_mean") => void;
}

function compare(value: number, comparator: Comparator, cutoff: number): boolean {
  if (comparator === ">") return value > cutoff;
  if (comparator === ">=") return value >= cutoff;
  if (comparator === "<") return value < cutoff;
  return value <= cutoff;
}

export function MarkerPanel({
  project,
  workspace,
  zoomId,
  openProgramId,
  onCreateProgram,
  onLoadProgram,
  onCreateSelection,
  onOverlay
}: MarkerPanelProps) {
  const clusterings = workspace.clusterings.filter((item) => item.zoom_id === zoomId);
  const [id, setId] = useState("");
  const [name, setName] = useState("");
  const [clusteringId, setClusteringId] = useState(clusterings[0]?.id ?? "");
  const [markers, setMarkers] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);
  const [suggestions, setSuggestions] = useState<FeatureDescriptor[]>([]);
  const [ctrlSize, setCtrlSize] = useState(50);
  const [bins, setBins] = useState(25);
  const [seed, setSeed] = useState(0);
  const [result, setResult] = useState<MarkerProgramResult | null>(null);
  const [overlayMode, setOverlayMode] = useState<"per_cell" | "cluster_mean">("cluster_mean");
  const [comparator, setComparator] = useState<Comparator>(">=");
  const [cutoff, setCutoff] = useState(0);
  const [selectionId, setSelectionId] = useState("");
  const [selectionName, setSelectionName] = useState("");
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!deferredQuery.trim()) {
      setSuggestions([]);
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      searchFeatures(project.features_url, deferredQuery.trim(), controller.signal)
        .then((items) => setSuggestions(items.slice(0, 8)))
        .catch((caught: unknown) => {
          if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : "Feature search failed");
        });
    }, 160);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [deferredQuery, project.features_url]);

  useEffect(() => {
    if (!openProgramId) return;
    setWorking(true);
    setError("");
    onLoadProgram(openProgramId).then((loaded) => {
      setResult(loaded);
      setId(loaded.program.id);
      setName(loaded.program.name);
      setMarkers(loaded.program.requested_markers);
      setClusteringId(loaded.program.clustering_id);
      setCtrlSize(Number(loaded.program.parameters.ctrl_size ?? 50));
      setBins(Number(loaded.program.parameters.n_bins ?? 25));
      setSeed(Number(loaded.program.parameters.random_state ?? 0));
      onOverlay(loaded, "cluster_mean");
      setMessage(`Opened saved marker program ${loaded.program.name}.`);
    }).catch((caught: unknown) => setError(caught instanceof Error ? caught.message : "Could not load marker result"))
      .finally(() => setWorking(false));
  }, [onLoadProgram, onOverlay, openProgramId]);

  const addMarker = (value: string) => {
    const token = value.trim();
    if (!token) return;
    setMarkers((current) => current.includes(token) ? current : [...current, token]);
    setQuery("");
    setSuggestions([]);
  };

  const run = async () => {
    setWorking(true);
    setError("");
    setMessage("");
    try {
      const loaded = await onCreateProgram({
        id,
        name,
        zoom_id: zoomId,
        clustering_id: clusteringId,
        markers,
        ctrl_size: ctrlSize,
        n_bins: bins,
        random_state: seed
      });
      setResult(loaded);
      setOverlayMode("cluster_mean");
      onOverlay(loaded, "cluster_mean");
      setMessage(`Saved ${loaded.program.name}; ${loaded.program.marker_ids.length} marker IDs resolved exactly.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Marker program failed");
    } finally {
      setWorking(false);
    }
  };

  const setMode = (mode: "per_cell" | "cluster_mean") => {
    setOverlayMode(mode);
    if (result) onOverlay(result, mode);
  };

  return (
    <section className="tool-card marker-tools">
      <header className="section-title-row"><div><span>Cluster-aware scoring</span><h2>Marker programs</h2></div><Badge>{workspace.marker_programs.filter((item) => item.zoom_id === zoomId).length} saved</Badge></header>
      {error && <Notice tone="danger" role="alert"><strong>Marker workflow blocked.</strong> {error}</Notice>}
      {working && <Notice role="status">Working. Marker scoring is synchronous and may take time.</Notice>}
      {message && <Notice>{message}</Notice>}
      <div className="form-grid marker-form">
        <Field label="Program ID"><input className="pt-input" value={id} disabled={Boolean(result && result.program.id === id)} onChange={(event) => setId(event.target.value)} /></Field>
        <Field label="Program name"><input className="pt-input" value={name} onChange={(event) => setName(event.target.value)} /></Field>
        <Field label="Saved clustering"><Select value={clusteringId} onChange={(event) => setClusteringId(event.target.value)}><option value="">Choose clustering</option>{clusterings.map((item) => <option value={item.id} key={item.id}>{item.name} ({item.cluster_count})</option>)}</Select></Field>
        <Field label="Feature search / validated token" hint="Choose a hit or press Enter. Missing and ambiguous names are rejected by the backend.">
          <input className="pt-input" value={query} placeholder="Gene symbol or exact feature ID" onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === ",") {
              event.preventDefault();
              addMarker(query);
            }
          }} />
        </Field>
        {suggestions.length > 0 && <div className="feature-results wide-field">{suggestions.map((feature) => <button type="button" key={feature.feature_index} onClick={() => addMarker(feature.feature_symbol || feature.feature_id)}><strong>{feature.feature_symbol || feature.feature_id}</strong><span>{feature.feature_id}</span></button>)}</div>}
        <div className="marker-tokens wide-field">{markers.map((marker) => <button type="button" key={marker} title={`Remove ${marker}`} onClick={() => setMarkers((current) => current.filter((item) => item !== marker))}>{marker} x</button>)}{markers.length === 0 && <span>No marker tokens.</span>}</div>
        <Field label="ctrl_size"><input className="pt-input" type="number" min={1} value={ctrlSize} onChange={(event) => setCtrlSize(Number(event.target.value))} /></Field>
        <Field label="n_bins"><input className="pt-input" type="number" min={2} value={bins} onChange={(event) => setBins(Number(event.target.value))} /></Field>
        <Field label="Random seed"><input className="pt-input" type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></Field>
        <div className="form-actions"><Button tone="primary" disabled={!id || !name || !clusteringId || markers.length === 0 || ctrlSize < 1 || bins < 2 || working || workspace.marker_programs.some((item) => item.id === id) || Boolean(result && result.program.id === id)} onClick={() => void run()}>Run and save marker program</Button></div>
      </div>

      {result && (
        <div className="marker-results">
          <header><div><strong>{result.program.name}</strong><code>{result.program.id}</code></div><div className="segmented"><Button active={overlayMode === "per_cell"} onClick={() => setMode("per_cell")}>Per-cell score</Button><Button active={overlayMode === "cluster_mean"} onClick={() => setMode("cluster_mean")}>Cluster mean</Button></div></header>
          <div className="resolved-markers"><span>Resolved marker IDs</span><code>{result.program.marker_ids.join(", ")}</code><span>Implementation</span><code>{result.program.implementation}</code></div>
          <div className="cutoff-controls">
            <Select aria-label="Cutoff comparator" value={comparator} onChange={(event) => setComparator(event.target.value as Comparator)}><option>&gt;</option><option>&gt;=</option><option>&lt;</option><option>&lt;=</option></Select>
            <input className="pt-input" aria-label="Cutoff" type="number" step="any" value={cutoff} onChange={(event) => setCutoff(Number(event.target.value))} />
            <input className="pt-input" aria-label="Cutoff selection ID" placeholder="selection-id" value={selectionId} onChange={(event) => setSelectionId(event.target.value)} />
            <input className="pt-input" aria-label="Cutoff selection name" placeholder="Selection name" value={selectionName} onChange={(event) => setSelectionName(event.target.value)} />
            <Button tone="primary" disabled={!selectionId || !selectionName || working} onClick={() => void (async () => {
              setWorking(true);
              setError("");
              try {
                await onCreateSelection({ id: selectionId, name: selectionName, zoom_id: zoomId, definition: { kind: "marker_cutoff", marker_program_id: result.program.id, comparator, cutoff } });
                setMessage(`Saved marker cutoff selection ${selectionName}. It is available to the Boolean builder.`);
              } catch (caught) {
                setError(caught instanceof Error ? caught.message : "Could not save marker cutoff selection");
              } finally {
                setWorking(false);
              }
            })()}>Save cutoff selection</Button>
          </div>
          <div className="cluster-table-wrap"><table className="cluster-table"><thead><tr><th>Cluster ID</th><th>Cells</th><th>Mean score</th><th>{comparator} {cutoff}</th></tr></thead><tbody>{result.cluster_table.map((row) => <tr key={row.cluster_id}><td><code>{row.cluster_id}</code></td><td>{row.cell_count}</td><td>{row.mean_score.toPrecision(5)}</td><td><Badge tone={compare(row.mean_score, comparator, cutoff) ? "success" : "default"}>{compare(row.mean_score, comparator, cutoff) ? "included" : "excluded"}</Badge></td></tr>)}</tbody></table></div>
        </div>
      )}
    </section>
  );
}
