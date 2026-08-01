import { useEffect, useMemo, useState } from "react";
import { getClusteringResult } from "./api";
import type {
  BooleanOperator,
  PointsPayload,
  SelectionCreate,
  WorkspaceDocument,
  ZoomCreate
} from "./types";
import { Badge, Button, Field, Notice, Select } from "./ui";
import { formatCount, resolveSelections } from "./workspace";

interface SelectionPanelProps {
  workspace: WorkspaceDocument;
  workspaceUrl: string;
  zoomId: string;
  embeddingId: string;
  points: PointsPayload;
  draftIds: string[];
  draftKind: "manual" | "lasso";
  draftPolygon: [number, number][];
  onCreateSelection: (request: SelectionCreate) => Promise<void>;
  onCreateZoom: (request: ZoomCreate) => Promise<void>;
  onDraftSaved: (selectionId: string) => void;
  onClearDraft: () => void;
}

function selectedValues(event: React.ChangeEvent<HTMLSelectElement>): string[] {
  return [...event.currentTarget.selectedOptions].map((option) => option.value);
}

function Status({ state, error }: { state: "idle" | "working" | "done"; error: string }) {
  if (error) return <Notice tone="danger" role="alert">{error}</Notice>;
  if (state === "working") return <Notice>Saving to workspace...</Notice>;
  if (state === "done") return <Notice>Saved. Workspace revision refreshed.</Notice>;
  return null;
}

export function SelectionPanel({
  workspace,
  workspaceUrl,
  zoomId,
  embeddingId,
  points,
  draftIds,
  draftKind,
  draftPolygon,
  onCreateSelection,
  onCreateZoom,
  onDraftSaved,
  onClearDraft
}: SelectionPanelProps) {
  const zoom = workspace.zooms.find((item) => item.id === zoomId)!;
  const selections = workspace.selections.filter((item) => item.zoom_id === zoomId);
  const clusterings = workspace.clusterings.filter((item) => item.zoom_id === zoomId);
  const [draftId, setDraftId] = useState("");
  const [draftName, setDraftName] = useState("");
  const [clusterId, setClusterId] = useState("");
  const [clusterName, setClusterName] = useState("");
  const [clusteringId, setClusteringId] = useState(clusterings[0]?.id ?? "");
  const [clusterIds, setClusterIds] = useState<string[]>([]);
  const [clusterOptions, setClusterOptions] = useState<Array<{ id: string; count: number }>>([]);
  const [clustersLoading, setClustersLoading] = useState(false);
  const [clustersError, setClustersError] = useState("");
  const [booleanId, setBooleanId] = useState("");
  const [booleanName, setBooleanName] = useState("");
  const [booleanOperator, setBooleanOperator] = useState<BooleanOperator>("union");
  const [booleanInputs, setBooleanInputs] = useState<string[]>([]);
  const [booleanBase, setBooleanBase] = useState("");
  const [booleanExclusions, setBooleanExclusions] = useState<string[]>([]);
  const [zoomObjectId, setZoomObjectId] = useState("");
  const [zoomName, setZoomName] = useState("");
  const [zoomOperator, setZoomOperator] = useState<BooleanOperator>("union");
  const [zoomInputs, setZoomInputs] = useState<string[]>([]);
  const [zoomBase, setZoomBase] = useState("");
  const [zoomExclusions, setZoomExclusions] = useState<string[]>([]);
  const [state, setState] = useState<"idle" | "working" | "done">("idle");
  const [error, setError] = useState("");

  useEffect(() => {
    setClusterIds([]);
    setClusterOptions([]);
    setClustersError("");
    if (!clusteringId) return;
    const controller = new AbortController();
    setClustersLoading(true);
    getClusteringResult(workspaceUrl, clusteringId, controller.signal).then((result) => {
      if (result.clustering.id !== clusteringId || result.clustering.zoom_id !== zoomId) {
        throw new Error("Clustering result does not belong to the selected clustering and zoom");
      }
      if (result.observation_ids.length !== result.cluster_ids.length) {
        throw new Error("Clustering IDs do not match their observation index");
      }
      if (new Set(result.observation_ids).size !== result.observation_ids.length) {
        throw new Error("Clustering result contains duplicate observation IDs");
      }
      const clusterByObservation = new Map(result.observation_ids.map((id, index) => [id, result.cluster_ids[index]]));
      const pointIds = new Set(points.observation_ids);
      const missing = points.observation_ids.filter((id) => !clusterByObservation.has(id));
      const extras = result.observation_ids.filter((id) => !pointIds.has(id));
      if (missing.length > 0 || extras.length > 0) {
        throw new Error(`Clustering observation alignment failed (${missing.length} missing, ${extras.length} outside this zoom)`);
      }
      const counts = new Map<string, number>();
      points.observation_ids.forEach((id) => {
        const cluster = clusterByObservation.get(id)!;
        counts.set(cluster, (counts.get(cluster) ?? 0) + 1);
      });
      setClusterOptions([...counts].sort(([a], [b]) => a.localeCompare(b)).map(([id, count]) => ({ id, count })));
    }).catch((caught: unknown) => {
      if (!controller.signal.aborted) setClustersError(caught instanceof Error ? caught.message : "Could not load saved clustering IDs");
    }).finally(() => {
      if (!controller.signal.aborted) setClustersLoading(false);
    });
    return () => controller.abort();
  }, [clusteringId, points.observation_ids, workspaceUrl, zoomId]);

  const booleanRecipeInputs = booleanOperator === "exclusion"
    ? [booleanBase, ...booleanExclusions.filter((id) => id !== booleanBase)].filter(Boolean)
    : booleanInputs;
  const zoomRecipeInputs = zoomOperator === "exclusion"
    ? [zoomBase, ...zoomExclusions.filter((id) => id !== zoomBase)].filter(Boolean)
    : zoomInputs;
  const previewIds = useMemo(() => resolveSelections(
    selections,
    zoomOperator,
    zoomRecipeInputs,
    zoom.observation_ids
  ), [selections, zoom.observation_ids, zoomOperator, zoomRecipeInputs]);
  const strict = previewIds.length > 0 && previewIds.length < zoom.observation_count;

  const run = async (action: () => Promise<void>) => {
    setState("working");
    setError("");
    try {
      await action();
      setState("done");
    } catch (caught) {
      setState("idle");
      setError(caught instanceof Error ? caught.message : "Workspace mutation failed");
    }
  };

  return (
    <section className="tool-card selection-tools">
      <header className="section-title-row">
        <div><span>Reusable evidence</span><h2>Selection and child zoom</h2></div>
        <Badge tone={draftIds.length > 0 ? "warning" : "default"}>{formatCount(draftIds.length)} draft</Badge>
      </header>
      <Status state={state} error={error} />
      <details open className="tool-section draft-section">
        <summary>Draft selection <span>unsaved</span></summary>
        <div className="form-grid">
          <Notice className="wide-field">
            {draftIds.length > 0
              ? `${formatCount(draftIds.length)} transient cells from ${draftKind}${draftKind === "lasso" ? `, ${draftPolygon.length} polygon vertices` : ""}.`
              : "Click a point or use the lasso on the plot. This transient outline is not yet reusable evidence."}
          </Notice>
          <Field label="Stable selection ID"><input className="pt-input" value={draftId} onChange={(event) => setDraftId(event.target.value)} /></Field>
          <Field label="Display name"><input className="pt-input" value={draftName} onChange={(event) => setDraftName(event.target.value)} /></Field>
          <div className="form-actions wide-field">
            <Button disabled={draftIds.length === 0} onClick={onClearDraft}>Discard draft</Button>
            <Button
              tone="primary"
              disabled={!draftId || !draftName || draftIds.length === 0 || state === "working"}
              onClick={() => void run(async () => {
                await onCreateSelection({
                  id: draftId,
                  name: draftName,
                  zoom_id: zoomId,
                  definition: {
                    kind: draftKind,
                    observation_ids: draftIds,
                    embedding_id: draftKind === "lasso" ? embeddingId : null,
                    polygon: draftKind === "lasso" ? draftPolygon : [],
                    provenance: {
                      method: draftKind === "lasso" ? "eyck.browser-lasso.v1" : "eyck.browser-manual.v1",
                      parameters: { interaction: draftKind, observation_count: draftIds.length },
                      input_ids: draftKind === "lasso" ? [embeddingId] : []
                    }
                  }
                });
                onDraftSaved(draftId);
                setDraftId("");
                setDraftName("");
              })}
            >Save named selection</Button>
          </div>
        </div>
      </details>

      <details className="tool-section">
        <summary>Saved cluster selection</summary>
        <div className="form-grid">
          <Field label="Selection ID"><input className="pt-input" value={clusterId} onChange={(event) => setClusterId(event.target.value)} /></Field>
          <Field label="Name"><input className="pt-input" value={clusterName} onChange={(event) => setClusterName(event.target.value)} /></Field>
          <Field label="Saved clustering">
            <Select value={clusteringId} onChange={(event) => setClusteringId(event.target.value)}>
              <option value="">Choose clustering</option>
              {clusterings.map((item) => <option value={item.id} key={item.id}>{item.name} ({item.cluster_count})</option>)}
            </Select>
          </Field>
          <Field label="Stable cluster IDs" hint="Loaded from the saved clustering result and aligned to the current zoom observations.">
            <Select multiple size={Math.min(7, Math.max(2, clusterOptions.length))} value={clusterIds} disabled={!clusteringId || clustersLoading || Boolean(clustersError)} onChange={(event) => setClusterIds(selectedValues(event))}>
              {clusterOptions.map((item) => <option value={item.id} key={item.id}>{item.id} ({formatCount(item.count)} cells)</option>)}
            </Select>
          </Field>
          {clustersLoading && <Notice className="wide-field" role="status">Loading saved cluster identities...</Notice>}
          {clustersError && <Notice className="wide-field" tone="danger" role="alert">{clustersError}</Notice>}
          <div className="form-actions wide-field">
            <Button tone="primary" disabled={!clusterId || !clusterName || !clusteringId || clusterIds.length === 0 || clustersLoading || Boolean(clustersError) || state === "working"} onClick={() => void run(() => onCreateSelection({
              id: clusterId,
              name: clusterName,
              zoom_id: zoomId,
              definition: { kind: "clusters", clustering_id: clusteringId, cluster_ids: clusterIds }
            }))}>Save cluster selection</Button>
          </div>
        </div>
      </details>

      <details className="tool-section">
        <summary>Boolean selection builder</summary>
        <div className="form-grid">
          <Field label="Selection ID"><input className="pt-input" value={booleanId} onChange={(event) => setBooleanId(event.target.value)} /></Field>
          <Field label="Name"><input className="pt-input" value={booleanName} onChange={(event) => setBooleanName(event.target.value)} /></Field>
          <Field label="Operator"><Select value={booleanOperator} onChange={(event) => setBooleanOperator(event.target.value as BooleanOperator)}><option>union</option><option>intersection</option><option>exclusion</option></Select></Field>
          {booleanOperator === "exclusion" ? <>
            <Field label="Base selection" hint="The result starts with this named selection."><Select value={booleanBase} onChange={(event) => setBooleanBase(event.target.value)}><option value="">Choose base</option>{selections.map((item) => <option value={item.id} key={item.id}>{item.name} ({item.observation_count})</option>)}</Select></Field>
            <Field label="Selections to exclude" hint="One or more named selections are subtracted from the base."><Select multiple size={Math.min(7, Math.max(2, selections.length))} value={booleanExclusions} onChange={(event) => setBooleanExclusions(selectedValues(event))}>{selections.filter((item) => item.id !== booleanBase).map((item) => <option value={item.id} key={item.id}>{item.name} ({item.observation_count})</option>)}</Select></Field>
          </> : <Field label="Named inputs">
            <Select multiple size={Math.min(7, Math.max(2, selections.length))} value={booleanInputs} onChange={(event) => setBooleanInputs(selectedValues(event))}>{selections.map((item) => <option value={item.id} key={item.id}>{item.name} ({item.observation_count})</option>)}</Select>
          </Field>}
          <div className="form-actions wide-field"><Button tone="primary" disabled={!booleanId || !booleanName || booleanRecipeInputs.length < 2 || (booleanOperator === "exclusion" && (!booleanBase || booleanExclusions.filter((id) => id !== booleanBase).length === 0)) || state === "working"} onClick={() => void run(() => onCreateSelection({
            id: booleanId,
            name: booleanName,
            zoom_id: zoomId,
            definition: { kind: "boolean", operator: booleanOperator, selection_ids: booleanRecipeInputs }
          }))}>Save Boolean selection</Button></div>
        </div>
      </details>

      <details open className="tool-section child-zoom-section">
        <summary>Create strict child zoom</summary>
        <div className="form-grid">
          <Field label="Child zoom ID"><input className="pt-input" value={zoomObjectId} onChange={(event) => setZoomObjectId(event.target.value)} /></Field>
          <Field label="Name"><input className="pt-input" value={zoomName} onChange={(event) => setZoomName(event.target.value)} /></Field>
          <Field label="Boolean operator"><Select value={zoomOperator} onChange={(event) => setZoomOperator(event.target.value as BooleanOperator)}><option>union</option><option>intersection</option><option>exclusion</option></Select></Field>
          {zoomOperator === "exclusion" ? <>
            <Field label="Base selection" hint="The child starts from this named population."><Select value={zoomBase} onChange={(event) => setZoomBase(event.target.value)}><option value="">Choose base</option>{selections.map((item) => <option value={item.id} key={item.id}>{item.name} ({item.observation_count})</option>)}</Select></Field>
            <Field label="Selections to exclude"><Select multiple size={Math.min(7, Math.max(2, selections.length))} value={zoomExclusions} onChange={(event) => setZoomExclusions(selectedValues(event))}>{selections.filter((item) => item.id !== zoomBase).map((item) => <option value={item.id} key={item.id}>{item.name} ({item.observation_count})</option>)}</Select></Field>
          </> : <Field label="Named selections"><Select multiple size={Math.min(7, Math.max(2, selections.length))} value={zoomInputs} onChange={(event) => setZoomInputs(selectedValues(event))}>{selections.map((item) => <option value={item.id} key={item.id}>{item.name} ({item.observation_count})</option>)}</Select></Field>}
          <div className={`subset-preview wide-field ${strict ? "valid" : "invalid"}`}>
            <strong>Browser preview: {formatCount(previewIds.length)} / {formatCount(zoom.observation_count)} parent cells</strong>
            <span>{strict ? `Valid strict subset. The child opens in explicit parent embedding ${embeddingId}.` : previewIds.length === 0 ? "Choose named inputs that resolve to a non-empty population." : "The resolved population equals the parent and is not a strict subset."}</span>
          </div>
          <div className="form-actions wide-field"><Button tone="primary" disabled={!zoomObjectId || !zoomName || zoomRecipeInputs.length === 0 || (zoomOperator === "exclusion" && (!zoomBase || zoomExclusions.filter((id) => id !== zoomBase).length === 0)) || !strict || state === "working"} onClick={() => void run(() => onCreateZoom({
            id: zoomObjectId,
            name: zoomName,
            parent_id: zoomId,
            parent_embedding_id: embeddingId,
            recipe: { operator: zoomOperator, selection_ids: zoomRecipeInputs }
          }))}>Create and open child</Button></div>
        </div>
      </details>
    </section>
  );
}
