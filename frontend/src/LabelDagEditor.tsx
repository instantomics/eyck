import { useEffect, useMemo, useRef, useState } from "react";
import { getLabels, previewLabelMutation, putLabels } from "./api";
import type { Label, LabelImpact, LabelMutation, LabelState } from "./types";
import { Badge, Button, Field, Notice, Select } from "./ui";
import { formatCount } from "./workspace";

interface LabelDagEditorProps {
  open: boolean;
  labelsUrl: string;
  csrfToken: string;
  onClose: () => void;
  onSaved: (state: LabelState) => Promise<void>;
}

interface NodePosition { x: number; y: number; }

function mutation(labels: Label[], deleted: Set<string>): LabelMutation {
  return { labels, delete_label_ids: [...deleted].sort(), confirm_cascade: false };
}

function layoutLabels(labels: Label[]): { positions: Map<string, NodePosition>; width: number; height: number } {
  const byId = new Map(labels.map((label) => [label.id, label]));
  const depths = new Map<string, number>();
  const visiting = new Set<string>();
  const depth = (id: string): number => {
    const cached = depths.get(id);
    if (cached !== undefined) return cached;
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const parents = (byId.get(id)?.parent_ids ?? []).filter((parent) => byId.has(parent));
    const value = parents.length === 0 ? 0 : Math.max(...parents.map(depth)) + 1;
    visiting.delete(id);
    depths.set(id, value);
    return value;
  };
  labels.forEach((label) => depth(label.id));
  const levels = new Map<number, Label[]>();
  labels.forEach((label) => levels.set(depth(label.id), [...(levels.get(depth(label.id)) ?? []), label]));
  const maxColumns = Math.max(1, ...[...levels.values()].map((items) => items.length));
  const positions = new Map<string, NodePosition>();
  levels.forEach((items, level) => {
    items.sort((a, b) => a.name.localeCompare(b.name));
    const rowWidth = items.length * 210;
    const offset = (maxColumns * 210 - rowWidth) / 2;
    items.forEach((label, index) => positions.set(label.id, { x: 30 + offset + index * 210, y: 34 + level * 150 }));
  });
  return { positions, width: maxColumns * 210 + 60, height: (Math.max(0, ...depths.values()) + 1) * 150 + 40 };
}

export function LabelDagEditor({ open, labelsUrl, csrfToken, onClose, onSaved }: LabelDagEditorProps) {
  const [state, setState] = useState<LabelState | null>(null);
  const [labels, setLabels] = useState<Label[]>([]);
  const [deleted, setDeleted] = useState<Set<string>>(new Set());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [newId, setNewId] = useState("");
  const [newName, setNewName] = useState("");
  const [impact, setImpact] = useState<LabelImpact | null>(null);
  const [confirmCascade, setConfirmCascade] = useState(false);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const [refreshError, setRefreshError] = useState("");
  const [impactPayloadKey, setImpactPayloadKey] = useState("");
  const previewGeneration = useRef(0);
  const previousFocus = useRef<HTMLElement | null>(null);
  const layout = useMemo(() => layoutLabels(labels), [labels]);
  const selected = labels.find((label) => label.id === selectedId) ?? null;

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setWorking(true);
    setError("");
    setRefreshError("");
    getLabels(labelsUrl, controller.signal).then((loaded) => {
      setState(loaded);
      setLabels(loaded.labels);
      setDeleted(new Set());
      setSelectedId(loaded.labels[0]?.id ?? null);
      setImpact(null);
      setImpactPayloadKey("");
      setConfirmCascade(false);
    }).catch((caught: unknown) => {
      if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : "Could not load labels");
    }).finally(() => { if (!controller.signal.aborted) setWorking(false); });
    return () => controller.abort();
  }, [labelsUrl, open]);

  useEffect(() => {
    if (!open) return;
    previousFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    window.setTimeout(() => document.getElementById("label-dag-dialog")?.focus(), 0);
    return () => {
      const restore = previousFocus.current;
      window.setTimeout(() => restore?.focus(), 0);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || working) return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, open, working]);

  const invalidate = () => {
    previewGeneration.current += 1;
    setImpact(null);
    setImpactPayloadKey("");
    setConfirmCascade(false);
    setError("");
    setRefreshError("");
  };

  const updateSelected = (patch: Partial<Label>) => {
    if (!selectedId) return;
    setLabels((current) => current.map((label) => label.id === selectedId ? { ...label, ...patch } : label));
    invalidate();
  };

  const preview = async (nextDeleted = deleted) => {
    if (!state) return;
    const payload = mutation(labels, nextDeleted);
    const payloadKey = JSON.stringify(payload);
    const generation = ++previewGeneration.current;
    setWorking(true);
    setError("");
    setRefreshError("");
    setImpact(null);
    setConfirmCascade(false);
    try {
      const loaded = await previewLabelMutation(labelsUrl, csrfToken, payload);
      if (previewGeneration.current !== generation) return;
      setImpact(loaded);
      setImpactPayloadKey(payloadKey);
    } catch (caught) {
      if (previewGeneration.current === generation) setError(caught instanceof Error ? caught.message : "Label impact preview failed");
    } finally {
      if (previewGeneration.current === generation) setWorking(false);
    }
  };

  const saveLabels = async () => {
    if (!state || !impact) return;
    const payload = mutation(labels, deleted);
    if (JSON.stringify(payload) !== impactPayloadKey || impact.labels_revision !== state.revision) {
      invalidate();
      setError("The label proposal changed after preview. Preview the exact current content again.");
      return;
    }
    const requiresCascade = impact.removed_label_ids.length > 0 || impact.child_edge_ids.length > 0;
    setWorking(true);
    setError("");
    setRefreshError("");
    let saved: LabelState;
    try {
      saved = await putLabels(labelsUrl, impact.labels_revision, csrfToken, {
        ...payload,
        confirm_cascade: requiresCascade && confirmCascade,
        expected_membership_revision: impact.membership_revision,
        expected_impact_sha256: impact.impact_sha256
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not save labels");
      setWorking(false);
      return;
    }
    setState(saved);
    setLabels(saved.labels);
    setDeleted(new Set());
    setImpact(null);
    setImpactPayloadKey("");
    setConfirmCascade(false);
    try {
      await onSaved(saved);
    } catch (caught) {
      setRefreshError(`Labels were committed, but the interface refresh failed: ${caught instanceof Error ? caught.message : "refresh required"}. Reload project data; do not repeat the label mutation.`);
    } finally {
      setWorking(false);
    }
  };

  if (!open) return null;
  return (
    <div className="modal-backdrop label-modal-backdrop">
      <section id="label-dag-dialog" className="modal-card label-dag-modal" role="dialog" aria-modal="true" aria-labelledby="label-dag-title" tabIndex={-1}>
        <header className="modal-header">
          <div><span>Identity structure, separate from zoom hierarchy</span><h2 id="label-dag-title">Visual label DAG editor</h2></div>
          <div className="modal-header-actions"><Badge>{labels.length} nodes</Badge><Button id="label-dag-close" disabled={working} onClick={onClose}>Close</Button></div>
        </header>
        {error && <Notice tone="danger" className="modal-notice" role="alert"><strong>Label update blocked.</strong> {error}</Notice>}
        {refreshError && <Notice tone="warning" className="modal-notice" role="status">{refreshError}</Notice>}
        <div className="label-editor-grid">
          <div className="dag-viewport">
            {working && !state ? <Notice>Loading current LabelState...</Notice> : (
              <div className="dag-canvas" style={{ width: layout.width, height: layout.height }}>
                <svg width={layout.width} height={layout.height} aria-label="Label parent relationships">
                  <defs><marker id="dag-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" /></marker></defs>
                  {labels.flatMap((label) => label.parent_ids.map((parentId) => {
                    const child = layout.positions.get(label.id);
                    const parent = layout.positions.get(parentId);
                    if (!child || !parent) return null;
                    return <path key={`${label.id}-${parentId}`} className={deleted.has(label.id) || deleted.has(parentId) ? "deleted-edge" : ""} d={`M ${parent.x + 82} ${parent.y + 76} C ${parent.x + 82} ${parent.y + 112}, ${child.x + 82} ${child.y - 36}, ${child.x + 82} ${child.y}`} markerEnd="url(#dag-arrow)" />;
                  }))}
                </svg>
                {labels.map((label) => {
                  const position = layout.positions.get(label.id)!;
                  return <button type="button" key={label.id} className={`dag-node ${selectedId === label.id ? "selected" : ""} ${deleted.has(label.id) ? "pending-delete" : ""}`} style={{ left: position.x, top: position.y }} onClick={() => setSelectedId(label.id)}><strong>{label.name}</strong><code>{label.id}</code><span>{label.parent_ids.length === 0 ? "root identity" : `${label.parent_ids.length} parent${label.parent_ids.length === 1 ? "" : "s"}`}</span>{deleted.has(label.id) && <b>pending cascade</b>}</button>;
                })}
              </div>
            )}
          </div>
          <aside className="label-inspector">
            <section className="label-add">
              <h3>Add label</h3>
              <Field label="Stable ID"><input className="pt-input" disabled={working} value={newId} onChange={(event) => setNewId(event.target.value)} /></Field>
              <Field label="Display name"><input className="pt-input" disabled={working} value={newName} onChange={(event) => setNewName(event.target.value)} /></Field>
              <Button tone="primary" disabled={working || !newId || !newName || labels.some((label) => label.id === newId)} onClick={() => {
                const added: Label = { id: newId, name: newName, description: "", ontology_ids: [], parent_ids: [] };
                setLabels((current) => [...current, added]);
                setSelectedId(newId);
                setNewId(""); setNewName(""); invalidate();
              }}>Add stable node</Button>
            </section>
            {selected ? (
              <section className="label-edit">
                <h3>Selected node</h3>
                <Field label="Immutable ID"><input className="pt-input" value={selected.id} readOnly /></Field>
                <Field label="Display name"><input className="pt-input" disabled={working} value={selected.name} onChange={(event) => updateSelected({ name: event.target.value })} /></Field>
                <Field label="Description"><textarea className="pt-input" disabled={working} rows={4} value={selected.description} onChange={(event) => updateSelected({ description: event.target.value })} /></Field>
                <Field label="Ontology IDs" hint="Comma-separated stable ontology identifiers."><input className="pt-input" disabled={working} value={selected.ontology_ids.join(", ")} onChange={(event) => updateSelected({ ontology_ids: event.target.value.split(",").map((item) => item.trim()).filter(Boolean) })} /></Field>
                <Field label="Parents" hint="Multi-select adds and removes directed parent edges.">
                  <Select multiple disabled={working} size={Math.min(10, Math.max(3, labels.length - 1))} value={selected.parent_ids} onChange={(event) => updateSelected({ parent_ids: [...event.currentTarget.selectedOptions].map((option) => option.value) })}>
                    {labels.filter((label) => label.id !== selected.id && !deleted.has(label.id)).map((label) => <option value={label.id} key={label.id}>{label.name} [{label.id}]</option>)}
                  </Select>
                </Field>
                {deleted.has(selected.id) ? <Button disabled={working} onClick={() => { setDeleted((current) => { const next = new Set(current); next.delete(selected.id); return next; }); invalidate(); }}>Undo pending deletion</Button> : <Button tone="danger" disabled={working} onClick={() => {
                  if (!state?.labels.some((label) => label.id === selected.id)) {
                    setLabels((current) => current.filter((label) => label.id !== selected.id).map((label) => ({ ...label, parent_ids: label.parent_ids.filter((parent) => parent !== selected.id) })));
                    setSelectedId(null);
                    invalidate();
                    return;
                  }
                  const next = new Set(deleted).add(selected.id);
                  setDeleted(next);
                  void preview(next);
                }}>{state?.labels.some((label) => label.id === selected.id) ? "Preview recursive deletion" : "Remove unsaved node"}</Button>}
              </section>
            ) : <Notice>Select a node card to inspect it.</Notice>}
          </aside>
        </div>
        {impact && <section className="label-impact">
          <h3>Server impact preview</h3>
          <div className="impact-counts"><Badge tone="success">Added: {impact.added_label_ids.join(", ") || "none"}</Badge><Badge>Changed: {impact.changed_label_ids.join(", ") || "none"}</Badge><Badge tone={impact.removed_label_ids.length ? "danger" : "default"}>Removed: {impact.removed_label_ids.join(", ") || "none"}</Badge></div>
          <dl><dt>Changed child edges</dt><dd>{impact.child_edge_ids.join(", ") || "none"}</dd><dt>Affected memberships</dt><dd>{formatCount(impact.membership_decision_count)} decisions / {formatCount(impact.membership_observation_ids.length)} cells</dd><dt>Affected selections</dt><dd>{impact.selection_ids.join(", ") || "none"}</dd></dl>
          {(impact.removed_label_ids.length > 0 || impact.child_edge_ids.length > 0) && <>
            {impact.membership_decision_count > 0 && <Notice tone="danger"><strong>Annotation invalidation:</strong> this semantic label change will invalidate {formatCount(impact.membership_decision_count)} membership decisions across {formatCount(impact.membership_observation_ids.length)} cells.</Notice>}
            <label className="confirm-check"><input type="checkbox" checked={confirmCascade} onChange={(event) => setConfirmCascade(event.target.checked)} /><span>I confirm the exact semantic edge/deletion impact and annotation invalidation shown by the server preview.</span></label>
          </>}
        </section>}
        <footer className="modal-footer">
          <span>{state ? `Label revision ${state.revision.slice(0, 12)}` : "Loading revision"}</span>
          <Button disabled={!state || working} onClick={() => void preview()}>Preview all changes</Button>
          <Button tone="primary" disabled={!state || !impact || working || ((impact.removed_label_ids.length > 0 || impact.child_edge_ids.length > 0) && !confirmCascade) || JSON.stringify(mutation(labels, deleted)) !== impactPayloadKey} onClick={() => void saveLabels()}>{working ? "Saving..." : "Save reviewed LabelState"}</Button>
        </footer>
      </section>
    </div>
  );
}
