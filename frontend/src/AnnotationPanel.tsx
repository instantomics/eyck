import { useEffect, useMemo, useRef, useState } from "react";
import type {
  LabelDescriptor,
  MembershipDraft,
  MembershipState,
  PointsPayload,
  SavedSelection,
  Scalar,
  Zoom
} from "./types";
import { Badge, Button, Field, Notice, PanelHeader } from "./ui";

interface AnnotationPanelProps {
  labels: LabelDescriptor[];
  points: PointsPayload;
  activeIndex: number | null;
  focusedSelection: SavedSelection | null;
  zoom: Zoom;
  draft: MembershipDraft | null;
  origin: string;
  destructiveDisabled: boolean;
  destructiveDisabledReason: string;
  onOpenLabels: () => void;
  onApply: (entityId: string, labelId: string, state: MembershipState) => void;
}

function displayValue(value: Scalar | undefined): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toPrecision(5);
  return String(value);
}

export function AnnotationPanel({
  labels,
  points,
  activeIndex,
  focusedSelection,
  zoom,
  draft,
  origin,
  destructiveDisabled,
  destructiveDisabledReason,
  onOpenLabels,
  onApply
}: AnnotationPanelProps) {
  const entityIds = useMemo(() => {
    if (!draft) return [];
    return [...new Set(draft.rows.map((row) => row.entity_id))].sort();
  }, [draft]);
  const [entityId, setEntityId] = useState("entity-1");
  const initializedEntity = useRef(false);
  const labelRows = useRef<Array<HTMLDivElement | null>>([]);
  const labelDepths = useMemo(() => {
    const byId = new Map(labels.map((label) => [label.label_id, label]));
    const depths = new Map<string, number>();
    const visiting = new Set<string>();
    const depth = (labelId: string): number => {
      const cached = depths.get(labelId);
      if (cached !== undefined) return cached;
      if (visiting.has(labelId)) return 0;
      visiting.add(labelId);
      const parents = byId.get(labelId)?.parents ?? [];
      const value = parents.length === 0 ? 0 : 1 + Math.max(...parents.map(depth));
      visiting.delete(labelId);
      depths.set(labelId, value);
      return value;
    };
    labels.forEach((label) => depth(label.label_id));
    return depths;
  }, [labels]);

  useEffect(() => {
    if (initializedEntity.current || entityIds.length === 0) return;
    initializedEntity.current = true;
    setEntityId(entityIds[0]);
  }, [entityIds]);

  const entityValid = entityId.length > 0
    && entityId.length <= 128
    && !/[\u0000-\u001f\u007f]/.test(entityId);
  const selectedObservationIds = focusedSelection?.observation_ids ?? [];
  const selectedSet = useMemo(() => new Set(selectedObservationIds), [selectedObservationIds]);
  const activeObservationId = activeIndex === null ? null : points.observation_ids[activeIndex];

  const stateForLabel = (labelId: string): MembershipState | "mixed" | null => {
    if (!draft || !focusedSelection || selectedSet.size === 0) return null;
    const rows = new Map(draft.rows
      .filter((row) => row.support_id === focusedSelection.id && row.entity_id === entityId && row.label_id === labelId)
      .map((row) => [row.observation_id, row.state]));
    const states = selectedObservationIds.map((id) => rows.get(id) ?? "unreviewed");
    return states.every((state) => state === states[0]) ? states[0] : "mixed";
  };

  const handleLabelKey = (
    event: React.KeyboardEvent<HTMLDivElement>,
    index: number,
    labelId: string
  ): void => {
    const states: Record<string, MembershipState> = { p: "present", a: "absent", u: "unreviewed" };
    const state = states[event.key.toLowerCase()];
    const derived = (labels.find((label) => label.label_id === labelId)?.parents.length ?? 0) > 1;
    if (state && entityValid && draft && focusedSelection && !derived) {
      event.preventDefault();
      onApply(entityId, labelId, state);
      return;
    }
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    const next = Math.max(0, Math.min(labels.length - 1, index + (event.key === "ArrowDown" ? 1 : -1)));
    labelRows.current[next]?.focus();
  };

  return (
    <aside className="annotation-panel panel">
      <PanelHeader
        title="Annotation"
        meta={<Button disabled={destructiveDisabled} title={destructiveDisabled ? destructiveDisabledReason : "Edit label identities"} onClick={onOpenLabels}>Edit label DAG</Button>}
      />
      <div className="annotation-body">
        {!focusedSelection ? (
          <Notice>
            Focus a saved selection to annotate it. Transient lasso cells must be named and saved first.
          </Notice>
        ) : (
          <>
            <div className="annotation-target">
              <span>Named support</span>
              <strong>{focusedSelection.name}</strong>
              <code>{focusedSelection.id}</code>
              <Badge tone="success">{focusedSelection.observation_count} cells</Badge>
            </div>
            <Field label="Entity ID" hint="Overlapping entities and labels are retained independently.">
              <input
                className="pt-input"
                list="entity-options"
                value={entityId}
                onChange={(event) => setEntityId(event.target.value)}
              />
            </Field>
            <datalist id="entity-options">
              {entityIds.map((id) => <option key={id} value={id} />)}
            </datalist>
            {!entityValid && <p className="inline-error">Use 1-128 characters without control characters.</p>}
            <div className="label-heading">
              <span>Label state</span>
              <small>P/A/U set state; arrows navigate</small>
            </div>
            <div className="label-list">
              {labels.map((label, index) => {
                const current = stateForLabel(label.label_id);
                const derived = label.parents.length > 1;
                return (
                  <div
                    className="label-row"
                    key={label.label_id}
                    ref={(element) => { labelRows.current[index] = element; }}
                    tabIndex={0}
                    aria-keyshortcuts={derived ? "ArrowUp ArrowDown" : "P A U ArrowUp ArrowDown"}
                    style={{ paddingLeft: 6 + (labelDepths.get(label.label_id) ?? 0) * 12 }}
                    onKeyDown={(event) => handleLabelKey(event, index, label.label_id)}
                  >
                    <div className="label-name">
                      <strong>{label.display_name}</strong>
                      <code>{label.label_id}</code>
                      {label.parents.length > 0 && <small>Parents: {label.parents.join(" + ")}</small>}
                    </div>
                    {derived ? <div className="derived-state" title="Multiple-parent labels are computed from their parent decisions and cannot be edited directly."><strong>Derived</strong><span>from {label.parents.length} parents</span></div> : <div className="state-buttons" aria-label={`${label.display_name} state`}>
                      {(["present", "absent", "unreviewed"] as MembershipState[]).map((state) => (
                        <Button
                          key={state}
                          active={current === state}
                          className={`state-${state}`}
                          disabled={!entityValid || !draft}
                          title={`Mark ${label.display_name} ${state}`}
                          onClick={() => onApply(entityId, label.label_id, state)}
                        >
                          {state === "present" ? "P" : state === "absent" ? "A" : "U"}
                        </Button>
                      ))}
                    </div>}
                  </div>
                );
              })}
              {labels.length === 0 && <p className="panel-empty">No labels are configured.</p>}
            </div>
          </>
        )}
      </div>
      <section className="observation-inspector">
        <div className="section-heading">
          <span>Observation in {zoom.name}</span>
          {activeObservationId && <code>{activeObservationId}</code>}
        </div>
        {activeIndex === null ? (
          <p className="panel-empty">Click a point to inspect its values.</p>
        ) : (
          <dl className="value-grid">
            <dt>Embedding x</dt><dd>{displayValue(points.coordinates[activeIndex]?.[0])}</dd>
            <dt>Embedding y</dt><dd>{displayValue(points.coordinates[activeIndex]?.[1])}</dd>
            {Object.entries(points.metadata).slice(0, 10).flatMap(([key, values]) => [
              <dt key={`${key}-term`}>{key}</dt>,
              <dd key={`${key}-value`}>{displayValue(values[activeIndex])}</dd>
            ])}
          </dl>
        )}
        {origin && <div className="draft-origin">Membership source <code>{origin}</code></div>}
      </section>
    </aside>
  );
}
