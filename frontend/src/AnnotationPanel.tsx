import { useEffect, useMemo, useState } from "react";
import type {
  LabelDescriptor,
  MembershipDraft,
  MembershipState,
  PointsPayload,
  Scalar
} from "./types";
import { Badge, Button, Field, Notice, PanelHeader } from "./ui";

interface AnnotationPanelProps {
  labels: LabelDescriptor[];
  points: PointsPayload;
  activeIndex: number | null;
  selectedObservationIds: string[];
  draft: MembershipDraft | null;
  origin: string;
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
  selectedObservationIds,
  draft,
  origin,
  onApply
}: AnnotationPanelProps) {
  const entityIds = useMemo(() => {
    if (!draft) return [];
    const selected = new Set(selectedObservationIds);
    return [...new Set(draft.rows
      .filter((row) => selected.size === 0 || selected.has(row.observation_id))
      .map((row) => row.entity_id))].sort();
  }, [draft, selectedObservationIds]);
  const [entityId, setEntityId] = useState("entity-1");

  useEffect(() => {
    if (entityIds.length > 0 && !entityIds.includes(entityId)) setEntityId(entityIds[0]);
  }, [entityId, entityIds]);

  const entityValid = /^[A-Za-z0-9._-]+$/.test(entityId);
  const activeObservationId = activeIndex === null ? null : points.observation_ids[activeIndex];
  const selectedSet = useMemo(() => new Set(selectedObservationIds), [selectedObservationIds]);

  const stateForLabel = (labelId: string): MembershipState | "mixed" | null => {
    if (!draft || selectedSet.size === 0) return null;
    const states = selectedObservationIds.map((observationId) => draft.rows.find((row) => (
      row.observation_id === observationId
      && row.entity_id === entityId
      && row.label_id === labelId
    ))?.state ?? "unreviewed");
    return states.every((state) => state === states[0]) ? states[0] : "mixed";
  };

  return (
    <aside className="annotation-panel panel">
      <PanelHeader
        title="Annotation"
        meta={<Badge tone={selectedObservationIds.length > 0 ? "success" : "default"}>
          {selectedObservationIds.length} selected
        </Badge>}
      />
      <div className="annotation-body">
        {selectedObservationIds.length === 0 ? (
          <Notice>Select a point or draw a lasso to start annotating.</Notice>
        ) : (
          <>
            <Field
              label="Entity ID"
              hint="Choose an existing ID or type a new path-safe ID."
            >
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
            {!entityValid && <p className="inline-error">Use letters, numbers, dots, underscores, or hyphens.</p>}
            <div className="label-heading">
              <span>Label state</span>
              <small>Applies to all {selectedObservationIds.length} selected</small>
            </div>
            <div className="label-list">
              {labels.map((label, index) => {
                const current = stateForLabel(label.label_id);
                return (
                  <div className="label-row" key={label.label_id}>
                    <div className="label-name">
                      <strong>{label.display_name}</strong>
                      <code>{label.label_id}</code>
                      {label.parents.length > 0 && <small>Parents: {label.parents.join(", ")}</small>}
                    </div>
                    <div className="state-buttons" aria-label={`${label.display_name} state`}>
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
                    </div>
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
          <span>Observation</span>
          {activeObservationId && <code>{activeObservationId}</code>}
        </div>
        {activeIndex === null ? (
          <p className="panel-empty">Click a point to inspect its values.</p>
        ) : (
          <dl className="value-grid">
            <dt>Embedding x</dt>
            <dd>{displayValue(points.coordinates[activeIndex]?.[0])}</dd>
            <dt>Embedding y</dt>
            <dd>{displayValue(points.coordinates[activeIndex]?.[1])}</dd>
            {Object.entries(points.metadata).slice(0, 8).flatMap(([key, values]) => [
              <dt key={`${key}-term`}>{key}</dt>,
              <dd key={`${key}-value`}>{displayValue(values[activeIndex])}</dd>
            ])}
            {Object.entries(points.modalities).slice(0, 4).flatMap(([key, values]) => [
              <dt key={`${key}-term`}>{key}</dt>,
              <dd key={`${key}-value`}>{displayValue(values[activeIndex])}</dd>
            ])}
          </dl>
        )}
        {origin && <div className="draft-origin">Draft origin <code>{origin}</code></div>}
      </section>
    </aside>
  );
}
