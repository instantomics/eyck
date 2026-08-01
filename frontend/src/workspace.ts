import type { BooleanOperator, SavedSelection, WorkspaceDocument, Zoom } from "./types";

export function formatCount(value: number): string {
  return new Intl.NumberFormat().format(value);
}

export function formatFraction(value: number): string {
  return `${(value * 100).toFixed(value < 0.01 ? 2 : 1)}%`;
}

export function zoomPath(workspace: WorkspaceDocument, zoomId: string): Zoom[] {
  const byId = new Map(workspace.zooms.map((zoom) => [zoom.id, zoom]));
  const path: Zoom[] = [];
  const seen = new Set<string>();
  let cursor = byId.get(zoomId);
  while (cursor && !seen.has(cursor.id)) {
    path.unshift(cursor);
    seen.add(cursor.id);
    cursor = cursor.parent_id ? byId.get(cursor.parent_id) : undefined;
  }
  return path;
}

export function resolveSelections(
  selections: SavedSelection[],
  operator: BooleanOperator,
  selectionIds: string[],
  population: string[]
): string[] {
  const byId = new Map(selections.map((selection) => [selection.id, selection]));
  const sets = selectionIds.flatMap((id) => {
    const selection = byId.get(id);
    return selection ? [new Set(selection.observation_ids)] : [];
  });
  if (sets.length === 0) return [];
  let resolved: Set<string>;
  if (operator === "union") {
    resolved = new Set(sets.flatMap((set) => [...set]));
  } else if (operator === "intersection") {
    resolved = new Set([...sets[0]].filter((id) => sets.slice(1).every((set) => set.has(id))));
  } else {
    const excluded = new Set(sets.slice(1).flatMap((set) => [...set]));
    resolved = new Set([...sets[0]].filter((id) => !excluded.has(id)));
  }
  return population.filter((id) => resolved.has(id));
}

export function selectionMethod(selection: SavedSelection): string {
  const definition = selection.definition;
  if ("provenance" in definition) {
    return `${definition.kind} / ${definition.provenance.method}`;
  }
  if (definition.kind === "clusters") return `clusters / ${definition.clustering_id}`;
  if (definition.kind === "marker_cutoff") {
    return `marker cutoff / ${definition.marker_program_id} ${definition.comparator} ${definition.cutoff}`;
  }
  return `boolean / ${definition.operator}`;
}

export function selectionInputs(selection: SavedSelection): string[] {
  const definition = selection.definition;
  if ("provenance" in definition) {
    return definition.provenance.input_ids;
  }
  if (definition.kind === "clusters") return [definition.clustering_id, ...definition.cluster_ids];
  if (definition.kind === "marker_cutoff") return [definition.marker_program_id];
  return definition.selection_ids;
}

export function selectionParameters(selection: SavedSelection): Record<string, unknown> {
  const definition = selection.definition;
  if ("provenance" in definition) {
    return {
      ...definition.provenance.parameters,
      embedding_id: definition.embedding_id ?? null,
      polygon_vertices: definition.polygon?.length ?? 0
    };
  }
  if (definition.kind === "clusters") return { cluster_ids: definition.cluster_ids };
  if (definition.kind === "marker_cutoff") {
    return { comparator: definition.comparator, cutoff: definition.cutoff };
  }
  return { operator: definition.operator };
}
