import type { SavedSelection, WorkspaceDocument } from "./types";
import { Badge } from "./ui";
import {
  formatCount,
  formatFraction,
  selectionInputs,
  selectionMethod,
  selectionParameters,
  zoomPath
} from "./workspace";

function SelectionEvidence({ selection }: { selection: SavedSelection }) {
  const inputs = selectionInputs(selection);
  const parameters = selectionParameters(selection);
  return (
    <article className="recipe-selection">
      <div className="recipe-selection-title">
        <strong>{selection.name}</strong>
        <code>{selection.id}</code>
        <Badge>{formatCount(selection.observation_count)} resolved</Badge>
      </div>
      <dl>
        <dt>Method / kind</dt><dd>{selectionMethod(selection)}</dd>
        <dt>Inputs</dt><dd>{inputs.length > 0 ? inputs.join(", ") : "none (explicit observations)"}</dd>
        <dt>Parameters</dt><dd><code>{JSON.stringify(parameters)}</code></dd>
      </dl>
    </article>
  );
}

export function RecipePanel({ workspace, zoomId }: { workspace: WorkspaceDocument; zoomId: string }) {
  const path = zoomPath(workspace, zoomId);
  const selectionById = new Map(workspace.selections.map((selection) => [selection.id, selection]));
  const current = path[path.length - 1];
  const local = workspace.selections.filter((selection) => selection.zoom_id === zoomId);
  return (
    <section className="recipe-panel">
      <header className="section-title-row">
        <div><span>Population evidence</span><h2>Full path recipe</h2></div>
        {current && <Badge>{formatCount(current.observation_count)} cells</Badge>}
      </header>
      <div className="recipe-path">
        {path.map((zoom, index) => (
          <article className="recipe-step" key={zoom.id}>
            <div className="recipe-step-index">{index}</div>
            <div className="recipe-step-body">
              <header>
                <div><strong>{zoom.name}</strong><code>{zoom.id}</code></div>
                <span>{formatCount(zoom.observation_count)} cells / {formatFraction(zoom.fraction_of_root)} root{zoom.parent_id ? ` / ${formatFraction(zoom.fraction_of_parent)} parent` : ""}</span>
              </header>
              {!zoom.recipe ? (
                <div className="root-evidence">
                  <strong>All loaded observations</strong>
                  <span>No implicit filter. Source count: {formatCount(zoom.observation_count)}.</span>
                  <code>source {workspace.source_identity}</code>
                </div>
              ) : (
                <div className="recipe-operator">
                  <span>Boolean operator</span><strong>{zoom.recipe.operator}</strong>
                  {zoom.recipe.selection_ids.map((selectionId) => {
                    const selection = selectionById.get(selectionId);
                    return selection ? <SelectionEvidence selection={selection} key={selection.id} /> : (
                      <div className="inline-error" key={selectionId}>Missing recipe selection: {selectionId}</div>
                    );
                  })}
                </div>
              )}
            </div>
          </article>
        ))}
      </div>
      <div className="local-selection-list">
        <h3>Local named selections in {current?.name ?? zoomId}</h3>
        {local.length === 0 ? <p className="panel-empty">No local selections have been saved.</p> : local.map((selection) => (
          <SelectionEvidence selection={selection} key={selection.id} />
        ))}
      </div>
    </section>
  );
}
