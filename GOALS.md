# Eyck annotation goals

This document defines the product goals and durable interaction model for Eyck.
Implementation plans and point-in-time architecture belong in separate documents.
When the current application conflicts with these goals, Eyck may make breaking
changes rather than preserve the old interaction model.

## North star

Eyck starts from the complete, relatively unfiltered single-cell dataset and
lets a researcher progressively zoom into smaller populations, recalculate the
local structure, and add biological annotations at any level.

The researcher must never lose how they arrived at the current population. Every
zoom exposes the complete selection recipe from the root, the cells selected at
each step, and the method and parameters used. Selections can be visualized on
the UMAP and reused to create annotations or a more specific child zoom.

The primary workflow is therefore:

1. Open all observations in the source dataset.
2. Explore and create explicit, named selections.
3. Visualize, combine, and annotate those selections.
4. Create a strict child zoom from an explicit Boolean selection recipe.
5. Recalculate neighbors, UMAP, clustering, and other local analyses as needed.
6. Repeat the process to move from broad populations to increasingly specific
   biological states while retaining the complete path and provenance.

Hierarchical zooming and annotation are the main purpose of Eyck. They are not
secondary features added to a flat UMAP viewer.

## Three distinct structures

Eyck must keep three related structures conceptually and visually distinct.

### Zoom tree

The zoom hierarchy is a rooted tree used for progressive exploration.

- The root contains every loaded observation. Eyck applies no hidden filtering
  before presenting it.
- A child zoom has exactly one parent and is a strict subset of that parent's
  observations.
- A zoom may have any number of child zooms, allowing alternative refinements
  to branch from the same population.
- The current path is always visible as a breadcrumb from the root to the
  current zoom.
- Recomputing an analysis over the same population is a revision of the current
  zoom, not another child zoom.
- Cross-branch merges are not zooms. If analyses need multiple inputs, those
  dependencies remain separate from the zoom tree.

### Label DAG

Biological labels form an identity DAG independent of the zoom tree. Labels may
have multiple parents and annotations may overlap. For example, a label can be
both a child of `portal hepatocyte` and an intersection with `sample 8` without
making the corresponding zoom a multi-parent node.

The label-DAG editor must make it easy to add and remove labels and parent edges,
inspect the resulting identities, and prevent invalid cycles. Editing label
identity does not silently restructure the zoom tree.

The editor is part of the Eyck application and visualizes labels as nodes and
parent relationships as directed edges. Simple controls support creating a
label, editing its display name, description, and ontology IDs, adding or
removing parent edges, and deleting a label. Label IDs remain stable once used.

Accepted edits atomically update the project-declared, Git-trackable
`labels.json` using optimistic revision checks. Before an edge change or label
deletion is accepted, Eyck validates cycles and membership semantics and shows
the affected child edges, annotations, membership decisions, selections, and
descendants. Confirmed deletion cascades through the previewed dependent state
as one transaction; it never leaves dangling identities or silently changes the
meaning of accepted annotations.

### Analysis dependencies

Selections, embeddings, clusterings, scores, differential-expression results,
and annotations may depend on other saved objects. Eyck records those
dependencies for reproducibility, impact previews, and safe deletion, but does
not expose the dependency graph as the primary navigation model.

## Root and current population

Opening a project shows all observations in the immutable source dataset. The
root population is represented explicitly as `all loaded observations`, with
its source identity and count, rather than as an unexplained default.

At every zoom, Eyck makes the following immediately visible:

- the current observation count and its fraction of the root and parent;
- the full breadcrumb from the root;
- the inherited population-defining selections from every ancestor;
- the local selections created at the current zoom;
- the exact Boolean recipe that produced the current population;
- the method, parameters, input identity, and resolved count for each selection;
- the active embedding, clustering, overlays, and their status; and
- annotations attached to selections in the current context.

No implicit filter or unexplained collection of highlighted cells may become a
zoom or reviewed annotation.

## Selections

A selection is a named, reusable definition over observations. Eyck supports
selection from:

- polygon or lasso geometry on a specific embedding;
- one or more clusters from a specific clustering;
- gene or marker-program expression, including average marker expression in
  clusters;
- metadata and categorical predicates;
- score or expression thresholds and scatter-based regions;
- explicit manual observation edits when no reproducible predicate is adequate;
  and
- explicit unions, intersections, and exclusions of named selections.

### Marker programs within clusterings

Marker-program selection is a first-class workflow, not merely a coloring mode.
A researcher can define and name a set of marker genes, choose an identity-
bearing saved clustering at the current zoom, calculate the marker program,
visualize its scores, and select complete clusters using an explicit cutoff.

The default calculation reproduces the established BuglerLamb and Movahedi
workflow:

1. Calculate the control-subtracted per-cell score with
   `scanpy.tl.score_genes` over the current zoom population.
2. Calculate the arithmetic mean of that cell score within each cluster of the
   selected saved clustering.
3. Associate that cluster mean with every cell in the cluster.
4. Select all cells in clusters whose mean satisfies the chosen comparator and
   cutoff.

The UI exposes both the per-cell score and the cluster mean as switchable UMAP
overlays. It also shows a cluster table with cluster identity, cell count, mean
score, cutoff result, scoring parameters, and resolved marker genes. A marker
program cannot run when a requested gene is missing or resolves ambiguously;
Eyck explains every unresolved gene rather than silently changing the program.

Each cutoff becomes a named, reusable selection rule. Rules support `>`, `>=`,
`<`, and `<=`, and can be combined through the same explicit union,
intersection, and exclusion operations as other named selections. The saved
definition binds the zoom population, expression input, Scanpy scoring
parameters and implementation identity, marker identities, clustering identity,
comparator, and cutoff. Arbitrary categorical metadata is not treated as a
clustering unless it has first been imported as an identity-bearing saved
clustering.

Selections retain their definitions, not only their resolved observation IDs.
Every selection displays how it was made, its inputs and parameters, its cell
count, and whether it is local or inherited. A researcher can inspect the
resolved cells and detect when replay changes the result.

Multiple selections can be visible on a UMAP at once. Each has layer visibility
and color controls, and one selection can be focused without hiding the others.
Overlaps remain inspectable rather than being silently resolved by precedence.
The researcher can visualize all selections, a chosen subset, or one focused
selection on the relevant UMAP.

Creating a child zoom requires a name and an explicit Boolean recipe over saved
or draft named selections. The proposed population and count are previewed on
the parent UMAP before creation. The resulting recipe becomes part of the
child's visible provenance.

## Zoom workflow and computation

A child zoom opens immediately using its selected cells in the parent UMAP
coordinates. This preserves context and does not make navigation wait for a new
embedding.

The researcher can then explicitly launch or rerun local computations, including
neighbors, UMAP, clustering, marker scoring, aggregation, and differential
expression. Eyck must:

- expose resolved parameters and project defaults before launch;
- show queued, running, completed, failed, and cancelled states in context;
- allow UMAP, clustering, and related results to be inspected independently;
- preserve the parent-coordinate view as contextual evidence; and
- make it easy to use a completed local UMAP and clustering for the next round
  of selection and zooming.

Rerunning an analysis on a saved zoom creates a revision. Before the revision is
promoted, Eyck previews every dependent selection, annotation, analysis, and
descendant zoom that may be affected. Previous accepted results remain in
history; Eyck never silently overwrites or reinterprets accepted work.

## Annotation workflow

Biological annotations attach to named selections, not to unexplained transient
highlights. A selection may receive one or more labels from the label DAG, and
overlapping selections may receive overlapping labels.

Annotations can be added at the root or at any child zoom. They remain linked to
the selection, zoom context, support population, and evidence that produced
them. Selecting an annotation in the hierarchy reveals its cells and supporting
selection on the relevant UMAP.

Zooms are exploration structure; labels are biological identities. Naming a
zoom can communicate its purpose, but the zoom name is not itself a biological
annotation.

## Hierarchy and layer navigator

The primary project navigator behaves like a hierarchy-aware layer panel. It
shows all meaningful objects in context:

- the root and child zooms;
- named selections and Boolean combinations;
- biological annotations and their labels;
- embeddings, clusterings, scores, and other key analyses; and
- relevant computation or validity state.

Objects use clear type, indentation, visibility, focus, and status indicators.
Draft, saved, computing, failed, stale, and invalid objects are visually
distinct. Draft objects appear in the hierarchy immediately and offer explicit
save and discard actions.

The navigator supports direct navigation, expansion and collapse, visibility
toggles, selection focus, rename where identity permits it, and contextual
actions such as annotate, create child zoom, rerun, inspect provenance, and
delete. It must remain understandable for a deeply nested project without
turning the zoom tree into an undifferentiated analysis DAG.

## Saving, revisions, and deletion

Exploration can remain draft until explicitly saved. Saving records sufficient
definitions, parameters, identities, and resolved summaries to reproduce the
object and explain its place in the hierarchy.

Every saved object participates in dependency-aware lifecycle management:

- changing an accepted input produces a revision rather than a silent overwrite;
- the impact on dependents is previewed before promotion;
- stale or invalid descendants remain visible and diagnosable;
- deleting any saved selection, annotation, analysis, or zoom previews all
  objects that will also be removed or invalidated; and
- confirmed deletion is atomic and preserves recoverable history rather than
  leaving a partially broken hierarchy.

Deleting a zoom cascades through its descendant zoom branch after an exact
preview and explicit confirmation. Eyck does not attempt to reparent descendants
whose population and evidence were defined through the deleted zoom.

## Success criteria

Eyck satisfies its central goal when a researcher can:

1. Begin with every loaded observation and understand exactly what the root
   represents.
2. Create, name, combine, toggle, focus, and inspect selections on the UMAP.
3. Turn an explicit selection recipe into a strict child zoom and navigate the
   complete path through a breadcrumb.
4. Enter the child immediately in parent coordinates, launch a local UMAP and
   clustering, and continue refining from the new result.
5. See the complete inherited and local selection recipe at every level,
   including methods, parameters, counts, and resolved cells.
6. Attach overlapping biological labels to selections while editing label
   identities in a separate DAG.
7. Manage zooms, selections, annotations, and analyses from one hierarchical
   layer navigator, including clearly marked drafts and computation states.
8. Revise or delete any saved object only after seeing the exact downstream
   impact, without silent loss or reinterpretation of accepted work.

Any design that makes these operations cumbersome, hides selection provenance,
starts from an unexplained filtered population, or conflates the zoom tree with
the label DAG fails the main goal even if it can display and annotate a UMAP.
