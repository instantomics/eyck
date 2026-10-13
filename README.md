# Eyck

Eyck is a standalone FastAPI application for hierarchical and overlapping
single-cell annotation. It discovers source-owned `eyck.toml` files and never
imports code from those source projects or mutates their H5AD inputs.

Run a local server with:

```bash
eyck annotations serve real_datasets
```

## Manifest v1

```toml
schema_version = 1

[[projects]]
id = "example"
title = "Example cells"
dataset_id = "example-v1"
source_owner = "source"
h5ad = "data/cells.h5ad"
labels = "annotations/example/labels.json"
output = "output/annotation_projects/example"
embeddings = [{ id = "umap", obsm = "X_umap" }]
metadata_columns = ["sample"]
modalities = [{ id = "crispr", obsm = "X_crispr" }]
initial_memberships = "output/annotation_projects/example/generated_initial.parquet"
overlay_columns = ["score_proliferation"]

[[projects.processes]]
id = "differentiation"
title = "Differentiation"
description = "0 = progenitor, 1 = mature."
```

All paths are relative to and confined within the directory containing the
manifest. `labels.json` has `schema_version` and a `labels` array. Each label has
an `id`, `name`, optional `description`, optional `ontology_ids`, and
`parent_ids`.

`views` optionally names a directory of derived views: extra 2D embeddings kept
beside the H5AD so that producers can add them without rewriting it. Each
`<view_id>.parquet` holds `observation_id`, `x` and `y` for any subset of the
observations (cells absent from a view have no coordinates there); its
`eyck_view` schema metadata may give a `title` and a `description` of how it was
made. Views are read on request, so new ones appear without a restart.

`overlay_columns` are observation columns offered for colouring on demand,
unlike `metadata_columns`, which are sent with every point payload. Use
`--project ID` (repeatable) to serve selected projects without parsing other
declarations.

## Process drawing

`/annotations/<project>/processes` draws continuous processes that co-occur in
one population. A process is an ordered list of parts, and each cell belongs to
the first part whose area holds it: inside any of the part's include regions
(every cell when it has none) and outside its exclude regions. Regions may lie
on any declared embedding or view and may have holes. A part has a status:

- *values*: a cell takes the arc-length position of its projection onto the
  part's nearest curve, so separate branches of one process project only onto
  their own curves. Each curve runs from `value_start` (default 0, where drawing
  began) to `value_end` (default 1). A value part with regions only marks a
  subset (value 1), so cells chosen on one view, including a view of a
  subpopulation, can be coloured on every other view.
- *unknown*: the process is active but its value is not known.
- *absent*: the process does not exist there, as in every cell no part holds.

Reordering parts sets which one wins where they overlap; parts can be merged,
and drawings moved between them. Undeclared process IDs may be added while
drawing.

The page shows one, two, four, or one large and four small plots, each with its
own embedding and colouring, so several genes or views can be compared while
drawing; plots of one embedding can share their zoom. Drawings can be selected
and reshaped (points dragged with their neighbours, added, removed, smoothed, or
the whole drawing moved) with undo, regions of one part merged, and regions cut
out of another, and cells lassoed in one plot are highlighted in all.

Drawings are GeoJSON in embedding coordinates, kept per annotation set so that
independent sets can be compared, with the ordered parts beside the features.
Each feature records the process, part, embedding, annotator, set, timestamps,
and draft or reviewed status. The autosaved draft and the reviewed copy live
under `<output>/processes/<set>/`. Export writes `values.parquet` there (one row
per observation and, per process, a nullable value column and a
`<process>.status` column) with an `export.json` that records the drawing
revision and the review status of every process.

## Annotated matrix exchange

Eyck can declaratively export annotated count matrices without importing the
producer's code or changing its H5AD and membership inputs. Each
`eyck.annotated_matrix.v1` export is a self-validating directory containing
string feature and observation axes and a canonical int32 CSR count matrix.
The format is task-neutral and records member integrity, ordered-axis and matrix
semantics, the normalized export selection, and selection outcomes.

Materialize a TOML plan and validate an export with:

```bash
eyck annotated-matrix materialize export-plan.toml exports --timeout-seconds 300
eyck annotated-matrix validate exports/example
```
