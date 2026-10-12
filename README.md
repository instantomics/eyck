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

`overlay_columns` are observation columns offered for colouring on demand,
unlike `metadata_columns`, which are sent with every point payload. Use
`--project ID` (repeatable) to serve selected projects without parsing other
declarations.

## Process drawing

`/annotations/<project>/processes` draws continuous processes that co-occur in
one population. A process is one or more directed curves on one declared
embedding; each curve runs from `value_start` (default 0, where drawing began)
to `value_end` (default 1). Optional include and exclude regions, on any
declared embedding, set the scope. A cell in scope takes the arc-length position
of its projection onto the nearest curve; every other cell is not applicable.
Undeclared process IDs may be added while drawing.

Drawings are GeoJSON in embedding coordinates, kept per annotation set so that
independent sets can be compared. Each feature records the process, embedding,
annotator, set, timestamps, and draft or reviewed status. The autosaved draft
and the reviewed copy live under `<output>/processes/<set>/`. Export writes
`values.parquet` there (one row per observation, one nullable column per
process) with an `export.json` that records the drawing revision and the review
status of every process.

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
