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
```

All paths are relative to and confined within the directory containing the
manifest. `labels.json` has `schema_version` and a `labels` array. Each label has
an `id`, `name`, optional `description`, optional `ontology_ids`, and
`parent_ids`.

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
