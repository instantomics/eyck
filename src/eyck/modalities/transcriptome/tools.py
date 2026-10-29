import numpy as np
import pandas as pd
import scanpy as sc


def paircorr(x, y, dim=-2):
    divisor = y.std(dim, keepdims=True) * x.std(dim, keepdims=True)
    divisor[np.isclose(divisor, 0)] = 1.0
    cor = (x - x.mean(dim, keepdims=True)) * (y - y.mean(dim, keepdims=True))  # / divisor
    return cor

def refcor(x, y, dim=0, eps=0.000001, n=1000):
    """
    Correlate two matrices x and y, where y is a matrix of features and x can be a single vector or a matrix.
    The function computes the correlation between each feature in y and the vector x.
    If x is a single vector, it returns a Series with the correlation values for each feature in y.
    If x is a matrix, it returns a matrix of correlation values.
    Parameters:
    x : np.ndarray
        A single vector or a matrix of shape (n_samples, n_features).
    y : np.ndarray
        A matrix of features of shape (n_samples, n_features).
    dim : int, optional
        The dimension along which to compute the correlation. Default is 0.
    
    """
    single = False
    if x.ndim == 1:
        x = x[:, None]
        single = True
    x = x[:n]
    y = y[:n]
    divisor = (y.std(0)[None, :] * x.std(0)[:, None]) + eps
    numerator = ((x - x.mean())[:, :, None] * (y - y.mean(0))[:, None, :]).mean(0)
    cor = numerator / divisor
    if single:
        return cor[0]
    return cor

def refjaccard(x, y, dim=0, n=1000):
    """
    Compute the Jaccard similarity between two matrices x and y, where y is a matrix of features and x can be a single vector or a matrix.
    The function computes the Jaccard similarity between each feature in y and the vector x.
    If x is a single vector, it returns a Series with the Jaccard similarity values for each feature in y.
    If x is a matrix, it returns a matrix of Jaccard similarity values.
    Parameters:
    x : np.ndarray
        A single vector or a matrix of shape (n_samples, n_features).
    y : np.ndarray
        A matrix of features of shape (n_samples, n_features).
    dim : int, optional
        The dimension along which to compute the Jaccard similarity. Default is 0.
    
    """
    single = False
    if x.ndim == 1:
        x = x[:, None]
        single = True
    x = x[:n]
    y = y[:n]
    a = (x > 0).astype(int)
    b = (y > 0).astype(int)
    
    numerator = (a[:, :, None] & b[:, None, :]).sum(0)
    denominator = (a[:, :, None] | b[:, None, :]).sum(0)
    
    jaccard_similarity = numerator / denominator
    
    if single:
        return jaccard_similarity[0]
    
    return jaccard_similarity


def correlate_single_gene(adata, gene_id, max_n = 2000):
    gene_ix = adata.var.index.get_loc(gene_id)
    try:
        X = np.array(adata.X[:max_n, :].todense())
    except AttributeError:
        X = adata.X[:max_n, :]
    x = X[:, gene_ix]
    y = np.array(X[:max_n, :])
    return pd.Series(refcor(x, y, n = max_n), adata.var.index).sort_values(ascending=False)



def correlate_feature(adata, feature, max_n = 2000):
    x = sc.get.obs_df(adata, keys=[feature]).values.flatten()[:max_n]
    try:
        X = np.array(adata.X[:max_n, :].todense())
    except AttributeError:
        X = adata.X[:max_n, :]
    y = np.array(X[:max_n, :])
    return pd.Series(refcor(x, y, n = max_n), adata.var.index).sort_values(ascending=False)


def jaccard_single_gene(adata, gene_id, max_n = 2000):
    gene_ix = adata.var.index.get_loc(gene_id)
    try:
        X = np.array(adata.X[:max_n, :].todense())
    except AttributeError:
        X = adata.X[:max_n, :]
    x = X[:, gene_ix]
    y = np.array(X[:max_n, :])
    a = x > 0
    b = y > 0

    y = (a[:, None] & b).sum(0) / (a[:, None] | b).sum(0)
    return pd.Series(y , adata.var.index).sort_values(ascending=False)



def reorder_umap(adata, ordering):
    """
    Flip UMAP x and y coordinates so that the cells with the lowest numbers or on top left
    """
    cor_x = np.corrcoef(adata.obsm["X_umap"][:, 0], ordering)[0, 1]
    cor_y = np.corrcoef(adata.obsm["X_umap"][:, 1], ordering)[0, 1]
    print(cor_x, cor_y)
    if cor_x > 0:
        adata.obsm["X_umap"][:, 0] = -adata.obsm["X_umap"][:, 0]
    if cor_y > 0:
        adata.obsm["X_umap"][:, 1] = -adata.obsm["X_umap"][:, 1]
    return adata


def score_marker_sets(adata, markers, *, gene_id_resolver=None, groupby=None, suffix="_score"):
    """Score marker sets per cell and optionally add group-level averages."""
    if gene_id_resolver is None:
        from .data import gene_id as gene_id_resolver
    score_columns = []
    for marker_name, marker_genes in markers.items():
        score_column = f"{marker_name}{suffix}"
        sc.tl.score_genes(
            adata,
            gene_list=gene_id_resolver(adata.var, marker_genes),
            score_name=score_column,
        )
        score_columns.append(score_column)

    if groupby is None:
        return None

    group_scores = adata.obs.groupby(groupby)[score_columns].mean()
    for marker_name in markers:
        adata.obs[f"{marker_name}_avg"] = (
            adata.obs[groupby].map(group_scores[f"{marker_name}{suffix}"]).astype(float)
        )
    return group_scores
