
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from scipy.spatial.distance import pdist, squareform
from tqdm.auto import tqdm

def _calculate_spatial_cohesion(adata: ad.AnnData, cluster_key: str, spatial_key: str = 'spatial'):
    """
    Calculates a spatial cohesion score for each cluster.

    The score is based on the average pairwise distance between cells in a cluster.
    A lower distance means a more cohesive cluster. The score is normalized.

    Args:
        adata: Annotated data object.
        cluster_key: Key in `adata.obs` where cluster labels are stored.
        spatial_key: Key in `adata.obsm` where spatial coordinates are stored.

    Returns:
        A dictionary mapping cluster labels to their spatial cohesion score.
    """
    cohesion_scores = {}
    for cluster in adata.obs[cluster_key].unique():
        cluster_cells = adata.obs[cluster_key] == cluster
        if cluster_cells.sum() > 1:
            coords = adata.obsm[spatial_key][cluster_cells]
            avg_dist = pdist(coords).mean()
            cohesion_scores[cluster] = avg_dist
        else:
            cohesion_scores[cluster] = np.nan
    
    # Normalize scores (lower is better)
    max_score = max(cohesion_scores.values()) if cohesion_scores else 1
    if max_score == 0: max_score = 1

    # Invert so higher is better, and normalize
    normalized_scores = {k: 1 - (v / max_score) for k, v in cohesion_scores.items()}
    
    return np.nanmean(list(normalized_scores.values()))


def _calculate_cluster_stability(
    adata: ad.AnnData,
    resolution: float,
    obsm_key: str,
    n_subsamples: int,
    subsample_frac: float,
    n_neighbors: int = 15,
    neighbors_key: str = "cnv_neighbors",
):
    """
    Calculates cluster stability using a subsampling approach.
    """
    # Ensure neighbors graph exists for CNV representation
    if neighbors_key not in adata.uns:
        sc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=n_neighbors, key_added=neighbors_key)

    # Cluster full data
    adata_full = sc.tl.leiden(
        adata, resolution=resolution, key_added="_full", neighbors_key=neighbors_key, copy=True
    )
    full_labels = adata_full.obs["_full"]
    
    ari_scores = []
    for i in range(n_subsamples):
        # Subsample the data
        subsample_indices = np.random.choice(adata.obs_names, size=int(adata.n_obs * subsample_frac), replace=False)
        adata_sub = adata[subsample_indices, :].copy()
        
        # Cluster subsample
        sc.pp.neighbors(adata_sub, use_rep=obsm_key, n_neighbors=n_neighbors, key_added=neighbors_key)
        sc.tl.leiden(adata_sub, resolution=resolution, key_added="_sub", neighbors_key=neighbors_key)
        
        # Compare to full data clustering
        sub_labels_in_full = full_labels.loc[subsample_indices]
        ari = adjusted_rand_score(sub_labels_in_full, adata_sub.obs['_sub'])
        ari_scores.append(ari)
        
    return np.mean(ari_scores)


def find_optimal_clustering(
    adata: ad.AnnData,
    resolutions: list,
    obsm_key: str = 'X_cnv',
    spatial_key: str = 'spatial',
    n_subsamples: int = 10,
    subsample_frac: float = 0.8,
    n_neighbors: int = 15,
    neighbors_key: str = "cnv_neighbors",
) -> pd.DataFrame:
    """
    Finds the optimal clustering resolution for CNV data based on a combination of
    cluster quality, stability, and spatial cohesion.

    Args:
        adata: An anndata object with CNV data in `adata.obsm`.
        resolutions: A list of resolutions to test for Leiden clustering.
        obsm_key: The key in `adata.obsm` where the CNV matrix is stored.
        spatial_key: The key in `adata.obsm` where spatial coordinates are stored.
        n_subsamples: Number of subsamples to use for stability analysis.
        subsample_frac: Fraction of cells to use in each subsample.

    Returns:
        A pandas DataFrame with metrics for each resolution, including:
        - silhouette_score
        - davies_bouldin_score
        - stability_score (Adjusted Rand Index from subsampling)
        - spatial_cohesion_score
        - putative_normal_cluster
    """
    
    cnv_matrix = adata.obsm[obsm_key]
    # Some metrics require a dense array
    if hasattr(cnv_matrix, "toarray"):
        cnv_matrix_dense = cnv_matrix.toarray()
    else:
        cnv_matrix_dense = cnv_matrix
    results = []

    # Calculate CNV burden once
    adata.obs['cnv_burden'] = np.mean(np.abs(cnv_matrix), axis=1)

    # Ensure neighbors graph exists for CNV representation
    if neighbors_key not in adata.uns:
        sc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=n_neighbors, key_added=neighbors_key)

    for res in tqdm(resolutions, desc="Testing resolutions"):
        cluster_key = f'cnv_leiden_{res}'
        sc.tl.leiden(
            adata,
            resolution=res,
            key_added=cluster_key,
            neighbors_key=neighbors_key,
        )
        
        labels = adata.obs[cluster_key]
        n_clusters = len(labels.unique())

        if n_clusters < 2:
            continue

        # --- Quantitative Assessment ---
        sil_score = silhouette_score(cnv_matrix_dense, labels)
        db_score = davies_bouldin_score(cnv_matrix_dense, labels)

        # --- Stability Analysis ---
        stability_score = _calculate_cluster_stability(
            adata.copy(),
            res,
            obsm_key,
            n_subsamples,
            subsample_frac,
            n_neighbors=n_neighbors,
            neighbors_key=neighbors_key,
        )

        # --- Spatial Cohesion ---
        spatial_cohesion = _calculate_spatial_cohesion(adata, cluster_key, spatial_key)
        
        # --- Putative Normal Cluster ---
        normal_cluster = adata.obs.groupby(cluster_key)['cnv_burden'].mean().idxmin()

        results.append({
            'resolution': res,
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
            'davies_bouldin_score': db_score,
            'stability_score': stability_score,
            'spatial_cohesion_score': spatial_cohesion,
            'putative_normal_cluster': normal_cluster
        })

    return pd.DataFrame(results)
