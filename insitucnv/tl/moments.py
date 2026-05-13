import numpy as np
from scipy.sparse import csr_matrix
from scvelo.preprocessing.neighbors import get_connectivities, get_n_neighs, verify_neighbors


def smooth_data_for_cnv(data, layer_w_norm_counts=None, n_neighbors=20, mode="connectivities", copy=False):
    """Smooths data for CNV inference using nearest neighbor connectivities.

    Parameters
    ----------
    data : :class:`~anndata.AnnData`
        Annotated data matrix.
    layer_w_norm_counts : str or None
        Layer name to smooth. Should be normalized but not log-transformed.
    n_neighbors : `int`, optional (default: 20)
        Number of neighbors to use for smoothing.
    mode : {'connectivities', 'distances'}, optional (default: 'connectivities')
        Metric to use for smoothing computations.

    Returns
    -------
    None or :class:`~anndata.AnnData`
        Modifies the input AnnData object in place by adding smoothed data to
        `adata.layers['M']`. If `copy=True`, returns a modified copy.
    """

    adata = data.copy() if copy else data

    # Ensure neighbor graph is computed if required
    if n_neighbors > get_n_neighs(adata):
        verify_neighbors(adata)

    # Compute smoothing based on the specified mode
    connectivities = get_connectivities(adata, mode, n_neighbors=n_neighbors, recurse_neighbors=False)

    if layer_w_norm_counts is None:
        matrix = adata.X
    else:
        if layer_w_norm_counts not in adata.layers:
            raise KeyError(
                f"Layer '{layer_w_norm_counts}' not found in adata.layers. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        matrix = adata.layers[layer_w_norm_counts]

    adata.layers["M"] = (
        csr_matrix.dot(connectivities, csr_matrix(matrix)).astype(np.float32).toarray()
    )
    
    return adata if copy else None
