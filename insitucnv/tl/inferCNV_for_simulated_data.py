import numpy as np
import scanpy as sc
import infercnvpy as cnv
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# Suppress warnings to avoid cluttered output
warnings.simplefilter("ignore")

# Set default figure parameters for plotting
sc.settings.set_figure_params(figsize=(5, 5))



def inferCNV_for_simulated_data(path, data_name, window_size, plot_dir=None, max_resolution_steps=100):
    """
    Perform CNV (Copy Number Variation) inference on a simulated dataset using infercnvpy.

    Parameters:
    - path (str): The directory path where the dataset is stored.
    - data_name (str): The name of the dataset (without file extension).
    - window_size (int): The size of the genomic window for CNV inference.
    - plot_dir (str | None): Directory to save figures. Defaults to `path`.
    - max_resolution_steps (int): Max attempts to tune Leiden resolution to 4 clusters.

    Outputs:
    - Saves CNV heatmap and UMAP visualizations.
    - Saves the processed AnnData object with inferred CNVs.
    """

    # Load the dataset from the specified path
    adata = sc.read_h5ad(path + '/' + data_name + '.h5ad')
    
    # Extract the simulated CNV layer and assign it to the main matrix (X)
    layer = 'CNV_simulated'
    adata.X = adata.layers[layer].copy()

    # Perform CNV inference using infercnvpy
    cnv.tl.infercnv(
        adata,
        reference_key="cell_type",  # Column in adata.obs containing reference cell types
        reference_cat=['ciliated cell', 'basal cell'],  # List of reference cell types
        window_size=window_size,
        step=1,
        calculate_gene_values=True,  # Compute CNV values for genes
    )

    # Perform dimensionality reduction and clustering
    cnv.tl.pca(adata)  # Compute PCA for feature reduction
    cnv.pp.neighbors(adata)  # Compute nearest neighbors graph
    cnv.tl.leiden(adata)  # Perform clustering using the Leiden algorithm

    # Adjust clustering resolution until exactly four clusters are obtained
    resol = 0.7
    cnv.tl.leiden(adata, resolution=resol, key_added='cnv_leiden')
    target_n_clusters = 4
    step = 0.005

    for i in range(int(max_resolution_steps)):
        n_clusters = len(np.unique(adata.obs['cnv_leiden']))
        if n_clusters == target_n_clusters:
            break

        print(f'Adjusting clustering resolution ({i + 1}/{max_resolution_steps})...')
        if n_clusters < target_n_clusters:
            resol += step
            print(f'Increasing resolution to {resol}')
        else:
            resol -= step
            print(f'Reducing resolution to {resol}')

        # Keep resolution in a valid positive range.
        if resol <= 0:
            resol = step

        cnv.tl.leiden(adata, resolution=resol, key_added='cnv_leiden')
    else:
        n_clusters = len(np.unique(adata.obs['cnv_leiden']))
        raise RuntimeError(
            f'Could not reach {target_n_clusters} clusters after {max_resolution_steps} '
            f'resolution adjustments (last resolution={resol}, last_n_clusters={n_clusters}).'
        )

    # Store final clustering results with the adjusted resolution
    cnv.tl.leiden(adata, resolution=resol, key_added=f'cnv_leiden_res{round(resol, 2)}')

    # Compute UMAP + score as analysis outputs; figure rendering is delegated to save_figures
    cnv.tl.umap(adata)
    cnv.tl.cnv_score(adata, groupby='cnv_leiden')

    # Save the processed AnnData object with inferred CNVs
    new_path = path + '/' + data_name + '_CNVinf.h5ad'
    adata.write(new_path, compression='gzip')

    # Save figures with the exact same style/settings as save_figures()
    save_figures(path=path, data_name=data_name, plot_dir=plot_dir)

    print(f'CNV inference for the {data_name} dataset has been completed. The results have been saved in: {new_path}')



def save_figures(path, data_name, plot_dir=None):
    """
    Load a CNV-inferred AnnData object and save heatmap/UMAP figures.

    Parameters:
        path (str): Directory where `<data_name>_CNVinf.h5ad` is stored.
        data_name (str): Base dataset name.
    """

    # Ensure scanpy saves files to the requested output directory.
    old_figdir = sc.settings.figdir
    target_plot_dir = path if plot_dir is None else str(plot_dir)
    Path(target_plot_dir).mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = target_plot_dir

    try:
        adata = sc.read_h5ad(path + '/' + data_name + '_CNVinf.h5ad')

        cnv_leiden_colors = ["#df9a57", "#fc7a57", "#fcd757", "#a22020"]
        simulated_subclone_colors = ["#bce784", "#5dd39e", "#348aa7", "#525174"]

        if "cnv_leiden" in adata.obs.columns:
            adata.uns["cnv_leiden_colors"] = cnv_leiden_colors
            cnv.pl.chromosome_heatmap(
                adata,
                groupby="cnv_leiden",
                dendrogram=False,
                vmin=-0.4,
                vmax=0.4,
                save=f'_{data_name}.pdf'
            )
            print('CNV heatmap saved!')
        else:
            print('SKIP heatmap: "cnv_leiden" not found in adata.obs')

        umap_colors = [k for k in ["cnv_leiden", "simulated_subclone"] if k in adata.obs.columns]
        if len(umap_colors) == 0:
            print('SKIP UMAP: no valid obs columns found')
            return

        if "simulated_subclone" in adata.obs.columns:
            adata.uns["simulated_subclone_colors"] = simulated_subclone_colors

        if "X_umap" not in adata.obsm:
            cnv.tl.umap(adata)

        cnv.pl.umap(
            adata,
            color=umap_colors,
            save=f'_{data_name}.pdf'
        )
        print('CNV UMAP saved!')
    finally:
        sc.settings.figdir = old_figdir
