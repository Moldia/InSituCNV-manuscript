# 02_Generate_template_datasets

This stage creates one full-gene simulated AnnData per CNV template. The notebook (`02_generate_template_adatas.ipynb`) loads the base dataset plus template definitions, runs CNV simulation, saves per-template files (`data/intermediate/template_adatas/*_fullgenes_simulated.h5ad`), and writes an index (`manifests/template_adata_index.csv`) used by later technical downsampling and inferCNV runs.
