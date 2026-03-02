# 00_Load_base_dataset

This stage prepares the base lung organoid AnnData used for all later simulations. The notebook (`00_preparing_lung_organoid_dataset.ipynb`) loads and standardizes the starting dataset, checks required metadata/layers, and writes a stable base object that downstream template generation can reuse without changing preprocessing each time.
