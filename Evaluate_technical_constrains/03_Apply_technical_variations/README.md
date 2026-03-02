# 03_Apply_technical_variations

This stage applies technical constraints (gene panel size and count downsampling) to each template dataset. The notebook (`03_apply_technical_constraints.ipynb`) expands the template index into a run grid from `config/technical_grid.yaml`, generates run-specific AnnData inputs in `data/runs/`, and updates `manifests/runs.csv` as the contract for inferCNV execution.
