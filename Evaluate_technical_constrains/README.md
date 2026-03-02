# Evaluate Technical Constrains

This folder is a scalable version of the Figure2 workflow to evaluate technical constraints for CNV inference across multiple simulated CNV templates.

## Goal
- Reproduce the same technical-condition grid used in Figure2:
  - count fractions: `100, 70, 50, 20, 10, 5, 3, 2, 1`
  - gene panels: `all, 20000, 15000, 10000, 5000, 1000, 500`
- Extend to many CNV templates (e.g. `T01..T10`) without exploding stored `.h5ad` files.

## Recommended execution model
- Keep one base simulated dataset per template in `data/templates/`.
- Define all runs in `manifests/runs.csv`.
- For each run:
  1. Load template dataset.
  2. Apply panel mask + count downsampling in memory.
  3. Run inferCNV.
  4. Save compact run output and append metrics row.
- Persist only what is needed for reproducibility:
  - run manifest
  - run logs/status
  - metrics tables
  - selected plots

## Folder layout
- `00_Load_base_dataset/`: prepare and freeze the starting organoid dataset.
- `01_Define_CNV_templates/`: define `CNV_dict` and metadata per template.
- `02_Generate_template_datasets/`: generate one full-gene simulated dataset per template.
- `03_Apply_technical_variations/`: optional utilities for on-the-fly panel/count transforms.
- `04_Run_inferCNV/`: inferCNV execution notebook/scripts driven by manifest.
- `05_Compute_metrics/`: ARI/NMI/F1/AUC/PR calculations across runs.
- `06_Visualize_results/`: summary plots and Figure-style panels.
- `config/`: global experiment settings.
- `manifests/`: run definitions and template metadata.
- `data/`: template and intermediate data.
- `results/`: metrics, plots, and logs.
- `scripts/`: reusable helper scripts.

## Minimal workflow
1. Fill `manifests/templates.csv` with template definitions and seeds.
2. Build `manifests/runs.csv` from template IDs x counts x panel sizes.
3. Execute inferCNV from `04_Run_inferCNV/` using `run_id` as key.
4. Compute metrics in `05_Compute_metrics/` using the same `run_id` contract.
5. Plot in `06_Visualize_results/`.
