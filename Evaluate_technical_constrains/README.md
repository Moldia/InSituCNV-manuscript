# Evaluate Technical Constraints

This folder contains a complete workflow to benchmark insituCNV under controlled technical variation across multiple simulated CNV templates.

## Scope
- Generate multiple CNV template datasets (`T01..T10` or more).
- Apply a fixed technical grid:
  - count fractions: `100, 70, 50, 20, 10, 5, 3, 2, 1`
  - gene panels: `all, 20000, 15000, 10000, 5000, 1000, 500`
- Run inferCNV for each template-condition combination.
- Compute and visualize performance metrics across all runs.

## Run Contract
Each condition is identified by `run_id` and tracked in `manifests/runs.csv`.  
All downstream outputs (inference, metrics, logs, plots) are keyed by this `run_id`.

## Folder Layout
- `00_Load_base_dataset/`: load and standardize the base organoid AnnData.
- `01_Define_CNV_templates/`: define CNV templates and subclone structures.
- `02_Generate_template_datasets/`: generate one full-gene simulated AnnData per template.
- `03_Apply_technical_variations/`: create run-level inputs with panel/count constraints.
- `04_Run_inferCNV/`: run inferCNV in batch; includes rerun notebook for failed/missing runs.
- `05_Compute_metrics/`: compute ARI, NMI, F1, AUC, PR in resumable batch mode.
- `06_Visualize_results/`: generate completeness plots and performance summaries.
- `config/`: experiment settings (technical grid and defaults).
- `manifests/`: template/run registries and metadata.
- `data/`: template, intermediate, and run-level `.h5ad` files.
- `results/`: logs, metrics tables, and plots.
- `scripts/`: reusable helper scripts.

## Minimal Execution Order
1. Build templates: `01_Define_CNV_templates`.
2. Generate template AnnDatas: `02_Generate_template_datasets`.
3. Apply technical constraints and create run inputs: `03_Apply_technical_variations`.
4. Run inferCNV for all runs: `04_Run_inferCNV`.
5. Compute metrics: `05_Compute_metrics`.
6. Visualize and export summary outputs: `06_Visualize_results`.

## Main Output Files
- `manifests/templates.csv`
- `manifests/runs.csv`
- `results/logs/run_status.csv`
- `results/metrics/metrics_master.csv`
- `results/plots/summary/*`
