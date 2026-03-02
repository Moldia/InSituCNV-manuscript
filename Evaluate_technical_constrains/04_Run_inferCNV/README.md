# 04_Run_inferCNV

This stage runs inferCNV across the run manifest and stores inferred outputs. The main notebook (`04_run_infercnv_batch.ipynb`) executes inference per `run_id`, saves `_CNVinf.h5ad` files and plots, and logs status to `results/logs/run_status.csv`. The rerun notebook (`04b_rerun_missing_infercnv.ipynb`) targets only missing/failed runs from `results/plots/summary/runs_missing_metrics.csv` with restart-safe per-run manifest updates.
