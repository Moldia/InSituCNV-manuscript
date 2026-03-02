# 05_Compute_metrics

This stage computes quantitative performance metrics from inferred runs. The notebook (`05_compute_metrics_batch.ipynb`) reads `_CNVinf.h5ad` outputs, computes ARI/NMI/F1/AUC/PR, appends results to `results/metrics/metrics_master.csv`, logs run-level status, and supports resume mode by skipping already computed `run_id`s so crashes do not require restarting from scratch.
