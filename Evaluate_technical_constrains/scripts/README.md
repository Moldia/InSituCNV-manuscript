# Scripts

Place reusable helpers here, for example:
- `build_run_manifest.py`: generate `manifests/runs.csv` from templates x technical grid.
- `run_infercnv_batch.py`: run inferCNV for each `run_id`.
- `compute_metrics_batch.py`: compute and append metrics keyed by `run_id`.

Use deterministic seeds from `config/technical_grid.yaml` and always key outputs by `run_id`.
