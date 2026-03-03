# Active Tools Scripts

> **Updated**: 2026-02-10
> **Full entry point reference**: See [ENTRYPOINTS.md](ENTRYPOINTS.md)

## Primary (unified core)

| Script | Purpose |
|--------|---------|
| `standardized_inference.py` | Oracle evaluation with GT boxes |
| `evaluate_e2e.py` | End-to-end: DAPI detection �?SAM segmentation |
| `comprehensive_eval.py` | **ARCHIVED**  `tools/archive/`. 被 `eval_ablation.py` 取代 |
| `test_unified_regression.py` | 10-test regression verification suite |

## Auxiliary

| Script | Purpose |
|--------|---------|
| `compare_models.py` | Model comparison analysis (non-primary) |
| `run_inference.py` | **DEPRECATED** �?legacy pipeline |

## Analysis & Visualization (retained, not archived)

These scripts remain in `tools/` for ongoing analysis work:

| Category | Scripts |
|----------|---------|
| ablation_* | 5 scripts (adaptive, dapi, detection params) |
| analyze_* | 7 scripts (boundary, channel, data, dapi, edge, stats) |
| compare_* | 5 scripts (actn2, baseline, boxes, segmentation, models) |
| view_* | 9 scripts (napari viewers for various data) |
| visualize_* | 8 scripts (nuclei, detection, npy, filtered, binuclear) |
| Other | 5 test_*, 2 verify_*, 2 run_*, diagnose/inspect/experiment/show (1 each) |

> **Note**: These may be archived in a future Wave 3 cleanup.
