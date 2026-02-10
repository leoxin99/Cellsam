# anti_test Directory

> **Status**: Archived (2026-02-10)
> **Active entry points**: See `docs/ENTRYPOINTS.md`

## Purpose

This directory contained early-stage detection and segmentation experiments
(cellfinder, DAPI detection, traditional methods). These scripts predate the
unified inference core (Phase 0) and use legacy APIs.

## Structure

```
anti_test/
├── archive/
│   └── deprecated_py/       # 13 archived .py scripts (deprecated_*)
├── fish2/                   # Sample data
├── fish3/                   # Sample data + channel_defs.json
├── *.tif                    # Annotation TIFF (kept in-place)
├── *.txt                    # Experiment result logs (kept in-place)
├── *.md                     # Analysis reports (kept in-place)
└── *.docx                   # Reference documents (kept in-place)
```

## Non-.py artifacts

The `.tif`, `.txt`, `.md`, `.docx` files are experiment outputs and reference
documents. They are kept in-place per the archive plan (A0.1b). If future
cleanup is needed, move them to `anti_test/archive/artifacts/`.

## Replacement entry points

| Old script | Replacement |
|-----------|-------------|
| test_full_pipeline.py | `tools/evaluate_e2e.py` |
| eval_metrics.py | `src/metrics/instance_metrics.py` |
| test_dapi_detection.py | `tools/evaluate_e2e.py` (DAPI detection built-in) |
| visualize_test_results.py | `tools/comprehensive_eval.py` |
