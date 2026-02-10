# CellSAM Entry Points

> **Updated**: 2026-02-10  
> **Authority**: This is the single source of truth for project entry points.

---

## Primary Entry Points

| Purpose | Script | Description |
|---------|--------|-------------|
| **Training** | `src/train.py` | Main training loop with unified metrics |
| **Oracle Evaluation** | `tools/standardized_inference.py` | GT-box evaluation via unified core |
| **E2E Evaluation** | `tools/evaluate_e2e.py` | DAPI detection â†?SAM segmentation |
| **Multi-model Comparison** | `tools/comprehensive_eval.py` | Compare BF vs Adapter checkpoints |
| **Regression Test** | `tools/test_unified_regression.py` | 10-test unified regression suite (Phase 0 + Phase 1) |

## Auxiliary Entry Points

| Script | Status | Notes |
|--------|--------|-------|
| `tools/compare_models.py` | auxiliary | Non-primary analysis tool |
| `tools/run_inference.py` | **DEPRECATED** | Legacy pipeline (`first_write` conflict) |

## Unified Core API

All primary entry points use:
- `inference.core.segment_with_boxes()` â€?segmentation
- `inference.core.InferenceConfig.default()` â€?configuration
- `inference.core.load_cellsam_checkpoint()` â€?model loading
- `metrics.instance_metrics.compute_all_metrics()` â€?evaluation

## Archived Scripts

See [ARCHIVE_PLAN.md](ARCHIVE_PLAN.md) for full directory structure.

| Location | Contents |
|----------|----------|
| `archive/root_scripts/` | 13 deprecated root-level scripts |
| `tools/archive/tests_deprecated/` | Pre-Phase-0 test scripts |
| `tools/archive/legacy_eval/` | Legacy evaluation scripts |
| `tools/archive/legacy_experiment/` | One-off experiment scripts |
| `tools/archive/legacy_visualization/` | Legacy visualization scripts |
| `tools/archive/legacy_compare/` | Legacy model comparison |
| `anti_test/archive/deprecated_py/` | Early detection experiments |
