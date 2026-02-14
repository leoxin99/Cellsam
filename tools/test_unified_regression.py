"""
Unified Regression Test (Phase 0 + Phase 1)

Verifies:
1. Synthetic metric correctness (BM-1to1, Coverage, Gap, PQ)
2. InferenceConfig.default() consistency
3. All imports work (no broken references)
4. train.py validate returns dict with diagnostic keys
5. All eval scripts use unified core (no local metrics)
6. Legacy pipeline is lazy-loaded (not eagerly imported)
7. Phase 1 config values, metric keys, best_dice semantics

Run: python tools/test_unified_regression.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_synthetic_metrics():
    """Verify metric correctness on synthetic data."""
    from metrics.instance_metrics import compute_all_metrics

    print("=== Test 1: Synthetic Metrics ===")

    # Scenario 1: Perfect match
    gt = np.zeros((64, 64), dtype=np.int32)
    gt[10:30, 10:30] = 1
    gt[35:55, 35:55] = 2

    m = compute_all_metrics(gt, gt)
    assert abs(m['bm_1to1_dice'] - 1.0) < 0.01, f"1to1 should be 1.0, got {m['bm_1to1_dice']}"
    assert abs(m['bm_coverage_dice'] - 1.0) < 0.01, f"Coverage should be 1.0, got {m['bm_coverage_dice']}"
    assert abs(m['gap_dice']) < 0.01, f"Gap should be 0.0, got {m['gap_dice']}"
    assert abs(m['pq'] - 1.0) < 0.01, f"PQ should be 1.0, got {m['pq']}"
    print(f"  [OK] Perfect: 1to1={m['bm_1to1_dice']:.3f} cov={m['bm_coverage_dice']:.3f} "
          f"gap={m['gap_dice']:.3f} pq={m['pq']:.3f}")

    # Scenario 2: Merged prediction (2 GT cells -> 1 pred blob)
    pred = np.zeros((64, 64), dtype=np.int32)
    pred[10:55, 10:55] = 1

    m2 = compute_all_metrics(pred, gt)
    assert m2['bm_coverage_dice'] > m2['bm_1to1_dice'], \
        f"Coverage ({m2['bm_coverage_dice']:.3f}) should > 1to1 ({m2['bm_1to1_dice']:.3f})"
    assert m2['gap_dice'] > 0, f"Gap should be > 0, got {m2['gap_dice']:.3f}"
    print(f"  [OK] Merged: 1to1={m2['bm_1to1_dice']:.3f} cov={m2['bm_coverage_dice']:.3f} "
          f"gap={m2['gap_dice']:.3f}")

    # Scenario 3: Empty prediction
    empty = np.zeros((64, 64), dtype=np.int32)
    m3 = compute_all_metrics(empty, gt)
    assert m3['bm_1to1_dice'] == 0.0
    assert m3['pq'] == 0.0
    print(f"  [OK] Empty: 1to1={m3['bm_1to1_dice']:.3f} pq={m3['pq']:.3f}")

    print("  PASS\n")


def test_config_consistency():
    """Verify InferenceConfig.default() returns consistent values."""
    from inference.core import InferenceConfig

    print("=== Test 2: InferenceConfig.default() ===")

    cfg1 = InferenceConfig.default()
    cfg2 = InferenceConfig.default()

    assert cfg1.mask_threshold == cfg2.mask_threshold == 0.5
    assert cfg1.box_expand == cfg2.box_expand == 0.1
    assert cfg1.conflict_policy == cfg2.conflict_policy == "argmax_prob"
    assert cfg1.apply_box_clipping == cfg2.apply_box_clipping == True
    assert cfg1.min_cell_area == cfg2.min_cell_area == 13884
    assert cfg1.max_cell_area == cfg2.max_cell_area == 174735

    print(f"  [OK] threshold={cfg1.mask_threshold}")
    print(f"  [OK] box_expand={cfg1.box_expand}")
    print(f"  [OK] conflict_policy={cfg1.conflict_policy}")
    print(f"  [OK] cell_area=[{cfg1.min_cell_area}, {cfg1.max_cell_area}]")
    print("  PASS\n")


def test_imports():
    """Verify all unified imports work without errors."""
    print("=== Test 3: Import Verification ===")

    # Core modules
    from inference.core import segment_with_boxes, InferenceConfig, InferenceResult, load_cellsam_checkpoint
    print("  [OK] inference.core: segment_with_boxes, InferenceConfig, InferenceResult, load_cellsam_checkpoint")

    from metrics.instance_metrics import (
        compute_all_metrics, compute_bm_1to1_dice,
        compute_bm_coverage_dice, compute_pq,
        compute_aji, compute_semantic_dice
    )
    print("  [OK] metrics.instance_metrics: all 6 public functions")

    # Verify InferenceConfig has all classmethods
    assert hasattr(InferenceConfig, 'default'), "Missing default()"
    assert hasattr(InferenceConfig, 'from_dict'), "Missing from_dict()"
    assert hasattr(InferenceConfig, 'from_yaml'), "Missing from_yaml()"
    print("  [OK] InferenceConfig: .default(), .from_dict(), .from_yaml()")

    # Verify compute_all_metrics output keys
    gt = np.zeros((32, 32), dtype=np.int32)
    gt[5:15, 5:15] = 1
    m = compute_all_metrics(gt, gt)
    expected = {
        'bm_1to1_dice', 'bm_coverage_dice', 'gap_dice',
        'pq', 'sq', 'rq', 'tp', 'fp', 'fn',
        'aji', 'semantic_dice', 'n_gt_cells', 'n_pred_cells',
    }
    missing = expected - m.keys()
    assert not missing, f"Missing keys: {missing}"
    # Verify TP/FP/FN types are int-like
    assert isinstance(m['tp'], (int, float)), f"tp should be numeric, got {type(m['tp'])}"
    assert m['tp'] == 1, f"Perfect match should have tp=1, got {m['tp']}"
    assert m['fp'] == 0, f"Perfect match should have fp=0, got {m['fp']}"
    assert m['fn'] == 0, f"Perfect match should have fn=0, got {m['fn']}"
    print(f"  [OK] compute_all_metrics returns: {sorted(expected)}")
    print(f"  [OK] TP/FP/FN: tp={m['tp']}, fp={m['fp']}, fn={m['fn']}")
    print("  PASS\n")


def test_lazy_pipeline_import():
    """Verify legacy pipeline is NOT eagerly imported via __init__.py."""
    print("=== Test 4: Lazy Pipeline Import ===")

    # Check __init__.py source for lazy pattern
    init_path = Path(__file__).parent.parent / "src" / "inference" / "__init__.py"
    source = init_path.read_text(encoding='utf-8')

    # Should NOT have 'from .pipeline import' at top level
    assert "from .pipeline import" not in source, \
        "__init__.py still eagerly imports pipeline"

    # Should have __getattr__ for lazy loading
    assert "__getattr__" in source, "__init__.py should use __getattr__ for lazy imports"
    assert "DeprecationWarning" in source, "__init__.py should warn on legacy access"

    print("  [OK] No eager 'from .pipeline import' in __init__.py")
    print("  [OK] __getattr__ lazy-load pattern present")
    print("  PASS\n")


def test_validate_returns_dict():
    """Verify train.py validate returns dict with full diagnostics."""
    print("=== Test 5: train.py validate return format ===")

    train_path = Path(__file__).parent.parent / "src" / "train.py"
    source = train_path.read_text(encoding='utf-8')

    # validate() returns dict
    assert "return {" in source and "'bm_1to1'" in source, \
        "validate() should return a dict with 'bm_1to1'"
    assert "'bm_coverage'" in source, "Missing 'bm_coverage' in return"
    assert "'gap'" in source, "Missing 'gap' in return"
    assert "'conflict'" in source, "Missing 'conflict' in return"

    # Training loop unpacks dict
    assert "val_metrics['bm_1to1']" in source, "Loop should unpack bm_1to1"
    assert "val_metrics['bm_coverage']" in source, "Loop should access bm_coverage"
    assert "val_metrics['gap']" in source, "Loop should access gap"

    # Logging line includes all diagnostics
    assert "BM-Cov" in source, "Log should print BM-Cov"
    assert "Gap" in source, "Log should print Gap"
    assert "Conflict" in source, "Log should print Conflict"

    # Uses InferenceConfig.default()
    assert "InferenceConfig.default()" in source, "Should use InferenceConfig.default()"

    print("  [OK] validate() returns dict: bm_1to1, bm_coverage, gap, pq, semantic_dice, conflict")
    print("  [OK] Training log: BM-1to1, BM-Cov, Gap, PQ, Sem, Conflict")
    print("  PASS\n")


def test_eval_scripts_unified():
    """Verify all eval scripts use unified core, no local metrics."""
    print("=== Test 6: Eval Scripts Unified Core ===")

    project_root = Path(__file__).parent.parent
    scripts = [
        ("tools/standardized_inference.py", "Oracle"),
        ("tools/evaluate_e2e.py", "E2E"),
        ("tools/comprehensive_eval.py", "Comprehensive"),
    ]

    for script_rel, name in scripts:
        source = (project_root / script_rel).read_text(encoding='utf-8')

        # Must import unified core
        assert "segment_with_boxes" in source, f"{name}: missing segment_with_boxes"
        assert "InferenceConfig" in source, f"{name}: missing InferenceConfig"
        assert "compute_all_metrics" in source, f"{name}: missing compute_all_metrics"
        assert "load_cellsam_checkpoint" in source, f"{name}: missing load_cellsam_checkpoint"

        # Must use InferenceConfig.default()
        assert "InferenceConfig.default()" in source, f"{name}: should use InferenceConfig.default()"

        # Must NOT have local metric functions
        assert "def compute_metrics" not in source, f"{name}: still has local compute_metrics!"
        assert "def segment_image" not in source, f"{name}: still has local segment_image!"

        print(f"  [OK] {name} ({script_rel}): unified core")

    print("  PASS\n")


def test_legacy_deprecated():
    """Verify run_inference.py is marked deprecated."""
    print("=== Test 7: Legacy Deprecation ===")

    run_inf = Path(__file__).parent / "run_inference.py"
    source = run_inf.read_text(encoding='utf-8')
    assert "DEPRECATED" in source, "run_inference.py should be marked DEPRECATED"
    assert "DeprecationWarning" in source, "run_inference.py should emit DeprecationWarning"
    assert "first_write" in source, "Should document the conflict policy discrepancy"

    print("  [OK] run_inference.py: DEPRECATED header + DeprecationWarning")
    print("  PASS\n")


def test_adapter_preprocessing():
    """Verify all eval scripts apply adapter preprocessing before segment_with_boxes."""
    print("=== Test 8: Adapter Preprocessing ===")

    project_root = Path(__file__).parent.parent
    scripts = [
        ("tools/standardized_inference.py", "Oracle"),
        ("tools/evaluate_e2e.py", "E2E"),
        ("tools/comprehensive_eval.py", "Comprehensive"),
    ]

    for script_rel, name in scripts:
        source = (project_root / script_rel).read_text(encoding='utf-8')

        # Must have adapter preprocessing pattern
        assert "adapter is not None" in source, \
            f"{name}: missing 'adapter is not None' check"
        assert "adapter(" in source, \
            f"{name}: adapter loaded but never called (adapter(...) missing)"
        assert "image_for_seg" in source, \
            f"{name}: should use image_for_seg variable after adapter preprocessing"

        # segment_with_boxes must use image_for_seg, not raw image
        # Check that segment_with_boxes call uses image_for_seg
        import re
        seg_calls = re.findall(r'segment_with_boxes\([^)]*image=(\w+)', source)
        for var_name in seg_calls:
            assert var_name == "image_for_seg", \
                f"{name}: segment_with_boxes uses image={var_name}, should be image_for_seg"

        print(f"  [OK] {name}: adapter preprocessing + image_for_seg in segment_with_boxes")

    print("  PASS\n")


def test_stale_tests_deprecated():
    """Verify stale test scripts are marked deprecated."""
    print("=== Test 9: Stale Tests Deprecated ===")

    stale_scripts = [
        ("tools/archive/tests_deprecated/deprecated_test_bestmatch_validation.py", "compute_best_match_dice"),
        ("tools/archive/tests_deprecated/deprecated_test_unified_inference.py", "compute_best_match_dice"),
    ]

    for script_rel, removed_api in stale_scripts:
        script_path = Path(__file__).parent.parent / script_rel
        source = script_path.read_text(encoding='utf-8')
        assert "DEPRECATED" in source, f"{script_rel}: should be marked DEPRECATED"
        assert "DeprecationWarning" in source, f"{script_rel}: should emit DeprecationWarning"
        # Must NOT still import the removed API
        assert f"import {removed_api}" not in source and \
               f"from train import load_config, validate, {removed_api}" not in source, \
            f"{script_rel}: still imports removed {removed_api}"
        print(f"  [OK] {script_rel}: DEPRECATED + removed import fixed")

    print("  PASS\n")


def test_phase1_config_smoke():
    """Verify Phase 1 config and smoke test script are valid."""
    print("=== Test 10: Phase 1 Config Smoke ===")
    import yaml

    # Check config exists and has correct values
    config_path = Path(__file__).parent.parent / "src" / "config" / "phase1_rebalance.yaml"
    assert config_path.exists(), "phase1_rebalance.yaml not found"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    assert config['training']['use_pq_early_stop'] is True, "PQ early stop should be enabled"
    assert config['loss']['boundary_weight'] == 1.5, f"boundary_weight should be 1.5, got {config['loss']['boundary_weight']}"
    assert config['loss']['contour_weight'] == 0.3, f"contour_weight should be 0.3, got {config['loss']['contour_weight']}"
    assert config['loss']['use_topology'] is False, "topology should be disabled for clean ablation"
    assert config['loss']['use_size'] is False, "size should be disabled for clean ablation"
    print("  [OK] phase1_rebalance.yaml: values correct")

    # Check smoke test uses correct metric keys
    smoke_path = Path(__file__).parent / "smoke_test_e2e.py"
    assert smoke_path.exists(), "smoke_test_e2e.py not found"
    source = smoke_path.read_text(encoding='utf-8')
    assert "bm_1to1_dice" in source, "smoke_test should use bm_1to1_dice key"
    assert "bm_coverage_dice" in source, "smoke_test should use bm_coverage_dice key"
    assert "bm_1to1'" not in source and "bm_coverage'" not in source, \
        "smoke_test should not use short key names (bm_1to1/bm_coverage)"
    print("  [OK] smoke_test_e2e.py: metric keys correct")

    # Check train.py best_dice semantic comment
    train_path = Path(__file__).parent.parent / "src" / "train.py"
    train_source = train_path.read_text(encoding='utf-8')
    assert "best-PQ epoch" in train_source, "train.py should document best_dice semantic under PQ mode"
    print("  [OK] train.py: best_dice semantic documented")

    print("  PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Unified Regression Test (Phase 0 + Phase 1)")
    print("=" * 60 + "\n")

    tests = [
        test_imports,
        test_config_consistency,
        test_synthetic_metrics,
        test_lazy_pipeline_import,
        test_validate_returns_dict,
        test_eval_scripts_unified,
        test_legacy_deprecated,
        test_adapter_preprocessing,
        test_stale_tests_deprecated,
        test_phase1_config_smoke,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Result: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)

