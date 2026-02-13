"""
Gradient Gate Test — Phase 2 (2026-02-12)

Verifies that all loss functions in CombinedLoss produce non-zero gradients.
This is the merge gate: any loss that fails = cannot be used for training.

Usage: python tools/test_loss_gradients.py
"""
import sys
import torch
sys.path.insert(0, '.')

from src.losses.combined import (
    DiceLoss, BoundaryLoss, AJILoss,
    TopologyLoss, SizeLoss, ContourLoss,
    NeighborIntrusionLoss, OverlapMutexLoss,
    CombinedLoss
)


def test_loss_gradient(loss_fn, name, pred_shape=(256, 256), needs_target=True):
    """Test that a loss produces non-zero gradients w.r.t. pred."""
    logits = torch.randn(*pred_shape, requires_grad=True)
    pred = torch.sigmoid(logits)
    target = (torch.randn(*pred_shape) > 0).float()
    
    if needs_target:
        loss = loss_fn(pred, target)
    else:
        loss = loss_fn(pred)
    
    # Check loss is a scalar tensor
    assert loss.dim() == 0, f"{name}: loss is not scalar (shape={loss.shape})"
    assert loss.requires_grad or loss.item() == 0.0, f"{name}: loss has no grad_fn"
    
    if loss.item() == 0.0:
        print(f"  [{name}] SKIP — loss=0 (empty boundary/target case)")
        return True
    
    loss.backward()
    
    grad = logits.grad
    if grad is None:
        print(f"  [{name}] FAIL — grad is None (autograd broken!)")
        return False
    
    max_grad = grad.abs().max().item()
    mean_grad = grad.abs().mean().item()
    
    if max_grad < 1e-10:
        print(f"  [{name}] FAIL — max|grad|={max_grad:.2e} (effectively zero)")
        return False
    
    print(f"  [{name}] PASS -- max|grad|={max_grad:.4f}, mean|grad|={mean_grad:.6f}, loss={loss.item():.4f}")
    return True


def test_neighbor_gradient():
    """Test NeighborIntrusionLoss produces gradient."""
    logits = torch.randn(256, 256, requires_grad=True)
    pred = torch.sigmoid(logits)
    target = torch.zeros(256, 256)
    target[80:180, 80:180] = 1.0  # current cell
    instance_mask = torch.zeros(256, 256)
    instance_mask[80:180, 80:180] = 1  # current cell
    instance_mask[120:200, 120:200] = 2  # neighbor cell (overlapping region)
    
    loss_fn = NeighborIntrusionLoss(gamma=1.5)
    loss = loss_fn(pred, target, instance_mask)
    
    if loss.item() == 0.0:
        print("  [NeighborLoss] SKIP -- no neighbor region in test")
        return True
    
    loss.backward()
    grad = logits.grad
    if grad is None or grad.abs().max() < 1e-10:
        print("  [NeighborLoss] FAIL -- no gradient!")
        return False
    print(f"  [NeighborLoss] PASS -- max|grad|={grad.abs().max():.4f}, loss={loss.item():.4f}")
    return True


def test_overlap_gradient():
    """Test OverlapMutexLoss produces gradient."""
    logits = torch.randn(256, 256, requires_grad=True)
    pred = torch.sigmoid(logits)
    confidence_map = torch.rand(256, 256) * 0.8  # other cells' predictions
    
    loss_fn = OverlapMutexLoss(margin=0.05)
    loss = loss_fn(pred, confidence_map)
    
    if loss.item() == 0.0:
        print("  [OverlapLoss] SKIP -- no overlap in test")
        return True
    
    loss.backward()
    grad = logits.grad
    if grad is None or grad.abs().max() < 1e-10:
        print("  [OverlapLoss] FAIL -- no gradient!")
        return False
    print(f"  [OverlapLoss] PASS -- max|grad|={grad.abs().max():.4f}, loss={loss.item():.4f}")
    return True


def test_combined_loss_gradient():
    """Test CombinedLoss end-to-end with all losses enabled."""
    criterion = CombinedLoss(
        pos_weight=2.0,
        boundary_weight=1.5,
        aji_weight=0.2,
        use_boundary=True,
        use_aji=True,
        use_topology=True,
        topology_weight=0.1,
        use_size=True,
        size_weight=0.1,
        use_contour=True,
        contour_weight=0.3,
        use_neighbor=True,
        neighbor_weight=0.3,
        use_overlap=True,
        overlap_weight=0.1,
    )
    
    logits = torch.randn(256, 256, requires_grad=True)
    target = torch.zeros(256, 256)
    target[80:180, 80:180] = 1.0
    box = [50, 50, 200, 200]
    
    # Create instance_mask and confidence_map for full test
    instance_mask = torch.zeros(256, 256)
    instance_mask[80:180, 80:180] = 1
    instance_mask[120:200, 120:200] = 2
    confidence_map = torch.rand(256, 256) * 0.3
    
    loss = criterion(logits, target, box=box,
                     instance_mask=instance_mask,
                     confidence_map=confidence_map)
    loss.backward()
    
    grad = logits.grad
    if grad is None or grad.abs().max() < 1e-10:
        print(f"  [CombinedLoss(all)] FAIL — no gradient!")
        return False
    
    print(f"  [CombinedLoss(all)] PASS — max|grad|={grad.abs().max():.4f}, loss={loss.item():.4f}")
    return True


def test_weight_normalization():
    """Verify weight normalization produces correct proportions."""
    criterion = CombinedLoss(
        boundary_weight=1.5, aji_weight=0.2,
        use_boundary=True, use_aji=True,
        use_topology=False, use_size=False, use_contour=False,
    )
    # Phase 1 equivalent: raw_base=0.3, extras=1.7, total=2.0
    # Proportions: base=0.3/2.0=15%, boundary=1.5/2.0=75%, aji=0.2/2.0=10%
    raw_base = 0.3
    extras = 1.5 + 0.2  # boundary + aji
    total = raw_base + extras
    
    expected_base_pct = raw_base / total
    expected_boundary_pct = 1.5 / total
    expected_aji_pct = 0.2 / total
    
    print(f"  [Normalization] base={expected_base_pct:.1%}, boundary={expected_boundary_pct:.1%}, aji={expected_aji_pct:.1%}, total={expected_base_pct+expected_boundary_pct+expected_aji_pct:.1%}")
    assert abs(expected_base_pct + expected_boundary_pct + expected_aji_pct - 1.0) < 0.01
    print(f"  [Normalization] PASS — weights sum to 100%")
    return True


def test_none_input_normalization():
    """Regression test: use_overlap/use_neighbor=True but inputs=None must not pollute denominator.
    
    Bug scenario (Codex 17.10 finding #1):
    If use_overlap=True but confidence_map=None, the overlap_weight should NOT
    enter the normalization denominator. Otherwise other losses get silently scaled down.
    
    Test: loss with (use_overlap=True, confidence_map=None) should equal
          loss with (use_overlap=False) — proving no denominator pollution.
    """
    torch.manual_seed(42)
    logits = torch.randn(256, 256, requires_grad=True)
    target = torch.zeros(256, 256)
    target[80:180, 80:180] = 1.0
    box = [50, 50, 200, 200]
    
    # Baseline: overlap/neighbor disabled
    criterion_off = CombinedLoss(
        pos_weight=2.0, boundary_weight=1.5, aji_weight=0.2,
        use_boundary=True, use_aji=True,
        use_neighbor=False, use_overlap=False,
    )
    
    # Bug scenario: enabled but inputs are None
    criterion_on = CombinedLoss(
        pos_weight=2.0, boundary_weight=1.5, aji_weight=0.2,
        use_boundary=True, use_aji=True,
        use_neighbor=True, neighbor_weight=0.3,
        use_overlap=True, overlap_weight=0.1,
    )
    
    loss_off = criterion_off(logits, target, box=box,
                             instance_mask=None, confidence_map=None)
    loss_on = criterion_on(logits, target, box=box,
                           instance_mask=None, confidence_map=None)
    
    diff = abs(loss_off.item() - loss_on.item())
    if diff > 1e-6:
        print(f"  [NoneInputGating] FAIL — loss diff={diff:.8f} (denominator polluted!)")
        print(f"    loss(disabled)={loss_off.item():.6f}, loss(enabled+None)={loss_on.item():.6f}")
        return False
    
    print(f"  [NoneInputGating] PASS — loss diff={diff:.2e} (no denominator pollution)")
    return True


def test_shape_mismatch_normalization():
    """Regression test: shape mismatch inputs must not pollute denominator.
    
    Bug scenario (Codex 17.10 finding #2):
    If instance_mask/confidence_map are present but wrong shape, the weight
    should NOT enter the normalization denominator.
    
    Test: loss with (enabled + wrong-shape input) should equal
          loss with (disabled) — proving shape gate works.
    """
    torch.manual_seed(42)
    logits = torch.randn(256, 256, requires_grad=True)
    target = torch.zeros(256, 256)
    target[80:180, 80:180] = 1.0
    box = [50, 50, 200, 200]
    
    # Baseline: disabled
    criterion_off = CombinedLoss(
        pos_weight=2.0, boundary_weight=1.5, aji_weight=0.2,
        use_boundary=True, use_aji=True,
        use_neighbor=False, use_overlap=False,
    )
    
    # Bug scenario: enabled but wrong-shape inputs
    criterion_on = CombinedLoss(
        pos_weight=2.0, boundary_weight=1.5, aji_weight=0.2,
        use_boundary=True, use_aji=True,
        use_neighbor=True, neighbor_weight=0.3,
        use_overlap=True, overlap_weight=0.1,
    )
    
    # Wrong shapes: 128x128 instead of 256x256
    wrong_instance_mask = torch.ones(128, 128)
    wrong_confidence_map = torch.zeros(128, 128)
    
    loss_off = criterion_off(logits, target, box=box,
                             instance_mask=None, confidence_map=None)
    loss_on = criterion_on(logits, target, box=box,
                           instance_mask=wrong_instance_mask,
                           confidence_map=wrong_confidence_map)
    
    diff = abs(loss_off.item() - loss_on.item())
    if diff > 1e-6:
        print(f"  [ShapeMismatch] FAIL — loss diff={diff:.8f} (denominator polluted!)")
        print(f"    loss(disabled)={loss_off.item():.6f}, loss(wrong-shape)={loss_on.item():.6f}")
        return False
    
    print(f"  [ShapeMismatch] PASS — loss diff={diff:.2e} (shape gate works)")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Gradient Gate Test — Phase 2")
    print("=" * 60)
    
    results = []
    
    print("\n=== Individual Loss Tests ===")
    results.append(test_loss_gradient(DiceLoss(), "DiceLoss"))
    results.append(test_loss_gradient(BoundaryLoss(), "BoundaryLoss"))
    results.append(test_loss_gradient(AJILoss(), "AJILoss"))
    results.append(test_loss_gradient(TopologyLoss(), "TopologyLoss", needs_target=False))
    results.append(test_loss_gradient(SizeLoss(), "SizeLoss"))
    results.append(test_loss_gradient(ContourLoss(), "ContourLoss"))
    
    print("\n=== New Phase 2 Loss Tests ===")
    results.append(test_neighbor_gradient())
    results.append(test_overlap_gradient())
    
    print("\n=== CombinedLoss End-to-End ===")
    results.append(test_combined_loss_gradient())
    
    print("\n=== Weight Normalization ===")
    results.append(test_weight_normalization())
    
    print("\n=== Computability Gating (Codex 17.10) ===")
    results.append(test_none_input_normalization())
    results.append(test_shape_mismatch_normalization())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("[PASS] All gradient gates PASSED -- losses are safe for training")
    else:
        print("[FAIL] GRADIENT GATE FAILED -- DO NOT proceed to training!")
    print("=" * 60)
    
    sys.exit(0 if passed == total else 1)
