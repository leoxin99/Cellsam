"""T11 LoRA implementation verification script.

Checks all review acceptance criteria:
  V1: Config validation
  V2: P1-1 freeze + LoRA order
  V3: Checkpoint save/load round-trip (P0-2)
  V4: LoRA gradient flow (must have nonzero grads after backward)
"""
import sys
import os
import tempfile

import torch
import yaml

sys.path.insert(0, "src")
sys.path.insert(0, "cellSAM_source")

from cellSAM import get_model
from lora import apply_lora_to_encoder, get_lora_state_dict, has_lora_keys


def test_config_validation():
    print("=== V1: Config Validation ===")
    for fname in ["src/config/t11_lora_r4.yaml", "src/config/t11_lora_r8.yaml"]:
        with open(fname, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["model"]["use_lora"] is True, f"{fname}: use_lora missing"
        rank = cfg["model"]["lora_rank"]
        assert rank in [4, 8], f"{fname}: invalid rank {rank}"
        assert cfg["model"]["freeze_encoder"] is True
        assert cfg["model"]["checkpoint"] is not None
        print(f"  {fname}: OK (lora_rank={rank})")
    print("PASS")


def test_freeze_then_lora():
    print("\n=== V2: P1-1 Freeze + LoRA Order ===")
    m = get_model()
    for p in m.model.image_encoder.parameters():
        p.requires_grad = False
    total_frozen = sum(1 for p in m.model.image_encoder.parameters()
                       if not p.requires_grad)
    print(f"  After freeze: {total_frozen} frozen tensors")

    apply_lora_to_encoder(m.model.image_encoder, rank=4)
    lora_trainable = [p for p in m.model.image_encoder.parameters()
                      if p.requires_grad]
    n_lora = sum(p.numel() for p in lora_trainable)
    print(f"  After LoRA: {len(lora_trainable)} trainable, {n_lora:,} params")
    assert len(lora_trainable) == 48, f"Expected 48, got {len(lora_trainable)}"
    assert n_lora == 147456, f"Expected 147456, got {n_lora}"
    print("PASS: freeze then LoRA works correctly")
    return m


def test_checkpoint_roundtrip(m):
    print("\n=== V3: Checkpoint Save/Load Round-trip ===")
    x = torch.randn(1, 3, 1024, 1024)
    m.eval()
    with torch.no_grad():
        img = m.sam_preprocess(x)
        emb_before = m.model.image_encoder(img).clone()

    tmpdir = tempfile.mkdtemp()
    ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
    checkpoint = {
        "model_state_dict": m.state_dict(),
        "config": {"model": {"use_lora": True, "lora_rank": 4}},
        "epoch": 1,
        "best_dice": 0.5,
    }
    torch.save(checkpoint, ckpt_path)
    print(f"  Saved to {ckpt_path}")

    from inference.core import load_cellsam_checkpoint
    m2, _, info = load_cellsam_checkpoint(ckpt_path, device="cpu")
    print(f"  Loaded: {info}")
    assert info.get("has_lora", False), "has_lora should be True"

    with torch.no_grad():
        img2 = m2.sam_preprocess(x)
        emb_after = m2.model.image_encoder(img2)

    diff = (emb_before - emb_after).abs().max().item()
    print(f"  Max diff between save/load: {diff:.8f}")
    assert diff < 1e-5, f"Round-trip failed, max diff: {diff}"
    print("PASS: checkpoint round-trip matches")

    os.remove(ckpt_path)
    os.rmdir(tmpdir)


def test_gradient_flow():
    """V4: Verify LoRA params receive nonzero gradients after backward."""
    print("\n=== V4: LoRA Gradient Flow ===")
    m = get_model()
    # Simulate create_model: freeze -> LoRA
    for p in m.model.image_encoder.parameters():
        p.requires_grad = False
    apply_lora_to_encoder(m.model.image_encoder, rank=4)

    # Forward (NO no_grad — simulates use_lora=True in train_one_epoch)
    x = torch.randn(1, 3, 1024, 1024)
    img = m.sam_preprocess(x)
    emb = m.model.image_encoder(img)

    # Backward
    loss = emb.sum()
    loss.backward()

    # Check LoRA params have nonzero gradients
    lora_params = [(n, p) for n, p in m.model.image_encoder.named_parameters()
                   if p.requires_grad and 'lora_' in n]
    has_grad = sum(1 for _, p in lora_params
                   if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  LoRA params with nonzero grad: {has_grad}/{len(lora_params)}")
    assert has_grad > 0, "No LoRA param has gradient!"

    # Verify base encoder params have NO gradient (frozen)
    base_with_grad = sum(1 for n, p in m.model.image_encoder.named_parameters()
                         if 'lora_' not in n and p.grad is not None
                         and p.grad.abs().sum() > 0)
    print(f"  Base encoder params with grad: {base_with_grad} (should be 0)")
    assert base_with_grad == 0, "Base encoder has gradients — freeze broken!"
    print("PASS: gradients flow through LoRA only")


if __name__ == "__main__":
    test_config_validation()
    m = test_freeze_then_lora()
    test_checkpoint_roundtrip(m)
    test_gradient_flow()
    print("\nALL VERIFICATION PASSED")
