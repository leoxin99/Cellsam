"""
Audit script: verify CellFinder backbone vs model/model_cp encoder (without neck).

Usage:
    conda run -n cellsam python tools/_audit_cellfinder_backbone_compare.py
"""

import sys
from collections import OrderedDict

import torch

sys.path.insert(0, "cellSAM_source")
from cellSAM import get_model


def strip_neck_keys(sd: OrderedDict) -> OrderedDict:
    return OrderedDict((k, v) for k, v in sd.items() if not k.startswith("neck."))


def summarize_pair(name_a: str, sd_a: OrderedDict, name_b: str, sd_b: OrderedDict) -> None:
    keys_a = list(sd_a.keys())
    keys_b = list(sd_b.keys())
    if keys_a != keys_b:
        print(f"[WARN] key mismatch: {name_a} vs {name_b}")
        print(f"  len(keys_a)={len(keys_a)} len(keys_b)={len(keys_b)}")
        only_a = sorted(set(keys_a) - set(keys_b))
        only_b = sorted(set(keys_b) - set(keys_a))
        print(f"  keys only in {name_a}: {len(only_a)}")
        print(f"  keys only in {name_b}: {len(only_b)}")
        if only_a:
            print(f"  sample only_a: {only_a[:3]}")
        if only_b:
            print(f"  sample only_b: {only_b[:3]}")
        return

    n_same = 0
    n_diff = 0
    for k in keys_a:
        max_abs = (sd_a[k].float() - sd_b[k].float()).abs().max().item()
        if max_abs < 1e-8:
            n_same += 1
        else:
            n_diff += 1
    print(f"{name_a} vs {name_b}: same={n_same}, diff={n_diff}")


def main() -> None:
    print("Loading CellSAM model...")
    m = get_model()
    m.eval()

    cellfinder_body_sd = m.cellfinder.decode_head.backbone.body.state_dict()
    model_no_neck_sd = strip_neck_keys(m.model.image_encoder.state_dict())
    model_cp_no_neck_sd = strip_neck_keys(m.model_cp.image_encoder.state_dict())

    print("Counts:")
    print(f"  cellfinder backbone keys: {len(cellfinder_body_sd)}")
    print(f"  model encoder (no neck) keys: {len(model_no_neck_sd)}")
    print(f"  model_cp encoder (no neck) keys: {len(model_cp_no_neck_sd)}")

    print("Head key samples:")
    print(f"  cellfinder: {list(cellfinder_body_sd.keys())[:5]}")
    print(f"  model(no-neck): {list(model_no_neck_sd.keys())[:5]}")
    print(f"  model_cp(no-neck): {list(model_cp_no_neck_sd.keys())[:5]}")

    summarize_pair(
        "cellfinder_backbone",
        cellfinder_body_sd,
        "model_encoder_no_neck",
        model_no_neck_sd,
    )
    summarize_pair(
        "cellfinder_backbone",
        cellfinder_body_sd,
        "model_cp_encoder_no_neck",
        model_cp_no_neck_sd,
    )
    summarize_pair(
        "model_encoder_no_neck",
        model_no_neck_sd,
        "model_cp_encoder_no_neck",
        model_cp_no_neck_sd,
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

