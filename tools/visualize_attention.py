#!/usr/bin/env python3
"""T20: Visualize CellSAM decoder attention and encoder features.

R1-approved approach: Method A (encoder feature maps) + Method C (decoder cross-attention).
Method B (Grad-CAM) is reserved for future LoRA experiments.

Usage:
    python tools/visualize_attention.py \\
        --checkpoint checkpoints/best_model.pt \\
        --image data/test/image_001.tif \\
        --box "100,200,300,400" \\
        -o figures/attention/

Dependencies: torch, matplotlib, numpy, segment_anything
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_model(checkpoint_path: Path, device: str = "cuda"):
    """Load CellSAM model from checkpoint."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cellSAM_source"))
    from cellSAM.sam_inference import CellSAM

    model = CellSAM()
    state = torch.load(str(checkpoint_path), map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def load_image(image_path: Path, channel: int = 0) -> np.ndarray:
    """Load TIFF image."""
    from tifffile import imread
    img = imread(str(image_path))
    if img.ndim == 3 and img.shape[0] < img.shape[1]:
        # Channel-first TIFF
        img = img[channel]
    return img


def extract_encoder_features(model, image_tensor: torch.Tensor, device: str = "cuda"):
    """Method A: Extract intermediate encoder feature maps.

    Hooks into ViT encoder blocks to capture feature maps at
    selected layers (early, middle, late).
    """
    features = {}
    hooks = []

    # Hook ViT encoder blocks
    mdl = model.model_cp if model.adv_mode else model.model
    encoder = mdl.image_encoder

    # Select layers: first, middle, last block
    blocks = encoder.blocks
    n_blocks = len(blocks)
    target_layers = [0, n_blocks // 2, n_blocks - 1]

    for layer_idx in target_layers:
        def hook_fn(module, input, output, idx=layer_idx):
            features[f"block_{idx}"] = output.detach().cpu()
        h = blocks[layer_idx].register_forward_hook(hook_fn)
        hooks.append(h)

    # Forward pass
    with torch.no_grad():
        embeddings = encoder(image_tensor.to(device))

    # Remove hooks
    for h in hooks:
        h.remove()

    return features, embeddings


def extract_decoder_attention(model, image_embedding: torch.Tensor,
                               box: torch.Tensor, device: str = "cuda"):
    """Method C: Extract decoder cross-attention weights.

    Hooks into SAM mask decoder's transformer cross-attention layers.
    """
    attention_maps = {}
    hooks = []

    mdl = model.model_cp if model.adv_mode else model.model

    # Hook mask decoder transformer layers
    decoder = mdl.mask_decoder
    transformer = decoder.transformer

    # Hook the cross-attention in each layer
    for layer_idx, layer in enumerate(transformer.layers):
        def hook_fn(module, input, output, idx=layer_idx):
            # Cross-attention: queries attend to image tokens
            # We capture the attention weights if available
            if isinstance(output, tuple) and len(output) > 1:
                attention_maps[f"layer_{idx}_cross_attn"] = output
            else:
                attention_maps[f"layer_{idx}_output"] = output.detach().cpu() if hasattr(output, 'detach') else output
        h = layer.cross_attn_token_to_image.register_forward_hook(hook_fn)
        hooks.append(h)

    # Forward pass through decoder
    with torch.no_grad():
        input_box = box.unsqueeze(0).unsqueeze(0).to(device)
        sparse_emb, dense_emb = mdl.prompt_encoder(
            points=None, boxes=input_box, masks=None
        )
        low_res_masks, iou_pred = mdl.mask_decoder(
            image_embeddings=image_embedding.unsqueeze(0).to(device),
            image_pe=mdl.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    return attention_maps, low_res_masks, iou_pred


def plot_encoder_features(features: dict, original_image: np.ndarray,
                          output_dir: Path, prefix: str = ""):
    """Visualize encoder feature maps (Method A)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features = len(features)
    fig, axes = plt.subplots(1, n_features + 1, figsize=(5 * (n_features + 1), 4.5), dpi=150)

    # Original image
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Input Image", fontsize=10)
    axes[0].axis("off")

    for idx, (name, feat) in enumerate(features.items()):
        # feat shape: [1, H, W, C] for ViT
        if feat.ndim == 4:
            feat_map = feat[0]  # [H, W, C]
        elif feat.ndim == 3:
            feat_map = feat[0]  # [tokens, C]
            h = w = int(feat_map.shape[0] ** 0.5)
            feat_map = feat_map[:h*w].reshape(h, w, -1)

        # PCA to 3 channels for visualization
        feat_flat = feat_map.reshape(-1, feat_map.shape[-1]).numpy()
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        feat_pca = pca.fit_transform(feat_flat)
        feat_pca = feat_pca.reshape(feat_map.shape[0], feat_map.shape[1], 3)

        # Normalize to [0, 1]
        feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min() + 1e-8)

        axes[idx + 1].imshow(feat_pca)
        axes[idx + 1].set_title(f"Encoder {name}\n(PCA→RGB)", fontsize=9)
        axes[idx + 1].axis("off")

    plt.suptitle(f"Encoder Feature Maps {prefix}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / f"encoder_features{prefix}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"  → Saved: {out_path}")
    plt.close(fig)


def plot_attention_comparison(bf_features: dict, mc_features: dict,
                              bf_image: np.ndarray, mc_image: np.ndarray,
                              output_dir: Path):
    """Side-by-side comparison: BF vs multi-channel encoder features (Method A main deliverable)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use last encoder block
    last_key = sorted(bf_features.keys())[-1]
    bf_feat = bf_features[last_key]
    mc_feat = mc_features[last_key]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)

    # Row 0: BF
    axes[0, 0].imshow(bf_image, cmap="gray")
    axes[0, 0].set_title("BF Input", fontsize=10)
    axes[0, 0].axis("off")

    # Row 1: Multi-channel
    if mc_image.ndim == 3 and mc_image.shape[2] == 3:
        axes[1, 0].imshow(mc_image)
    else:
        axes[1, 0].imshow(mc_image, cmap="gray")
    axes[1, 0].set_title("3ch Input (BF+Actn2+DAPI)", fontsize=10)
    axes[1, 0].axis("off")

    # Feature PCA for both
    for row, feat, label in [(0, bf_feat, "BF"), (1, mc_feat, "3ch")]:
        if feat.ndim == 4:
            feat_map = feat[0]
        elif feat.ndim == 3:
            feat_map = feat[0]
            h = w = int(feat_map.shape[0] ** 0.5)
            feat_map = feat_map[:h*w].reshape(h, w, -1)

        feat_flat = feat_map.reshape(-1, feat_map.shape[-1]).numpy()
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        feat_pca = pca.fit_transform(feat_flat)
        feat_pca = feat_pca.reshape(feat_map.shape[0], feat_map.shape[1], 3)
        feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min() + 1e-8)

        axes[row, 1].imshow(feat_pca)
        axes[row, 1].set_title(f"{label} — Encoder Features (PCA)", fontsize=10)
        axes[row, 1].axis("off")

        # Mean activation magnitude
        mean_act = feat_map.numpy().mean(axis=-1)
        mean_act = (mean_act - mean_act.min()) / (mean_act.max() - mean_act.min() + 1e-8)
        axes[row, 2].imshow(mean_act, cmap="hot")
        axes[row, 2].set_title(f"{label} — Mean Activation", fontsize=10)
        axes[row, 2].axis("off")

    plt.suptitle("Encoder Feature Comparison: BF vs 3-Channel", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "encoder_feature_comparison.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"  → Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="T20: CellSAM Attention Visualization")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--image", type=Path, required=True,
                        help="Path to input TIFF image")
    parser.add_argument("--channel", type=int, default=0,
                        help="Channel index for single-channel (BF=0)")
    parser.add_argument("--box", type=str, default=None,
                        help="Bounding box 'x1,y1,x2,y2' (optional, for decoder attention)")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("figures/attention"),
                        help="Output directory")
    parser.add_argument("--method", choices=["A", "C", "AC"], default="AC",
                        help="Visualization method: A=encoder features, C=decoder attention, AC=both")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device)

    print(f"Loading image from {args.image}...")
    image = load_image(args.image, args.channel)

    # Preprocess: normalize to [0, 1], expand to 3ch, resize to 1024
    img_float = image.astype(np.float32)
    img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)
    img_3ch = np.stack([img_float] * 3, axis=0)  # [3, H, W]
    img_tensor = torch.from_numpy(img_3ch).unsqueeze(0).float()  # [1, 3, H, W]
    img_tensor = F.interpolate(img_tensor, size=(1024, 1024), mode="bilinear", align_corners=False)

    if "A" in args.method:
        print("Extracting encoder features (Method A)...")
        features, embeddings = extract_encoder_features(model, img_tensor, args.device)
        plot_encoder_features(features, image, args.output_dir)

    if "C" in args.method and args.box:
        box_coords = [float(x) for x in args.box.split(",")]
        box_tensor = torch.tensor(box_coords)
        print("Extracting decoder attention (Method C)...")
        attn_maps, masks, iou = extract_decoder_attention(
            model, embeddings[0] if 'embeddings' in dir() else None,
            box_tensor, args.device
        )
        print(f"  Decoder attention layers captured: {len(attn_maps)}")
        print(f"  IoU prediction: {iou[0][0].item():.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
