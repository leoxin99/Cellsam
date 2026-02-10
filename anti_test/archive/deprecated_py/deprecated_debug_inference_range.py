# [DEPRECATED] This script has been archived.
#
# Archived: 2026-02-10
# Reason: Superseded by unified inference core (Phase 0)
# Replacement entry points:
#   - Training:           src/train.py
#   - Oracle evaluation:  tools/standardized_inference.py
#   - E2E evaluation:     tools/evaluate_e2e.py
#   - Multi-model eval:   tools/comprehensive_eval.py
#   - Regression test:    tools/test_phase0_regression.py
#
import warnings as _warnings
_warnings.warn(
    "This script is deprecated. See header for replacement entry points.",
    DeprecationWarning, stacklevel=2
)
"""
Debug script to determine if model expects 0-1 or 0-255 input range.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from skimage import transform as skt

sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
from cellSAM.model import get_model

MODEL_PATH = "d:/AI/paper/CellSam/checkpoints/expanded_20260108_034352/best_model.pt"
SAMPLE_IMG = "d:/AI/paper/CellSam/data/processed/images/006167ed_5500000013_63X_20190807_S2_P8_C4.npy"
SAMPLE_MASK = "d:/AI/paper/CellSam/data/processed/masks/006167ed_5500000013_63X_20190807_S2_P8_C4.npy"

def test_inference(model, device, img_tensor, boxes, description):
    print(f"\n--- Testing {description} ---")
    print(f"Input range: {img_tensor.min().item():.3f} - {img_tensor.max().item():.3f}")
    
    with torch.no_grad():
        img_preprocessed = model.sam_preprocess(img_tensor)
        print(f"Preprocessed range: {img_preprocessed.min().item():.3f} - {img_preprocessed.max().item():.3f}")
        
        embedding = model.model.image_encoder(img_preprocessed)
        
        box_tensor = torch.tensor(boxes, dtype=torch.float32).to(device)
        
        sparse, dense = model.model.prompt_encoder(points=None, boxes=box_tensor, masks=None)
        
        low_res, _ = model.model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False
        )
        
        probs = torch.sigmoid(low_res).cpu().numpy()
        print(f"Output probability statistics: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
        print(f"Pixels > 0.5: {(probs > 0.5).sum()}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    
    # Load sample
    img = np.load(SAMPLE_IMG) # uint8 0-255
    mask = np.load(SAMPLE_MASK)
    
    # Get a dummy box from mask
    from skimage.measure import regionprops
    props = regionprops(mask.astype(np.int32))
    if not props:
        print("No cells in mask")
        return
    
    # Resize to 1024 like pipeline
    img_resized = skt.resize(img, (1024, 1024), preserve_range=True)
    
    # Scale box
    scale_y = 1024 / img.shape[0]
    scale_x = 1024 / img.shape[1]
    y1, x1, y2, x2 = props[0].bbox
    box = [[x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y]] # SAM format x1, y1, x2, y2
    
    # Test 1: Range 0-255
    img_255 = img_resized.astype(np.float32) # 0-255
    img_tensor_255 = torch.from_numpy(np.stack([img_255]*3)).float().unsqueeze(0).to(device)
    test_inference(model, device, img_tensor_255, box, "Range 0-255")
    
    # Test 2: Range 0-1
    img_01 = img_resized.astype(np.float32) / 255.0
    img_tensor_01 = torch.from_numpy(np.stack([img_01]*3)).float().unsqueeze(0).to(device)
    test_inference(model, device, img_tensor_01, box, "Range 0-1")

if __name__ == "__main__":
    main()
