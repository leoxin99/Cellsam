"""
official_preprocess.py — 官方 CellSAM 预处理管线封装

将 CellSAM 的 prep_2() + forward() 的预处理链封装为独立函数，
供 train.py 和 inference/core.py 共用。

严格复刻 prep_2() 的顺序 (R1 审核 #2):
  1. Resize(1024)
  2. sam_preprocess_pad() — 只做 padding
  3. PercentileThreshold()
  4. ImageNet Normalize (adv_mode=True)
  5. Standardize (min-max)
  6. sam_preprocess(div_255=True) — SAM pixel_mean/255 + pixel_std/255

参考:
  - cellSAM_source/cellSAM/sam_inference.py L218-231 (prep_2)
  - cellSAM_source/cellSAM/sam_inference.py L201-216 (forward)
"""
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T


def official_preprocess_and_encode(model, images, device=None):
    """
    使用官方 prep_2() + forward() 链获取图像 embeddings。
    
    直接调用 CellSAM 内部方法，确保预处理顺序与官方完全一致。
    
    Args:
        model: CellSAM 模型实例
        images: [B, C, H, W] float [0,1] 来自 augmented_dataset
        device: 目标设备
    
    Returns:
        image_embeddings: [B, 256, 64, 64] 图像特征
    """
    if device is None:
        device = next(model.parameters()).device
    
    # 1. [0,1] → [0,255] (官方 prep_2 期望 [0,255] 输入)
    images_255 = (images * 255.0).clamp(0, 255)
    
    # 2. 拆成 list (prep_2 期望 list of tensors)
    img_list = [img for img in images_255]
    
    # 3. 调用官方 prep_2: Resize→padding→Percentile→Normalize→Standardize
    transformed, paddings = model.prep_2(img_list, percentile=True)
    # transformed: [B, C, 1024, 1024] float tensor
    
    # 4. 调用官方 forward: sam_preprocess(div_255=True) → model_cp.image_encoder
    #    注: forward() 内部根据 adv_mode 选择 model_cp (True) 或 model (False)
    #    我们确保 adv_mode=True 以使用 model_cp
    embeddings = model.forward(transformed, return_preprocessed=False)
    
    return embeddings


def official_preprocess_only(model, images):
    """
    只做预处理，不编码。用于需要手动控制 encoder 梯度的场景。
    
    Returns:
        preprocessed: [B, C, 1024, 1024] 经过 prep_2 + sam_preprocess(div_255=True)
    """
    device = next(model.parameters()).device
    
    # 1. [0,1] → [0,255]
    images_255 = (images * 255.0).clamp(0, 255)
    img_list = [img for img in images_255]
    
    # 2. prep_2
    transformed, paddings = model.prep_2(img_list, percentile=True)
    
    # 3. sam_preprocess(div_255=True) — 从 forward() 提取
    transformed = transformed.to(device)
    preprocessed_list = [
        model.sam_preprocess(img, return_paddings=False, div_255=True)
        for img in transformed
    ]
    preprocessed = torch.stack(preprocessed_list, dim=0)
    
    return preprocessed
