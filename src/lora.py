# -*- coding: utf-8 -*-
"""
LoRA (Low-Rank Adaptation) for SAM ViT-B Encoder.

Applies LoRA to Q/V projections of fused QKV attention in each transformer block.
SAMed (ICLR 2024) validated approach: LoRA on encoder Q/V + full decoder fine-tuning.

Design doc: docs/t11_lora_design.md
Review: docs/inbox/t11_review_r1a1.md

Usage:
    from lora import apply_lora_to_encoder, get_lora_state_dict
    apply_lora_to_encoder(model.model_cp.image_encoder, rank=4)
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA decomposition: output += B(A(x)).
    
    A: [in_features → rank]  (Kaiming init)
    B: [rank → out_features] (zero init → starts as identity = no change)
    
    Uses nn.Linear(bias=False) per R1-F3 review (consistent with SAMed).
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        
        # A: down-projection
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        
        # B: up-projection (zero init → LoRA output starts at 0)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        return self.lora_B(self.lora_A(x))


class LoRAQKVLinear(nn.Module):
    """Wraps SAM's fused QKV Linear with LoRA on Q and V slices.
    
    SAM's qkv weight: [2304, 768] = concat([Q(768), K(768), V(768)], dim=0)
    LoRA adds low-rank updates to Q and V output slices only (K unchanged).
    
    Forward:
        qkv_out = original_qkv(x)                    # [B, N, 2304]
        qkv_out[..., :768]   += lora_q(x)            # Q slice
        qkv_out[..., 1536:]  += lora_v(x)            # V slice
    """
    
    def __init__(self, original_linear: nn.Linear, rank: int = 4):
        super().__init__()
        self.original = original_linear
        
        # Freeze original weights (P1-1: LoRA init sets requires_grad=True on its own params)
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        dim = original_linear.in_features  # 768 for ViT-B
        self.lora_q = LoRALinear(dim, dim, rank)
        self.lora_v = LoRALinear(dim, dim, rank)
    
    def forward(self, x):
        # Original fused QKV output (frozen)
        qkv = self.original(x)  # [B, N, 3*dim]
        
        # Add LoRA to Q (first dim) and V (last dim)
        dim = self.original.in_features
        qkv = qkv.clone()  # Avoid in-place on frozen output
        qkv[..., :dim] = qkv[..., :dim] + self.lora_q(x)
        qkv[..., 2*dim:] = qkv[..., 2*dim:] + self.lora_v(x)
        
        return qkv


def apply_lora_to_encoder(encoder, rank: int = 4, use_grad_checkpoint: bool = True):
    """Apply LoRA to all attention QKV layers in SAM ViT-B encoder.
    
    Must be called AFTER freeze_encoder (P1-1 fix):
        1. freeze_encoder sets all encoder params requires_grad=False
        2. apply_lora creates NEW LoRA params with requires_grad=True
        → LoRA params are trainable, base weights stay frozen
    
    Args:
        encoder: SAM ViT-B image_encoder (model.model_cp.image_encoder)
        rank: LoRA rank (4 or 8)
        use_grad_checkpoint: Enable gradient checkpointing to reduce VRAM
            (~38GB → ~3GB for encoder activations). Trades ~30% speed for memory.
    
    Returns:
        encoder with LoRA applied (modified in-place)
    """
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
    
    lora_count = 0
    for block in encoder.blocks:
        original_qkv = block.attn.qkv
        block.attn.qkv = LoRAQKVLinear(original_qkv, rank=rank)
        lora_count += 1
        
        # Fix-1: Gradient checkpointing — wrap each block's forward
        # This drops intermediate activations and recomputes during backward,
        # reducing VRAM from ~38GB to ~3GB (only block boundary tensors kept)
        if use_grad_checkpoint:
            original_forward = block.forward
            # Use closure to capture the correct reference
            def _make_ckpt_forward(fn):
                def ckpt_forward(*args, **kwargs):
                    return torch_checkpoint(fn, *args, use_reentrant=False, **kwargs)
                return ckpt_forward
            block.forward = _make_ckpt_forward(original_forward)
    
    lora_params = [p for p in encoder.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in lora_params)
    print(f"Applied LoRA (rank={rank}) to {lora_count} attention blocks")
    print(f"  LoRA trainable params: {n_lora:,}")
    if use_grad_checkpoint:
        print(f"  Gradient checkpointing: ENABLED (VRAM ~38GB → ~3GB)")
    return encoder


def get_lora_state_dict(model):
    """Extract only LoRA parameters from model state dict.
    
    Useful for saving lightweight LoRA-only checkpoints.
    """
    return {k: v for k, v in model.state_dict().items() if 'lora_' in k}


def has_lora_keys(state_dict: dict) -> bool:
    """Check if a state dict contains LoRA parameters."""
    return any('lora_' in k for k in state_dict.keys())
