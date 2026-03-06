# T11: LoRA Encoder Fine-tuning 设计文档

> **状态**: ✅ R1+A1 审核通过 (有条件) — 实施中  
> **创建日期**: 2026-02-25  
> **创建者**: A2 (Claude)  
> **审核**: R1 + A1 (2026-02-25 20:45) — [审核报告](inbox/t11_review_r1a1.md)  
> **优先级**: P1 — 缩小与 MedSAM 差距的主要技术路径  
> **文献依据**: `docs/technical/update_cellsam.md` §9

---

## 1. 动机

### 1.1 当前瓶颈

| 模型 | PQ | BM-Dice | 差距 |
|------|:--:|:-------:|:----:|
| **Ours (Best Config)** | 0.484 | 0.720 | — |
| **MedSAM** | 0.576 | 0.771 | **+9.2pp PQ** |

当前方案 (decoder-only fine-tuning, `freeze_encoder: true`) 的 encoder 仍使用 SAM ViT-B 原始权重，未针对心肌细胞图像做任何适配。文献表明：

| 来源 | 结论 |
|------|------|
| SAMed (ICLR, 2024) | **LoRA on encoder + full decoder → 小数据最优** |
| FSAM (IEEE, 2024) | Encoder+Decoder 微调 > Decoder-only |
| S-SAM (MICCAI, 2024) | SVD tuning encoder (0.4% params) 即超越 decoder-only |

### 1.2 为什么 LoRA 而非全量微调

- **数据量限制**: 仅 334 张训练图 × ~10 cell/image ≈ 3,300 样本
- **过拟合风险**: encoder 有 ~89M 参数，全量微调极易过拟合
- **LoRA 优势**: 仅增加 ~0.17% 参数 (rank=4)，保留预训练特征，学习域特异偏移

### 1.3 预期效果

LoRA 使 encoder 学到心肌细胞的长条形特征 → 两个潜在收益:
1. **直接**: decoder 获得更好的特征 → mask 精度提升 → PQ 上升
2. **间接**: 模型学会"框外抑制" → 减少对 box clipping 的依赖 (参照 `docs/technical/update_cellsam.md` §10.5)

---

## 2. 技术方案

### 2.1 SAM ViT-B Encoder 结构

```
Image Encoder (ViT-B):
├── patch_embed (Conv2d: 3→768, 16x16)
├── pos_embed
├── blocks × 12
│   ├── norm1 (LayerNorm)
│   ├── attn
│   │   ├── qkv (Linear: 768 → 2304)  ← fused Q/K/V
│   │   ├── proj (Linear: 768 → 768)   ← output projection
│   │   ├── rel_pos_h, rel_pos_w       ← relative position
│   │   └── (MultiHeadAttn, 12 heads)
│   ├── norm2 (LayerNorm)
│   └── mlp (Linear → GELU → Linear)
└── neck (Conv2d blocks)
```

### 2.2 LoRA 注入点

**目标层**: 每个 block 的 `attn.qkv` (fused Q/K/V Linear)

```
原始: x → qkv (Linear [768→2304]) → Q,K,V
LoRA: x → qkv (Linear [768→2304]) + [LoRA_Q(x), 0, LoRA_V(x)] → Q,K,V

LoRA_Q = B_q @ A_q  (A_q: [768→r], B_q: [r→768])
LoRA_V = B_v @ A_v  (A_v: [768→r], B_v: [r→768])
```

- **Q 和 V 加 LoRA** (SAMed 策略), K 不加
- `A` 矩阵: Kaiming Uniform 初始化
- `B` 矩阵: Zero 初始化 → 训练开始时 LoRA 输出为 0 (等价无 LoRA)

### 2.3 参数量

| Rank | 每 block params | 12 blocks total | 占 encoder % |
|:----:|:--------------:|:---------------:|:------------:|
| 4 | 12,288 | **147,456** | 0.17% |
| 8 | 24,576 | **294,912** | 0.33% |

加上 decoder (~4M trainable): 总训练参数 ~4.15M (rank=4) 或 ~4.3M (rank=8)。

### 2.4 训练策略

| 参数 | 值 | 依据 |
|------|-----|------|
| Base checkpoint | Best Config (`BestConfig_posw10_noCont_20260224_052553/best_model.pt`) | 当前最优 decoder |
| freeze_encoder | true (base weights) | LoRA params 单独 trainable |
| freeze_decoder | false | 与 Best Config 一致 |
| use_lora | true | 新增 |
| lora_rank | 4 (主实验) / 8 (对比) | SAMed 推荐 rank=4 |
| epochs | 80 | 与 Best Config 一致 |
| lr | 1e-4 | 与 Best Config 一致，降 lr 作为消融点 |
| early_stop | PQ, patience=15 | 与 Best Config 一致 |
| 通道 | BF-only (先隔离变量) | 排除三通道交互影响 |
| Seeds | 42 + 123 | 与 T12/T18 一致 |

---

## 3. 实现清单

### 3.1 新建文件

| 文件 | 内容 |
|------|------|
| `src/lora.py` | `LoRALinear` 类 + `apply_lora_to_encoder()` + `remove_lora_and_merge()` |
| `src/config/t11_lora_r4.yaml` | rank=4, BF-only, from Best Config |
| `src/config/t11_lora_r8.yaml` | rank=8, BF-only, from Best Config |
| `scripts/train_t11_lora.sh` | SLURM 脚本 (L4 + A100) |

### 3.2 修改文件

| 文件 | 改动 |
|------|------|
| `src/train.py` `create_model()` | 新增 ~15 行: 读取 `use_lora`/`lora_rank` → 调用 `apply_lora_to_encoder()` |
| `src/train.py` `train_one_epoch()` | **⛔ P0-1 修复**: LoRA 时移除 `torch.no_grad()` 包裹 encoder forward (~8 行) |
| `src/train.py` checkpoint saving | 新增 ~5 行: 额外保存 LoRA 元信息 |
| `src/inference/core.py` `load_cellsam_checkpoint()` | **⛔ P0-2 修复**: 检测 LoRA keys → 先 apply_lora → 再 load (~15 行) |

### 3.3 审核发现的关键修复 (⛔ P0 阻塞项)

> 来源: `docs/inbox/t11_review_r1a1.md` (R1 + A1 综合审核)

#### P0-1: `torch.no_grad()` 切断 LoRA 梯度

`train.py:232` 的 `with torch.no_grad()` 包裹 encoder forward。LoRA 参数虽 `requires_grad=True`，但 no_grad 上下文阻止 autograd 追踪 → LoRA 零梯度。

```diff
-# 当前: encoder forward 被 no_grad 包裹
-with torch.no_grad():
-    img_preprocessed = model.sam_preprocess(images)
-    image_embedding = model.model.image_encoder(img_preprocessed)
+# 修复: LoRA 时 encoder 需要梯度流
+use_lora = config['model'].get('use_lora', False)
+if use_lora:
+    img_preprocessed = model.sam_preprocess(images)
+    image_embedding = model.model.image_encoder(img_preprocessed)
+else:
+    with torch.no_grad():
+        img_preprocessed = model.sam_preprocess(images)
+        image_embedding = model.model.image_encoder(img_preprocessed)
```

#### P0-2: 推理加载不支持 LoRA

`inference/core.py` `load_cellsam_checkpoint()` 使用 `strict=False` 加载 state_dict，LoRA keys 被静默丢弃 → 评估用原始 SAM 权重。

```diff
+# 修复: 检测 LoRA keys → 先注入 LoRA 层 → 再加载权重
+state_dict = checkpoint['model_state_dict']
+has_lora = any('lora_' in k for k in state_dict.keys())
+if has_lora:
+    from lora import apply_lora_to_encoder
+    lora_rank = checkpoint.get('config', {}).get('model', {}).get('lora_rank', 4)
+    apply_lora_to_encoder(model.model.image_encoder, rank=lora_rank)
+model.load_state_dict(state_dict, strict=False)
```

#### P1-1: freeze_encoder 执行顺序

`create_model()` 中 `freeze_encoder` 在 LoRA 应用之前执行 → LoRA params 也被冻结。

**修复**: 调整执行顺序为 **先 freeze → 后 apply LoRA**。`apply_lora_to_encoder()` 内部会设置 LoRA params `requires_grad=True`，不受之前的 freeze 影响。

#### P1-2: eval 脚本 CLI 支持

~~`comprehensive_eval.py`~~ **已归档**。T11 使用 `eval_ablation.py --exp-dir` (已内置于 SLURM 脚本)。

### 3.4 `src/lora.py` 核心设计

```python
import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """LoRA decomposition: output += B(A(x)), where A down-projects and B up-projects."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.rank = rank
        # A: down-projection [in→rank], Kaiming init (R1-F3: use nn.Linear)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # B: up-projection [rank→out], zero init → starts as identity
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        return self.lora_B(self.lora_A(x))


class LoRAQKVLinear(nn.Module):
    """Wraps SAM's fused QKV Linear with LoRA on Q and V slices.
    
    SAM's qkv weight: [2304, 768] = [Q(768)|K(768)|V(768), 768]
    LoRA adds to Q and V output slices only.
    """
    
    def __init__(self, original_linear: nn.Linear, rank: int = 4):
        super().__init__()
        self.original = original_linear
        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        dim = original_linear.in_features  # 768
        self.lora_q = LoRALinear(dim, dim, rank)  # Q slice
        self.lora_v = LoRALinear(dim, dim, rank)  # V slice
    
    def forward(self, x):
        # Original fused QKV output
        qkv = self.original(x)  # [B, N, 2304]
        
        # Add LoRA to Q (first 768) and V (last 768)
        dim = self.original.in_features  # 768
        qkv[..., :dim] += self.lora_q(x)        # Q slice
        qkv[..., 2*dim:] += self.lora_v(x)      # V slice
        # K slice (middle 768) unchanged
        
        return qkv


def apply_lora_to_encoder(encoder, rank: int = 4):
    """Apply LoRA to all attention QKV layers in SAM ViT-B encoder."""
    lora_count = 0
    for block in encoder.blocks:
        original_qkv = block.attn.qkv
        block.attn.qkv = LoRAQKVLinear(original_qkv, rank=rank)
        lora_count += 1
    
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Applied LoRA (rank={rank}) to {lora_count} attention blocks")
    print(f"  LoRA trainable params: {n_params:,}")
    return encoder


def get_lora_state_dict(model):
    """Extract only LoRA parameters from model state dict."""
    return {k: v for k, v in model.state_dict().items() if 'lora_' in k}
```

---

## 4. 实验计划

### 4.1 实验矩阵

| 实验 | Rank | 通道 | Seed | GPU | 说明 |
|------|:----:|:----:|:----:|-----|------|
| T11-r4-s42 | 4 | BF | 42 | L4 | 主实验 |
| T11-r4-s123 | 4 | BF | 123 | A100 | seed 交叉 |
| T11-r8-s42 | 8 | BF | 42 | L4 | rank 对比 |
| T11-r8-s123 | 8 | BF | 123 | A100 | rank 对比 |

### 4.2 评估

- **评估集**: test(73)
- **评估方式**: Oracle (GT box), `tools/eval_ablation.py --exp-dir`
- **指标**: PQ, BM-Dice, AJI, Sem.Dice

### 4.3 成功标准

| 级别 | 条件 | 含义 |
|------|------|------|
| ✅ 成功 | PQ > 0.500 | 超过 T18-C (当前 best) |
| ⚠️ 部分成功 | PQ 0.490~0.500 | 与 T18-C 持平 |
| ❌ 失败 | PQ < 0.484 | 低于 Best Config (decoder-only) |

### 4.4 后续 (T11 成功后)

- T11 + T18-C 结合: LoRA encoder + 三通道 → 可能叠加增益
- 更高 rank (16/32) 或 Conv-LoRA 探索
- 论文叙事: "LoRA encoder fine-tuning closes X% of gap to MedSAM"

---

## 5. 设计决策 (R1 已审批)

| # | 问题 | A2 建议 | R1 决策 |
|---|------|---------|:------:|
| 1 | 先 BF-only 还是 3ch? | BF-only (隔离变量) | **✅ 同意** |
| 2 | rank=4 先跑还是 4+8 一起? | 一起跑 (节省墙钟) | **✅ 同意** |
| 3 | lr 保持 1e-4 还是降? | 先 1e-4 | **✅ + 预备 5e-5 fallback** |
| 4 | Seeds 数量? | 2 seeds (42+123) | **✅ 同意** |

---

## 6. 验收标准 (来自审核报告)

- [ ] `src/lora.py` smoke test: 加载模型 → apply LoRA → forward → 无报错
- [ ] LoRA 梯度流验证: 训练 1 epoch 后 LoRA params 有非零 gradient
- [ ] freeze 不影响 LoRA: `freeze_encoder=true` 后 LoRA params 仍 `requires_grad=True`
- [ ] Checkpoint save/load round-trip: 保存后重新加载，输出一致
- [ ] SLURM 脚本使用绝对 checkpoint 路径

---

## 更新日志

| 日期 | 内容 |
|------|------|
| 2026-02-25 19:20 | A2: 初版创建, 提交 R1 审核 |
| 2026-02-25 20:45 | R1+A1: 审核通过 (有条件), 发现 2 P0 + 2 P1 + 1 P2 |
| 2026-02-25 20:55 | A2: 整合审核反馈, 更新 §3.3 修复方案 + §3.4 代码修正 + §5 决策 + §6 验收标准 |

