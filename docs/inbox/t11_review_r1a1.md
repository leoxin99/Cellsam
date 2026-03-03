# T11 LoRA Design — R1 + A1 综合审核报告

> **审核人**: R1 (Reviewer) + A1 (Codex)  
> **日期**: 2026-02-25 20:00  
> **审核对象**: `docs/t11_lora_design.md`  
> **依据**: SAM 官方源码, SAMed 官方实现, 项目 `train.py` / `inference/core.py`

---

## 综合结论: ✅ 方案正确, ⛔ 2 个阻塞项必须先修

LoRA on Q/V of fused QKV 是 SAMed 验证过的标准做法，方向正确。但当前 `train.py` 和 `inference/core.py` 对 LoRA **不兼容**，不修就训不了/评不了。

---

## 所有发现 (按严重度排序)

### ⛔ P0-1: `torch.no_grad()` 切断 LoRA 梯度 (A1 发现)

**位置**: `src/train.py:232-239`

```python
# 当前代码:
with torch.no_grad():                          # ← 梯度全断!
    img_preprocessed = model.sam_preprocess(images)
    image_embedding = model.model.image_encoder(img_preprocessed)  # LoRA 在这里
```

**问题**: LoRA 注入到 `image_encoder.blocks[*].attn.qkv`，但整个 encoder forward 被 `no_grad()` 包裹。LoRA 参数虽然 `requires_grad=True`，但 `no_grad` 上下文阻止 autograd 追踪 → LoRA 零梯度 → 等于没加。

**修复**:
```python
if use_lora:
    img_preprocessed = model.sam_preprocess(images)
    image_embedding = model.model.image_encoder(img_preprocessed)
else:
    with torch.no_grad():
        img_preprocessed = model.sam_preprocess(images)
        image_embedding = model.model.image_encoder(img_preprocessed)
```

---

### ⛔ P0-2: 推理加载不支持 LoRA (A1 发现)

**位置**: `src/inference/core.py:90-99`

```python
model = get_model()  # 原始 SAM, 无 LoRA 层
model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # LoRA keys 被静默丢弃!
```

**问题**: `strict=False` 允许 key 不匹配。如果 checkpoint 含 LoRA keys 但模型无 LoRA 层，权重被静默忽略 → 评估用原始 SAM 权重。

**修复**: `load_cellsam_checkpoint` 需检测 checkpoint 是否含 LoRA keys → 若有, 先 `apply_lora_to_encoder` → 再 load。

---

### 🟡 P1-1: freeze_encoder 会冻结 LoRA 参数 (R1 + A1 均发现)

**位置**: `src/train.py:144-147`

```python
if config['model']['freeze_encoder']:
    for param in model.model.image_encoder.parameters():
        param.requires_grad = False  # ← LoRA params 也在 image_encoder 下!
```

**修复**: 执行顺序 = 先 freeze 全部 → 再 apply LoRA (LoRA init 会设 requires_grad=True)。或 freeze 时排除 `lora_` 参数。

---

### 🟡 P1-2: eval 脚本硬编码 checkpoint (A1 发现)

**位置**: `tools/comprehensive_eval.py:32`

T11 有 4 个配置 (r4/r8 × s42/s123), 手工改硬编码容易出错。需确认有 CLI 参数支持。

---

### 🟢 P2: 文档/代码风格小问题 (R1 发现)

| ID | 内容 |
|----|------|
| R1-F1 | §2.1 写 ViT-H 630M → 实际 ViT-B ~89M (标注错误) |
| R1-F3 | `LoRALinear` 建议用 `nn.Linear(bias=False)` 替代 `nn.Parameter` (与 SAMed 一致) |
| R1-F4 | 集成行数低估 (~20→~50 行)，不影响可行性 |

---

## 设计决策审批 (R1)

| # | 问题 | A2 建议 | 决策 |
|---|------|---------|:----:|
| 1 | 先 BF-only 还是 3ch? | BF-only | **✅ 同意** |
| 2 | rank=4 先跑还是 4+8 一起? | 一起跑 | **✅ 同意** |
| 3 | lr 保持 1e-4 还是降? | 先 1e-4 | **✅ + 预备 5e-5 fallback config** |
| 4 | Seeds 数量? | 2 seeds | **✅ 同意** |

---

## 补充: Box Clipping 策略

T11 **保持 box clipping 不变** (与 Best Config 一致, `apply_box_clipping: true`)。

LoRA 的长远价值在于: encoder 学到心肌长条特征后，可能减少框外泄漏 → 未来可尝试关闭 clipping。但 T11 Phase 1 严格隔离变量，不改 clipping。

---

## 修复优先级

| 优先级 | 修复 | 改哪里 | ~行数 |
|:------:|------|--------|:----:|
| **P0** | LoRA 时移除 `no_grad` | `train.py` L232 | ~8 |
| **P0** | 推理 LoRA 注入+加载 | `inference/core.py` L90 | ~15 |
| **P1** | freeze/LoRA 顺序控制 | `train.py` `create_model` | ~10 |
| **P1** | eval checkpoint CLI | `comprehensive_eval.py` | ~5 |
| **P2** | 文档修正 + 代码风格 | 设计文档 + `lora.py` | ~10 |

**所有修复应作为 T11 实现的一部分一并完成。**

---

## 验收标准

- [ ] `src/lora.py` smoke test: 加载模型 → apply LoRA → forward → 无报错
- [ ] LoRA 梯度流验证: 训练 1 epoch 后 LoRA params 应有非零 gradient
- [ ] freeze 不影响 LoRA: `freeze_encoder=true` 后 LoRA params 仍 `requires_grad=True`
- [ ] Checkpoint save/load round-trip: 保存后重新加载，输出一致
- [ ] SLURM 脚本使用绝对 checkpoint 路径
