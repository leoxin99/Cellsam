# 会议纪要 — 2026-02-19 (曹老师 × 学生)

> **时间**: 2026-02-19，约 51 分钟
> **形式**: 线上会议

---

## 一、论文写作指导

### 1.1 Loss 设计的 motivation (必须)
- **必须讲清**: 原来的 loss 是什么 → 为什么不好 → 改变的 motivation
- 加了哪些 loss、怎么组合、怎么平衡、来源 (reference) 是什么

### 1.2 Semantic Dice 假象
- **❌ 不用写进论文** — 导师明确表示不需要
- CellSAM 本身就是 instance segmentation，semantic target 不合理

### 1.3 N/O Loss 退化 (可以写)
- ✅ **放进论文**: 做了实验、发现退化、做了诊断 (微调/延迟介入/权重递减) → 说明此路不通
- 导师评价"挺有意思的"
- **不需要深入量化冲突原因**，放到后面做/future work 即可

### 1.4 Training curves (必须)
- **必须提供**: epochs vs loss/accuracy 图 (training + validation)
- 类似 "accuracy, training loss, validation loss" 的标准曲线
- 目的: 展示 overfitting/underfitting 情况、performance 如何提升

### 1.5 Grad-CAM / 可解释性 (可选)
- 导师建议: "用 Grad-CAM 把权重 project 到 image 上，看模型 focus 哪些区域"
- 可作为 "为什么多通道有效/无效" 的证据补强
- **前提**: "得看你有没有时间"，优先级低于核心实验

---

## 二、实验优先级 (导师排序)

导师指定了明确的优先级顺序：

| 优先级 | 任务 | 状态 | 备注 |
|:--:|------|:--:|------|
| **1** 🔴 | **Baseline 对比** (Cellpose, StarDist, MedSAM) | 模型已下载 | "非常重要，一定要对比" |
| **2** 🔴 | **Training curves 图** | 数据已有 | 标准论文要求 |
| **3** | **三通道 Decoder 实验** (BF+Actn2, BF+DAPI+Actn2) | 需执行 | 用 Phase 1 最优配置，只改通道数 |
| **4** | **框外像素分割策略优化** | 需调研 | 新发现的方向 (详见下方) |
| **5** | **LoRA Encoder 微调** | 需执行 | "时间有就做，没有也 OK" |

### 关键决定:
- **三通道排在 LoRA 前面** — 导师明确要求调换顺序
- LoRA/Encoder 实验开始时 → **同时开始写论文**
- N/O Loss 冲突量化 → **推迟或 future work**
- **框外像素策略排在 LoRA 前面** — 导师: "你甚至可以把它放到 encode 前面"

---

## 三、三通道实验要求 (重点 — 必须做)

导师认为这是**必须做的**，否则 "不太有信服力"：

1. **已有信息**: 数据集提供 BF + DAPI + Actn2 三通道
2. **当前状态**: 分割只用 BF 单通道 (DAPI 和 Actn2 仅用于检测阶段)
3. **导师要求**:
   - 用 Phase 1 最优 loss 配置 (不变)
   - 只改通道输入: BF → BF+Actn2 (2ch) → BF+DAPI+Actn2 (3ch)
   - 都只做 Decoder fine-tuning
   - 比较结果即可

**导师理由**: "你不可能说我只用了一个通道，别人会问 Actin 放进去会不会好？数据提供了三通道，你要充分利用"

**导师补充**: encoder 虽然没微调，但三通道输入信息更丰富 → encoder 产生的 feature 也更丰富 → 有可能帮助 decoder 找边界

---

## 四、新发现：框外像素策略 (导师很感兴趣)

讨论中发现 CellSAM **会考虑框外像素**（不只在框内分割）：

- SAM 对每个 cell 生成整张图的预测（框外也有，但置信度较低）
- 多个 cell 的预测叠加后，通过推理阶段的冲突裁决（置信度比较）分配像素
- GT 框效果好很多 → 说明虽然框外有预测，但主要还是靠框内

**导师观点**: 
- "哇，那很有意思！" / "这个是会是一个很好的亮点"
- "我们框已经非常 plausible 了…可以在 segmentation 这块做调整"
- "你甚至可以把它放到 encode 前面"
- **建议排在 LoRA 前面做**

**潜在研究思路**: 调整框内/框外的权重比例，让模型更多考虑框外上下文

---

## 五、论文发表讨论

- 导师倾向于发 **conference paper** (一次审稿)，而非 journal (多轮修改)
- 取决于最终实验结果
- 导师希望好的工作能被 community 看到
- 学生也倾向 conference

---

## 六、时间线讨论

| 事项 | 详情 |
|------|------|
| 项目开始 | 2025 年 10 月 (联系导师) |
| 学分 | 42 EC ≈ 1176 小时 (practical + research ≈ 952h) |
| 导师建议 | Master thesis 标准约 9 个月 → 理论上到 2026 年 7 月 |
| 学生希望 | 尽快毕业 (3~4 月) |
| **达成一致** | 先全力做实验，**4~5 月答辩**比较合理；若工时算够可尝试提前 |
| 下次会议 | **2 周后** (有结果可先发邮件) |
| 面谈 | 学生已回荷兰，**周四/五**可当面找导师看结果 |

---

## 七、Action Items (按优先级)

### P0 必做 (直接影响论文可信度)
- [ ] 🔴 **Baseline 对比** — Cellpose / StarDist / MedSAM / SAMCell 在 test(73) 统一评估
- [ ] 🔴 **Training curves 图** — epochs vs loss/PQ (训练集 + 验证集), 展示收敛过程
- [ ] **三通道实验** — BF / BF+Actn2 / BF+DAPI+Actn2，用 Phase 1 最优配置训练

### P1 重要 (有则加分)
- [ ] **框外像素策略调研** — 了解 CellSAM 如何处理框外预测，调整权重
- [ ] **Grad-CAM 可视化** — 展示模型关注区域，补强多通道实验解释
- [ ] **LoRA Encoder** — 做一个简单的 rank=4~8 测试

### P2 可选
- [ ] N/O Loss 深度量化 → future work
- [ ] Encoder 系统性扫参 → future work

### 论文写作
- [ ] 开始搭论文框架（与 LoRA 实验并行）
- [ ] Loss motivation + 技术原理先写
- [ ] 有结果发邮件给导师
