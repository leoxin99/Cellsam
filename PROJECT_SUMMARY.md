# 心肌细胞分割项目 - 协作交接文档

**创建日期**: 2025-12-27  
**项目位置**: `d:\AI\paper\CellSam\`  
**学生账号**: s3890074 (莱顿大学)

---

## 一、项目目标

### 研究问题
使用 **CellSAM** 模型对**心肌细胞 (Cardiomyocytes)** 进行准确分割。

### 核心挑战
- 心肌细胞形态**不规则**（细长、非圆形）
- 具有**肌节条纹**结构，与普通细胞不同
- 通用分割模型效果不佳，需要**微调 (Fine-tuning)**

### 预期成果
1. 在 Allen hiPSC-CM 数据集上微调 CellSAM
2. 实现高精度心肌细胞自动分割
3. 完成相关学术报告/论文

---

## 二、技术方案

### 模型架构
**CellSAM** = SAM (Segment Anything Model) + 细胞图像优化

```
输入图像 → ViT Backbone → CellFinder (检测) → Mask Decoder → 分割掩膜
                 ↑(冻结)              ↓(提供框)        ↑(训练)
```

### 微调策略
- **冻结**: ViT Image Encoder (预训练权重保留)
- **训练**: Mask Decoder (适应心肌细胞)
- **Prompt**: 使用 DAPI 通道生成细胞核边界框

### 数据来源
**Allen Institute for Cell Science - hiPSC-CM Dataset**
- 格式: 3D OME-TIFF (50 层 Z-stack, 1736×1776 像素)
- 通道: Brightfield (明场), GFP (Alpha-actinin-2), DAPI (细胞核)
- 来源: AWS S3 公开数据集

---

## 三、当前成果

### ✅ 已完成

| 类别 | 成果 |
|-----|------|
| **环境配置** | Conda + PyTorch + CellSAM (开发者模式) |
| **数据获取** | 下载 5 个标注样本 + 3 个原始 Brightfield 图像 |
| **预处理** | 3D→2D Sum Projection 脚本完成并测试 |
| **模型验证** | CellSAM 推理成功，检测到 14 个细胞 |
| **训练脚本** | 完整的微调训练流水线 |
| **可视化工具** | Napari 查看器 + 分割叠加显示 |

### 📁 核心文件清单

```
d:\AI\paper\CellSam\
├── allen_brightfield/          # 原始 3D TIFF 数据
├── allen_brightfield_processed/ # 预处理后 2D NPY 文件
├── cellSAM_source/             # CellSAM 源码 (可编辑)
│
├── preprocess_allen_data.py    # 3D→2D 预处理
├── generate_prompts_from_dapi.py # DAPI 生成边界框
├── cardiomyocyte_dataset.py    # PyTorch Dataset
├── train_cardiomyocyte.py      # 微调训练脚本
├── view_with_cellsam.py        # Napari 可视化
│
└── download_*.py               # 数据下载工具
```

---

## 四、当前障碍

### 🔴 主要问题：ALICE HPC 访问受阻

| 问题 | 状态 |
|-----|------|
| eduVPN 连接 | ✅ 已解决 (手动安装 Wintun) |
| SSH 连接 ALICE | ❌ 账号可能未激活 |
| 错误信息 | `Connection closed by remote host` |

**原因分析**：
- 服务器在 SSH 密钥交换前拒绝连接
- 不是密码问题，而是账号权限问题
- 需要联系 helpdesk-alice@science.leidenuniv.nl 确认账号状态

### 🟡 待解决事项

1. **ALICE 账号激活** - 最高优先级
2. **Ground Truth 掩膜获取** - 需要下载完整标注数据
3. **模型训练** - 等待服务器就绪或先本地小规模测试

---

## 五、技术细节

### 数据预处理流程

```bash
# 1. 下载原始数据
python download_brightfield.py

# 2. 3D→2D 投影
python preprocess_allen_data.py \
    --input_dir allen_brightfield \
    --output_dir allen_brightfield_processed \
    --projection sum  # Sum Projection 保留肌节纹理

# 3. 生成 DAPI 提示框
python generate_prompts_from_dapi.py \
    --input_dir allen_brightfield_processed \
    --output_dir allen_brightfield_processed/prompts \
    --pattern "*dapi*.npy"
```

### 训练命令

```bash
python train_cardiomyocyte.py \
    --data_dir processed_data \
    --epochs 50 \
    --batch_size 2 \
    --lr 1e-4 \
    --freeze_encoder  # 冻结 ViT
```

### 关键参数

| 参数 | 值 | 说明 |
|-----|---|------|
| 投影方法 | Sum Projection | 保留肌节纹理完整性 |
| 目标尺寸 | 1024×1024 | CellSAM 输入要求 |
| bbox_threshold | 0.3 | 检测阈值 (越低检测越多) |
| Learning Rate | 1e-4 | 微调推荐值 |

---

## 六、下一步任务建议

### 紧急任务

1. **解决 ALICE 访问**
   - 发邮件给 helpdesk-alice@science.leidenuniv.nl
   - 或尝试网页版访问 https://ondemand.alice.leidenuniv.nl

2. **下载完整数据集**
   - 需要更多样本 (建议 50-100 个)
   - 包含 Ground Truth 分割掩膜

### 中期任务

3. **本地调试训练**
   - 用现有 3 个样本测试训练脚本
   - 验证损失函数正常下降

4. **论文阅读**
   - CellSAM 原论文：理解模型细节
   - SAM 原论文：理解 Prompt Engineering

5. **实验设计**
   - 定义评估指标 (IoU, Dice Score)
   - 设计对比实验方案

### 长期任务

6. **模型优化**
   - 尝试不同 Loss 权重组合
   - 数据增强策略

7. **论文撰写**
   - Method: 描述微调策略
   - Results: 定量和定性结果
   - Discussion: 对比其他方法

---

## 七、协作建议

### 推荐 AI 工具分工

| AI 助手 | 擅长领域 | 推荐任务 |
|---------|---------|---------|
| **Antigravity** | 本地文件操作、终端命令 | 代码调试、环境配置、数据处理 |
| **Claude** | 长文档分析、学术写作 | 论文阅读、文献综述、方法设计 |
| **Gemini** | 多模态理解、中文支持 | 图像分析、中文报告、多语言交流 |

### 如何使用本文档

**复制以下内容发送给 Claude/Gemini**：

```
我在做一个心肌细胞分割项目，使用 CellSAM 模型。
以下是项目完整背景：

[粘贴本文档内容]

我现在需要帮助：
1. [具体描述您的需求，如：分析 CellSAM 论文、设计实验方案等]
```

---

## 八、重要链接

| 资源 | URL |
|-----|-----|
| CellSAM GitHub | https://github.com/vanvalenlab/cellSAM |
| Allen Cell 数据集 | https://open.quiltdata.com/b/allencell |
| ALICE Wiki | https://pubappslu.atlassian.net/wiki/spaces/HPCWIKI/ |
| Napari 文档 | https://napari.org/ |

---

## 九、常见问题

### Q: 为什么用 Sum Projection 而不是 Max Projection？
A: 心肌细胞的肌节条纹分布在多个 Z 层，Sum Projection 能累加所有层的信号，保留纹理完整性；Max Projection 只保留最亮的层，容易断裂。

### Q: 数据量需要多少？
A: 建议至少 50-100 个有标注的样本进行微调。可以使用伪标签扩展数据集。

### Q: 本地训练还是服务器训练？
A: 本地可以做小规模测试（3-5 个样本），完整训练需要 GPU 服务器（ALICE HPC）。

---

**文档生成时间**: 2025-12-27 20:42  
**状态**: 等待 ALICE 账号激活，同时可进行论文阅读和实验设计
