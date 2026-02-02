# CellSAM 项目清理方案

> **创建日期**: 2026-01-30
> **状态**: 待审批
> **预计耗时**: 2-3 小时

---

## 一、问题概览

| 类别 | 问题数量 | 严重程度 |
|------|---------|---------|
| 根目录散落脚本 | 13 个 | 中 |
| 代码重复 | 3 组 | 高 |
| 目录命名不规范 | 2 个 | 低 |
| 垃圾/临时文件 | 2 个 | 低（已清理） |

---

## 二、根目录 Python 脚本处理方案

### 2.1 当前状态

根目录存在 13 个 Python 脚本，造成项目结构混乱：

| 脚本 | 行数 | 创建阶段 | 当前状态 |
|------|------|---------|---------|
| `verify_env.py` | 23 | - | 一次性验证脚本 |
| `test_loss_fn.py` | 24 | E09 | 过时测试脚本 |
| `debug_class_imbalance.py` | 55 | E01 | 过时调试脚本 |
| `verify_cell_matching.py` | 63 | E01 | 过时验证脚本 |
| `run_cellsam.py` | 86 | - | 与 tools/ 功能重复 |
| `debug_validation.py` | 95 | E04 | 过时调试脚本 |
| `debug_trained_model.py` | 112 | E08 | 过时调试脚本 |
| `test_model.py` | 163 | E05 | 过时测试脚本 |
| `compare_models.py` | 174 | E12 | 与 tools/compare_models.py 重复 |
| `evaluate_test_set.py` | 180 | - | 与 tools/ 功能重复 |
| `finetune_boundary.py` | 341 | E12 | 有价值，但已集成到 train.py |
| `finetune_boundary_simple.py` | 440 | E12 | finetune_boundary 的变体 |
| `train_expanded.py` | 578 | E15 | 已被 src/train.py 完全取代 |

### 2.2 处理方案

#### 方案 A：归档保留（推荐）

将所有脚本移动到 `anti_test/archive/root_scripts/`，保留历史记录：

```bash
# 创建归档目录
mkdir -p anti_test/archive/root_scripts

# 移动所有根目录 Python 脚本
mv verify_env.py anti_test/archive/root_scripts/
mv test_loss_fn.py anti_test/archive/root_scripts/
mv debug_class_imbalance.py anti_test/archive/root_scripts/
mv verify_cell_matching.py anti_test/archive/root_scripts/
mv run_cellsam.py anti_test/archive/root_scripts/
mv debug_validation.py anti_test/archive/root_scripts/
mv debug_trained_model.py anti_test/archive/root_scripts/
mv test_model.py anti_test/archive/root_scripts/
mv compare_models.py anti_test/archive/root_scripts/
mv evaluate_test_set.py anti_test/archive/root_scripts/
mv finetune_boundary.py anti_test/archive/root_scripts/
mv finetune_boundary_simple.py anti_test/archive/root_scripts/
mv train_expanded.py anti_test/archive/root_scripts/
```

#### 方案 B：选择性保留

保留有价值的脚本到 `tools/`，删除其余：

| 脚本 | 操作 | 原因 |
|------|------|------|
| `finetune_boundary.py` | 移动到 `tools/legacy/` | E12 实验参考代码 |
| `compare_models.py` | **删除** | 与 tools/ 完全重复 |
| `train_expanded.py` | **删除** | 已被 src/train.py 取代 |
| 其余 10 个 | 移动到 `anti_test/archive/` | 过时但保留记录 |

### 2.3 推荐方案

**推荐方案 A**（归档保留），理由：
1. 保留完整历史记录，便于回溯
2. 操作简单，风险低
3. 不影响 Git 历史

---

## 三、重复代码处理方案

### 3.1 重复代码清单

项目中存在 3 组重复代码：

#### 3.1.1 SemanticChannelMapper（语义通道映射器）

| 位置 | 文件路径 | 行数 | 参数设置 |
|------|---------|------|---------|
| **主版本** | `src/augmented_dataset.py:25-82` | 58 | P1-P99, sigma=1.5 |
| 重复版本 1 | `src/comparison/sarcgraph_pipeline/preprocessing.py:17-*` | ~100 | P0.5-P99.5, sigma=3 |
| 重复版本 2 | `claude tempt/cellsam_pipeline/preprocessing.py` | ~100 | 同上 |

#### 3.1.2 IndependentChannelAdapter（通道适配器）

| 位置 | 文件路径 | 参数量 |
|------|---------|-------|
| **主版本** | `src/adapters/channel_adapter.py` | ~30 |
| 重复版本 1 | `src/comparison/sarcgraph_pipeline/channel_adapter.py` | ~30 |
| 重复版本 2 | `claude tempt/cellsam_pipeline/channel_adapter.py` | ~30 |

#### 3.1.3 compare_models 脚本

| 位置 | 文件路径 |
|------|---------|
| **主版本** | `tools/compare_models.py` |
| 重复版本 | 根目录 `compare_models.py` |

### 3.2 处理方案

#### 方案 A：删除 claude tempt，保留 src/comparison（推荐）

```bash
# 删除 claude tempt 目录（与 src/comparison 内容相同）
rm -rf "claude tempt"

# src/comparison 保留作为实验参考，但标记为非活跃
echo "# ARCHIVED - 实验性代码，未集成到主项目" > src/comparison/README.md
```

**理由**：
- `claude tempt` 和 `src/comparison/sarcgraph_pipeline/` 内容完全相同
- `src/comparison` 位置更规范，保留一份即可
- 主版本在 `src/augmented_dataset.py` 和 `src/adapters/`

#### 方案 B：全部删除重复，仅保留主版本

```bash
# 删除所有重复目录
rm -rf "claude tempt"
rm -rf src/comparison

# 主版本已在正确位置：
# - src/augmented_dataset.py (SemanticChannelMapper)
# - src/adapters/channel_adapter.py (IndependentChannelAdapter)
```

**理由**：
- 最干净的方案
- 但会丢失 `prompt_generator.py`（SarcGraph Z-线检测）等未集成代码

#### 方案 C：合并后归档

```bash
# 创建研究归档目录
mkdir -p research_archive/sarcgraph_experiment

# 移动 claude tempt 内容（重命名规范化）
mv "claude tempt/cellsam_pipeline" research_archive/sarcgraph_experiment/
rm -rf "claude tempt"

# 删除 src/comparison（内容已归档）
rm -rf src/comparison
```

### 3.3 推荐方案

**推荐方案 A**，理由：
1. 保留 `src/comparison` 作为实验参考（包含 `prompt_generator.py` 等未集成代码）
2. 删除 `claude tempt`（完全重复且命名不规范）
3. 主版本代码位置不变

---

## 四、目录命名规范化方案

### 4.1 问题目录

| 当前名称 | 问题 | 建议名称 |
|---------|------|---------|
| `claude tempt` | 空格 + 拼写错误（tempt→temp） | 删除（见第三节） |
| `ai guide` | 空格 + 中文 | `docs/research_papers` 或删除 |

### 4.2 处理方案

#### claude tempt 目录
按第三节方案删除。

#### ai guide 目录

```bash
# 查看内容
ls -la "ai guide/"

# 方案 A：移动到 docs/research_papers/
mkdir -p docs/research_papers
mv "ai guide"/* docs/research_papers/
rm -rf "ai guide"

# 方案 B：直接删除（如果内容不重要）
rm -rf "ai guide"
```

---

## 五、执行计划

### 阶段 1：安全清理（已完成）

- [x] 删除 `nul` 垃圾文件
- [x] 归档 `temp_act2n_content.txt` 到 `docs/`

### 阶段 2：根目录脚本归档

```bash
# 1. 创建归档目录
mkdir -p anti_test/archive/root_scripts

# 2. 移动所有根目录 Python 脚本
for f in verify_env.py test_loss_fn.py debug_class_imbalance.py \
         verify_cell_matching.py run_cellsam.py debug_validation.py \
         debug_trained_model.py test_model.py compare_models.py \
         evaluate_test_set.py finetune_boundary.py finetune_boundary_simple.py \
         train_expanded.py; do
    mv "$f" anti_test/archive/root_scripts/ 2>/dev/null
done

# 3. 创建归档说明
cat > anti_test/archive/root_scripts/README.md << 'EOF'
# 归档脚本

这些脚本从项目根目录归档，属于早期实验阶段产物。

| 脚本 | 原用途 | 归档原因 |
|------|--------|---------|
| debug_*.py | 调试脚本 | E01-E08 阶段，已过时 |
| test_*.py | 测试脚本 | 功能已集成到 pytest |
| verify_*.py | 验证脚本 | 一次性使用 |
| finetune_*.py | E12 微调 | 功能已集成到 src/train.py |
| train_expanded.py | E15 训练 | 已被 src/train.py 取代 |
| compare_models.py | 模型对比 | 与 tools/ 重复 |
| run_cellsam.py | 推理脚本 | 与 tools/ 重复 |
| evaluate_test_set.py | 评估脚本 | 与 tools/ 重复 |

**归档日期**: 2026-01-30
EOF

echo "✅ 根目录脚本归档完成"
```

### 阶段 3：删除重复代码

```bash
# 1. 删除 claude tempt 目录
rm -rf "claude tempt"

# 2. 标记 src/comparison 为归档状态
cat > src/comparison/README.md << 'EOF'
# ARCHIVED - 实验性代码

此目录包含 SarcGraph 集成实验代码，**未集成到主项目**。

## 状态
- **创建日期**: 2026-01-23
- **状态**: 归档（不再维护）
- **原因**: 主版本代码已在 src/augmented_dataset.py 和 src/adapters/

## 内容
- `preprocessing.py` - SemanticChannelMapper 实验版
- `channel_adapter.py` - ChannelAdapter 实验版
- `prompt_generator.py` - SarcGraph Z-线检测（未集成）
- `pipeline.py` - 完整 Pipeline（未集成）

## 主版本位置
- SemanticChannelMapper → `src/augmented_dataset.py`
- IndependentChannelAdapter → `src/adapters/channel_adapter.py`

如需使用 SarcGraph 功能，请参考 `prompt_generator.py` 进行集成。
EOF

echo "✅ 重复代码处理完成"
```

### 阶段 4：目录规范化

```bash
# 检查 ai guide 目录内容并决定处理方式
ls -la "ai guide/" 2>/dev/null && echo "需要处理 ai guide 目录" || echo "ai guide 目录不存在"
```

---

## 六、预期结果

### 清理前

```
CellSam/
├── compare_models.py          ❌ 重复
├── debug_*.py (3个)           ❌ 过时
├── finetune_*.py (2个)        ❌ 已集成
├── train_expanded.py          ❌ 已取代
├── test_*.py (2个)            ❌ 过时
├── verify_*.py (2个)          ❌ 一次性
├── run_cellsam.py             ❌ 重复
├── evaluate_test_set.py       ❌ 重复
├── "claude tempt"/            ❌ 重复 + 命名不规范
├── src/
│   ├── comparison/            ⚠️ 实验代码
│   └── ...
└── ...
```

### 清理后

```
CellSam/
├── src/
│   ├── adapters/              ✅ 主版本 Adapter
│   ├── augmented_dataset.py   ✅ 主版本 Mapper
│   ├── comparison/            📦 归档（带 README 说明）
│   └── ...
├── anti_test/
│   └── archive/
│       └── root_scripts/      📦 13个脚本归档
└── ...
```

---

## 七、回滚方案

如果清理后发现问题，可通过 Git 回滚：

```bash
# 查看清理前的状态
git status

# 回滚特定文件
git checkout HEAD -- <file_path>

# 回滚所有更改
git checkout HEAD -- .
```

---

## 八、审批检查清单

执行前请确认：

- [ ] 已阅读并理解本方案
- [ ] 确认根目录 13 个脚本可以归档
- [ ] 确认 `claude tempt` 目录可以删除
- [ ] 确认 `src/comparison` 保留但标记为归档
- [ ] 已备份重要文件（或确认 Git 状态干净）

---

**等待审批后执行。**
