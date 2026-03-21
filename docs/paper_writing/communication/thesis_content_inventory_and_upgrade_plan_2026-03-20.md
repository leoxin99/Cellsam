# 论文内容盘点与增补方案（A3，2026-03-20）

## 1. 当前已具备内容（按章节）

1. `abstract/ch1/ch6`：主线已固定为 `T28`（分割）+ `T33g+T28`（端到端），并保留 split-aware 证据边界。  
2. `ch2`：生物背景、相关工作、人鼠差异、CellProfiler 价值已写入。  
3. `ch3`：数据集定义、Oracle/E2E 评估协议、指标口径与 reporting rules 已写入。  
4. `ch4`：T28 训练配置、算法流程（Algorithm 1/2）和方法章节主干已具备。  
5. `ch5`：主表与主结果分支（T28/T29/T12/T33g+T28/S2 锚点/T30 简述）已具备。  
6. `appendix A/B`：checkpoint 审计事实、补充负结果与机制图位已具备。  

## 2. 本轮识别的关键缺口（已处理）

1. `T34/official-path` 叙事残留在 `ch1/ch3/ch4/ch5/ch6`。  
2. `ch2` 缺少“CellSAM/Cellpose/MedSAM 结构原理与本项目角色分工”的集中段。  
3. `ch3` 指标只有文字定义，缺公式层闭环。  
4. `ch4` 预处理/后处理描述偏流程化，不够可复现。  

## 3. 本轮落地改动（LaTeX-first）

1. 删除并收口 `T34` 线：移除 `ch5` 的 T34 小节，并清理 `ch1/ch3/ch4/ch6` 相关叙事。  
2. `ch2` 新增模型原理小节：明确 CellSAM 架构、`T33g`(box) + `T28`(mask) 角色分工，以及 Cellpose v4.0.1 与 MedSAM 的方法定位。  
3. `ch3` 新增公式化指标定义：PQ/SQ/RQ、Precision/Recall/F1、BM-1to1 Dice（Hungarian）、AJI。  
4. `ch4` 新增可复现方法段：  
   - Final Inference Pipeline（端到端主流程）  
   - Preprocessing and Postprocessing Design（预处理/后处理细节）  
   - Optimization Objectives（分割损失与检测损失公式）  
   - 模块角色映射表（Detector/Segmenter/Prompt/Postprocess）  

## 4. 仍需后续补齐的内容（与实验封板联动）

1. H1b 最终封板图文（以最终图资产为准）与正文 caption 的最终数值核对。  
2. Fig.1/Fig.2/Fig.3 原理图最终成图插入（当前已锁定文案语义，待你生图后替换）。  
3. 若后续新增封板数字，需仅在 split-aware 口径下更新 `ch5/ch6/abstract` 三处同步。  

## 5. 证据优先级（写作准则）

1. A1/H1b 最新交接文档（2026-03-20）  
2. 对应独立实验文档（`docs/experiments/active|completed`）  
3. `experiments_log` 仅作历史索引，不作为最新数值唯一来源  
