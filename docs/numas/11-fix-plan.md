# 中文 NL2SQL 微调 Recipe 方案设计 (issue #11)

> 任务仓库: inclusionAI/ling-cookbook · 任务 issue: #11 · 任务ID: 98 · 实验室编码: antgroup

## 1. 问题理解

**任务本质**：这是任务型 issue（非 bug）。要求基于 `Ling-3.0-tiny`（7.9B MoE / 1.3B 激活）面向任意具体场景做微调，产出可复现的 Recipe 合入本仓库。

**issue 四个必做动作**：
1. 场景与数据 — 选取/构建特定场景数据集，定义输入输出格式。
2. 推理与微调 — 按硬件选方案（Mac/MLX、GPU/PEFT·AReno、线上 Kaggle/Colab）。
3. 评测与验证 — 自动化脚本或 Notebook，对比微调前后效果。
4. 沉淀 Recipe — 说明文档 + 训练配置 + 推理样例 + 评测方案。

**验收标准**：
1. 说明明确、可复现；2. 评测对比明确（微调后优于原生基线）；3. 资产合理托管（LoRA → HF/GitHub，数据 → HF/GitHub）；4. 代码与文档遵循项目规范。

## 2. 定位与仓库现状

- 交付目录（待新建）：`guide/finetuning/ling-3.0-tiny/chinese-nl2sql/`
- 仓库现状：仅有 `guide/local-deploy/`，无任何 `finetuning` 目录或内容。
- 仓库规范：内容写 `src/` 下 MyST Markdown → `scripts/compile_notebooks.py` 单向编译为 `guide/*.ipynb`，禁止直接改 `.ipynb`。
- 交付物为独立脚本 + README，不占用 jupytext 编译链路（`.py`/`.md` 作为普通文件直接入库）。
- 格式样板参考：`src/guide/local-deploy/ling-3.0-tiny/mac-mlx-ling-3-tiny.md`。

## 3. 候选方案对比（已由用户拍板）

| 决策点 | 候选 | 取舍 | 最终选择 |
|:---|:---|:---|:---|
| 微调场景 | 中文 NL2SQL / 数学COT / NLU / 代码修复 | NL2SQL 输入输出清晰、评测客观可量化、数据易合成 | **中文 NL2SQL** |
| 硬件/技术栈 | Kaggle·Colab+PEFT / 本地GPU / Mac+MLX | 无本地卡假设下 Kaggle T4 门槛最低 | **Kaggle T4 + PEFT/TRL QLoRA(4bit NF4)** |
| 数据来源 | 程序化合成 / 公开数据集 / 混合 | 合成完全可控可复现、无版权风险 | **程序化合成电商订单领域** |
| 评测指标与形式 | EX+EM+合法率脚本 / 仅EM / Notebook | EX 对等价 SQL 更鲁棒 | **EX + EM + 合法率，独立脚本** |
| 业务领域 | 电商/教育/医疗 | 电商多表 JOIN/聚合贴近常用 NL2SQL | **电商订单领域** |
| 项目目录名 | chinese-nl2sql / chinese-nl2sql-ecommerce | 简短通用 | **chinese-nl2sql** |
| 交付结构 | README+独立脚本 / 全Notebook / 混合 | 契合 issue 交付清单且脚本可单独复现 | **README + 独立脚本** |

## 4. 最终方案

### 4.1 场景与数据（程序化合成电商 NL2SQL）

- **领域 Schema**：电商订单库，拟含 `customers` / `products` / `orders` / `order_items` / `payments` 等多表。
- **覆盖查询类型**：单表过滤(WHERE)、多表 JOIN、聚合(GROUP BY/HAVING)、排序(ORDER BY)、限定(LIMIT)、条件表达式等。
- **样本格式（SFT 三段式）**：
  - `system`：中文 NL2SQL 助手角色设定；
  - `user`：中文自然语言问题 + 数据库 schema（DDL/表字段说明）；
  - `assistant`：目标 SQL。
- **规模**：约 900 条（训练 600 / 验证 100 / 测试 200），符合 issue「几百~一千条」。

### 4.2 推理与微调

- **基座**：`inclusionAI/Ling-3.0-tiny`（HF / ModelScope）。
- **方法**：QLoRA，4-bit NF4，LoRA rank 8~16，`SFTTrainer`（TRL）。
- **硬件**：Kaggle 免费 T4 (16GB)。
- **关键超参（训练配置中固化）**：`lr ≈ 1e-4`、`batch_size`、`epochs ≈ 3`、`max_seq_len`、`target_modules`、`lora_alpha`、`lora_dropout` 等。
- **产物**：LoRA Adapter（仅几十 MB），托管至 Hugging Face。

### 4.3 评测与验证

- **指标**：
  - 执行准确率 EX：在内存 SQLite 执行 gold/pred SQL 比对结果集；
  - 精确匹配 EM：标准化(去空白/大小写)后字符串比对；
  - SQL 合法率：预测 SQL 可被解析/执行的占比。
- **对比方式**：原生 `Ling-3.0-tiny`（baseline） vs 微调后，在留出测试集上算上述指标并输出对比表。
- **形式**：独立评测脚本（`eval/evaluate.py`）。

### 4.4 交付文件结构

```
guide/finetuning/ling-3.0-tiny/chinese-nl2sql/
├── README.md                  # 说明文档：环境、数据、训练、评测、复现步骤、资产链接
├── data/
│   └── generate_data.py       # 程序化合成电商 NL2SQL 数据集
├── train/
│   ├── config.yaml            # QLoRA 训练配置（基座/超参/数据路径）
│   └── train.py               # PEFT/TRL QLoRA 训练脚本
├── inference/
│   └── infer.py               # 推理样例（加载 LoRA 对话式 NL2SQL）
└── eval/
    └── evaluate.py            # 评测脚本（EX/EM/合法率，对比微调前后）
```

## 5. 验收标准（映射 issue）

1. **可复现**：README 给出清晰环境依赖与操作步骤，他人可跑通评测/推理。
2. **评测对比明确**：测试集上体现微调后相较原生 baseline 的改进（EX/EM 提升）。
3. **资产托管**：LoRA Adapter → HF；数据集与评测集 → HF/GitHub；不要求全参。
4. **规范**：遵循仓库规范（README + 脚本，`src/*.md`↔`*.ipynb` 编译规则不受影响）。

## 6. 边界与额外要求

- **不要求**多机分布式训练、复杂前端交互。
- **AI 使用策略报告**：issue 要求，需在交付物中附 1~2 段说明（用哪些 AI 工具、如何辅助生成代码/数据、有无启发或不靠谱建议）。
- 认领指引：官方建议在 issue 评论区留言登记（学号/GitHub ID + 场景 + 技术栈），提交前可考虑补登。