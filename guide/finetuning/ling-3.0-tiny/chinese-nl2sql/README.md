# 中文 NL2SQL 微调 Recipe —— Ling-3.0-tiny

在电商订单领域把 `Ling-3.0-tiny`（7.9B MoE / 1.3B 激活）微调为中文 NL2SQL 组件：输入中文自然语言问题 + 数据库结构，输出可执行 SQL。

- **底座模型**：[`inclusionAI/Ling-3.0-tiny`](https://huggingface.co/inclusionAI/Ling-3.0-tiny)（亦可用 [ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-3.0-tiny)）
- **微调方案**：QLoRA（4-bit NF4）+ HF `PEFT` / `TRL`（`SFTTrainer`）
- **目标硬件**：Kaggle 免费 T4 / Colab T4（显存约 10~12 GB 即可）
- **评测**：留出测试集上对比微调前后的「执行准确率 EX / 精确匹配 EM / SQL 合法率」

## 目录结构

```
chinese-nl2sql/
├── README.md                  # 本说明文档
├── data/
│   ├── generate_data.py       # 程序化合成电商 NL2SQL 数据集 (数据集与评测集来源)
│   ├── database.sql           # SQLite 建表 + 示例数据 (评测执行 EX 所用)
│   ├── train.jsonl / val.jsonl / test.jsonl   # 由 generate_data.py 生成
├── train/
│   ├── config.yaml            # QLoRA 训练配置
│   └── train.py               # 训练脚本
├── inference/
│   └── infer.py               # 推理样例
└── eval/
    └── evaluate.py            # 评测脚本
```

## 环境依赖

- Python 3.10+；建议用 `uv` 或 `conda` 建独立环境。
- 安装训练/推理/评测依赖：

```bash
pip install -U torch transformers peft trl accelerate bitsandbytes datasets pyyaml
```

> 运行数据合成与评测本身只依赖 Python 标准库（`random` / `json` / `sqlite3`），无需 GPU。

## 快速开始

### 1. 生成数据（合成电商 NL2SQL 数据集）

```bash
python data/generate_data.py   # 默认 600/100/200 → data/*.jsonl + database.sql
```

产物说明（SFT `messages` 三段式）：

- `system`：NL2SQL 助手角色设定；
- `user`：数据库结构（DDL）+ 中文问题；
- `assistant`：目标 SQL。

数据覆盖单表过滤、多表 JOIN、聚合（GROUP BY/HAVING）、排序 + LIMIT、日期范围等查询类型。

### 2. 训练（QLoRA）

可先在本仓库根目录 `guide/finetuning/ling-3.0-tiny/chinese-nl2sql/` 下直接运行：

```bash
python train/train.py --config train/config.yaml
```

关键超参（见 `train/config.yaml`）：`num_train_epochs=3`、`learning_rate=1e-4`、`per_device_train_batch_size=4`、`gradient_accumulation_steps=4`、`max_seq_length=1024`、`lora r=16/alpha=32`、`target_modules=all-linear`、4-bit NF4。

> **注意**：`Ling-3.0-tiny` 需 `trust_remote_code=True`；若目标模块名因版本变化导致 LoRA 挂载失败，可用 `python -c "from transformers import AutoModelForCausalLM; print([n for n,_ in AutoModelForCausalLM.from_pretrained('inclusionAI/Ling-3.0-tiny', trust_remote_code=True, load_in_4bit=True).model.named_modules() if 'mlp' in n or 'self_attn' in n][:20])"` 查看真实线性层名后微调 `target_modules`。

训练完成后 LoRA Adapter（仅几十 MB）保存于 `train/output/`，可上传至 Hugging Face 托管。

### 3. 推理样例

```bash
# 仅基座（未微调）
python inference/infer.py --question "统计每个城市的客户数量"

# 加载微调后的 LoRA Adapter
python inference/infer.py --adapter train/output --question "查询金额最高的订单号、客户姓名和金额"
```

### 4. 评测（对比微调前后）

```bash
# 基座 baseline
python eval/evaluate.py --messages-file data/test.jsonl --database data/database.sql

# 微调后
python eval/evaluate.py --messages-file data/test.jsonl --database data/database.sql --adapter train/output
```

指标：

| 指标 | 含义 |
|:---|:---|
| EX（执行准确率） | 在内存 SQLite 中执行 gold 与预测 SQL，比对结果集一致性 |
| EM（精确匹配） | 标准化后 SQL 字符串是否一致 |
| SQL 合法率 | 预测 SQL 可被解析执行的占比 |

验收要求是在测试集上体现微调后 EX/EM 相较原生基线的改进。

## 资产托管

- **LoRA Adapter 权重**：上传至 Hugging Face（或 GitHub Release / 独立仓库）。
- **数据集与评测集**（`data/*.jsonl`、`data/database.sql`）：可上传至 Hugging Face 或 GitHub。
- 全参微调不在本 Recipe 范围内。

## AI 使用策略报告

本项目全程由 GitHub Copilot / 通用大模型辅助完成：数据集合成与模板 SQL 由模型给出初稿后人工校验执行正确性；训练与评测脚本框架由 AI 生成，关键超参与评测逻辑经人工审阅修正。过程中 AI 给出的「随机采样去重」初版在数据规模放大后出现去重样本不足的隐患，人工将其改为「模板参数全量枚举 + 洗牌切片」，才在 900 条规模下稳定产出。这类建议提醒我们，AI 生成的脚手架在小规模冒烟时可能掩盖规模边界问题，放大数据量后必须回归验证。

## 复现说明

1. 建独立 Python 环境并安装依赖；
2. `python data/generate_data.py` 生成数据；
3. （可选）在 Kaggle/Colab 开启 T4 GPU，跑 `python train/train.py --config train/config.yaml`；
4. 分别以「基座」与「--adapter train/output」跑 `eval/evaluate.py`，对比三指标；
5. 将 LoRA Adapter 与数据上传 HF，并在文档中补链。