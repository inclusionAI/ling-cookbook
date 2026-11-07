# Ling-3.0-tiny 单词记忆助手：LoRA 微调 Recipe

**作者：** [DoubleCore](https://github.com/DoubleCore)

本 Recipe 将 [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) 微调为面向英语单词记忆的结构化助手：输入单词，输出【释义】【记忆方法】【例句】三段，并保持正常结束生成。

> 结果范围：当前已验证的是 **4 条训练记录、48 步、LoRA r=8 的小规模 pilot**。它用于验证训练/推理因果一致性和 Recipe 的可运行性，不冒充完整语料训练。12 个独立测试词的结构完整率由 **11/12 提升到 12/12**；样本很小，不构成统计显著性或事实/教学质量提升的证明。

## 关键兼容性修复

在已测试的 `transformers==4.57.6` 与该 checkpoint 的远程建模代码组合中，`BailingMoeV3MultiLatentAttention` 固定调用 eager attention。默认 `sdpa` 加载且输入无右侧 padding 时，模型可能得到 `attention_mask=None`，使 teacher-forcing 前向缺少因果约束：训练 loss 接近零，但自回归生成退化为空小节标题后 EOS。

同一 Adapter、同一真实前缀的实测：

| 后端 | 完整序列与纯前缀末位 logits 最大差 | 下一 token |
|---|---:|---|
| 默认 SDPA、无 padding | 22.375 | 错误 |
| eager | 0.46875 | 正确 |

因此：

- **推理/评测固定使用 `attn_implementation="eager"`**。
- 动态 padding 训练也必须使用 eager。
- 固定右侧 padding 可提供显式 mask，但任何 SDPA 训练配置仍须单独通过“保存→独立进程重载→生成”验收，不能仅看 loss。
- 已经在错误 SDPA 前向上训练的旧 Adapter，单纯切换 eager 推理不能挽救，需要重新训练。

## 实测结果

硬件：单张 NVIDIA RTX 5880 Ada 48 GB；BF16；贪心解码；最多 384 个新 token；Base 与 Adapter 均使用 eager。

| 指标（12 个独立测试词） | Base | LoRA r=8 pilot |
|---|---:|---:|
| 三段齐全且非空、含英文例句 | 11/12 | **12/12** |
| 正常 EOS | 11/12 | **12/12** |
| 以【释义】开头 | 12/12 | 12/12 |

补充检查：2 条训练集内样例均正常生成并 EOS。Base 失败例 `deteriorate` 达到 384-token 上限且未完成结构；pilot 在该词上生成 85 tokens 并正常 EOS。

这些自动指标只检查格式与停止行为。逐条原始输出保留在 `results/`，便于人工检查释义、词源和例句是否可靠。小 pilot 的输出仍可能出现不严谨的词源解释，**不应直接当作权威词典或教学内容**。

## 文件

```text
word-memory-lora/
├── train_lora.py           # LoRA SFT，assistant-only labels，支持断点续训
├── infer.py                # 单条 Base/LoRA 推理
├── data/prepare_data.py    # 用户自有 JSONL 的校验和稳定哈希划分
├── eval/cases.json         # 独立构造的 12 个测试词 + 2 条训练样例标记
├── eval/run_eval.py        # Base/LoRA 逐条真实生成
├── eval/score_outputs.py   # CPU 结构评分
├── results/                # 真实原始输出和评分
├── scripts/                # 可复制的训练/评测命令
└── requirements.txt
```

## 环境与数据

推荐 NVIDIA 24 GB 以上显存；本 Recipe 的 7.9B BF16 基座本身约占 16 GB。`r=32` 目标模块覆盖 MLA/KDA、稠密 FFN 和 MoE 专家，Adapter 体积会明显大于只挂注意力投影的常见配置。

训练数据来自用户已有的出版物整理记录，**原书和派生训练集均不随 PR 分发**。如需复现，请准备自己有权使用的 JSONL，每行格式：

```json
{"word":"example","instruction":"请给出英语单词「example」的记忆方法，并附上释义和例句帮助我记住它。只输出【释义】【记忆方法】【例句】三段，不要加标题、编号、emoji 或额外解释。","output":"【释义】...\n【记忆方法】...\n【例句】..."}
```

校验和按词哈希划分：

```bash
python data/prepare_data.py /path/to/your/records.jsonl --output-dir work/data
```

原任务数据为 3,823 条训练记录、363 条内部留出记录；当前发布的 pilot 只取训练文件前 4 条、训练 48 步，**并未使用 3,823 条完整训练**。

## 复现

设 Recipe 根目录为 `$RECIPE`，基座模型本地路径或 Hub ID 为 `$MODEL`。

```bash
python -m pip install -r requirements.txt

# 最小 pilot：单卡，48 步，动态 padding + eager（与已验收配置一致）。
CUDA_VISIBLE_DEVICES=0 python train_lora.py \
  --model "$MODEL" --train-file work/data/train.jsonl \
  --output-dir work/lora-r8-pilot --limit 4 --max-steps 48 \
  --rank 8 --learning-rate 1e-4 --micro-batch-size 2 \
  --gradient-accumulation-steps 1 --max-length 896 --padding dynamic \
  --gradient-checkpointing --attn-implementation eager
```

先对少量样本跑通保存/重载，再扩大数据。Base 与 LoRA 必须在同一后端、同一解码参数下评测：

```bash
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py \
  --mode base --model "$MODEL" --cases eval/cases.json \
  --attn-implementation eager --output results/base-eager.json

CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py \
  --mode lora --model "$MODEL" --adapter work/lora-r8-pilot/final \
  --cases eval/cases.json --attn-implementation eager \
  --output results/lora-r8-eager-with-train.json

python eval/score_outputs.py results/base-eager.json \
  results/lora-r8-eager-with-train.json --output results/scores.json
```

单条推理：

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model "$MODEL" --adapter work/lora-r8-pilot/final --word photosynthesis
```

## 性能与限制

- 本 pilot 实际训练约 **750 秒**（r=8、4 条、48 步、eager、动态 padding）；生成 12 条独立测试约 **252 秒**。
- 全 eager 的注意力稠密化较慢。`r=32`、batch 4、gradient checkpointing 的固定 896 长度实测约 **79 秒/有效步**；未开 gradient checkpointing 时该配置在 48 GB 卡上 OOM。
- 不要根据训练 loss 判断成功。必须独立重载并检查训练集内、独立测试词的生成。
- 12 条测试样本过小，后续应扩大独立测试集，加入人工事实核验、词源可靠性和多随机种子评测。

## AI 使用策略报告

本任务使用 AI 协助整理训练/推理脚本、构造独立测试用例和阅读 Transformers/TRL 与远程建模代码。AI 最初根据“loss 接近零”推测训练正常，但真实生成推翻了该判断；后续所有结论均改为以实际 Base/LoRA 生成、同一前缀的双前向和独立重载结果为准。AI 也协助发现了 SDPA 下自定义 MLA 的因果 mask 缺口，并通过 eager 小规模重训验证，而不是把未经实测的建议直接写进文档。

## 权重

已验证的小规模 Adapter 托管于 [GitHub Release](https://github.com/Picodet/ling-cookbook/releases/tag/ling-word-memory-lora-r8-pilot)，符合 Issue #11 的 LoRA 资产托管方式：

```bash
mkdir -p work/lora-r8-pilot/final
BASE_URL=https://github.com/Picodet/ling-cookbook/releases/download/ling-word-memory-lora-r8-pilot
for file in adapter_model.safetensors adapter_config.json tokenizer.json \
  tokenizer_config.json special_tokens_map.json chat_template.jinja; do
  curl -fL "$BASE_URL/$file" -o "work/lora-r8-pilot/final/$file"
done
echo '1c682d99cd845fcd4d6d58cf250bb44e2e3f5ef502cbabcc1e32adc66b31ba4c  work/lora-r8-pilot/final/adapter_model.safetensors' | sha256sum -c -
```

Adapter 为 603,042,504 字节，因目标模块覆盖 MLA/KDA 和专家层，不能按“只有几十 MiB”估计。下载后用本地目录推理：

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model "$MODEL" --adapter work/lora-r8-pilot/final --word photosynthesis
```

[Hugging Face 模型仓库](https://huggingface.co/yucki/ling-word-memory-lora-r8-pilot)目前仅存放 README、配置和 tokenizer，**尚无 Adapter 大文件**，不能作为该权重的加载地址。
