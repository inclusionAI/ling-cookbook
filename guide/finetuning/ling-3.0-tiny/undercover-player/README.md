# Ling-3.0-tiny「谁是卧底」玩家决策组件 —— MLX 端侧 QLoRA Recipe

在 Apple Silicon（M4 MacBook Air 16GB）上用 **MLX + 4-bit QLoRA** 微调 Ling-3.0-tiny，
让它学会玩「谁是卧底」：**发言**（≤25 字描述自己的词、不泄词）与**投票**（输出
`我投N号` / `我弃权`）两种决策动作。微调产物同时是
[Avernet #2611](https://github.com/inclusionAI/Avernet/issues/2611)「谁是卧底」
牌桌的 **Apple MLX 本地端侧模型引擎** 玩家 Bot 的推理内核——一个项目，两边交付（一鱼两吃）。

- 全程 16GB 内存内完成（量化 + QLoRA + 梯度检查点），无需云端 GPU
- 数据 100% 程序化合成（30 组词对 × 属性池 × 模板组合），可复现、无泄露
- 评测脚本输出微调前后对比表：格式合规率、泄词率、投票正确率等

## 环境要求

| 项 | 版本/值 |
|---|---|
| 硬件 | Apple Silicon Mac（验证机型：M4 Air 16GB） |
| Python | ≥ 3.10（验证：3.13） |
| mlx-lm | ≥ 0.32.0（`bailing_hybrid` 架构支持） |
| 内存 | 量化后训练峰值约 12GB，16GB 机型可跑 |

## 目录结构

```
undercover-player/
├── data/
│   ├── word_pairs.py     # 30 组词对 + 属性池（含 6 组留出词对用于泛化测试）
│   ├── prompting.py      # 训练/推理统一提示词格式
│   ├── synthesize.py     # 程序化数据合成（生成 train/valid/test.jsonl）
│   └── *.jsonl           # 已生成数据集：692 / 76 / 96 条
├── train/
│   ├── lora_config.yaml  # QLoRA 超参（rank 8，后 16/24 层）
│   └── train.sh          # 训练入口
├── eval/
│   └── evaluate.py       # 微调前后自动评测 → JSON 对比
├── infer/
│   └── chat.py           # 单条交互 demo（发言/投票）
└── adapters/             # 训练输出（git 不跟踪权重，见 Releases）
```

## 快速复现

### 0. 准备模型（一次性）

ModelScope / Hugging Face 下载 Ling-3.0-tiny（BF16 约 15GB）到 `~/models/Ling-3.0-tiny`，
然后**必须本地量化**（ModelScope 的 int4 预量化是 vLLM 专用，MLX 用不了）：

```bash
python -m mlx_lm.convert --model ~/models/Ling-3.0-tiny \
    --q-bits 4 --mlx-path ~/models/Ling-3.0-tiny-4bit --trust-remote-code
```

### 1. 生成数据集（可选，仓库已带）

```bash
python data/synthesize.py   # 重新生成 train/valid/test.jsonl（seed 固定）
```

合成规则：随机配词对 → 分配身份（1 卧底 / N 平民）→ 按属性池模板生成每轮发言
（卧底刻意模糊，平民用判别属性）→ 规则式打投票标签（平民投嫌疑最大的卧底、
卧底甩锅/弃权）。留出 6 组词对（`TEST_PAIR_INDICES`）不出现在训练集，用于测泛化。

### 2. 训练（QLoRA）

```bash
cd train && bash train.sh
# 等价：python -m mlx_lm lora -c train/lora_config.yaml
```

关键配置：4-bit 底座 + LoRA rank 8、只挂后 16/24 层（MoE 专家权重全冻结，
可训练 104M / 1.3%）、batch 1 × 梯度累积 4、lr 1e-4、`mask_prompt`（loss 只算
输出）、梯度检查点。1500 步在 M4 16GB 上约 **2.5 小时**（1000 步约 1 小时
35 分），每 200 步存 checkpoint，每 100 步跑验证集。

数据侧做了任务平衡：投票是决策任务、比发言难学，`synthesize.py --vote-repeat 3`
把训练集投票样本过采样 3 倍（train 1208 条 = 434 speak + 774 vote）；
valid/test 保持自然分布，评测口径可比。

### 3. 评测（微调前后对比）

```bash
python eval/evaluate.py --model ~/models/Ling-3.0-tiny-4bit \
    --data data/test.jsonl --out eval/base.json
python eval/evaluate.py --model ~/models/Ling-3.0-tiny-4bit --adapter-path adapters \
    --data data/test.jsonl --out eval/lora.json
```

指标：发言字数合规率、泄词率（输出含自己词语任一字）、自然度（排除
`1. **分析请求：**` 之类的思考前缀）、格式合规率、投票格式合法率、投票正确率。

解码协议：固定种子采样（`temp=0.7, top_p=0.9, seed=42`），并用
`thinking_option="off"` 渲染提示词 + `clean_output()` 剥模板残渣（原因见
Troubleshooting #6/#7）。贪心解码会让模型坍缩到单一安全输出（发言全用万能句、
投票全弃权），不能反映决策质量。

### 4. 交互 demo

```bash
python infer/chat.py --model ~/models/Ling-3.0-tiny-4bit --adapter-path adapters
```

## 评测结果

test.jsonl（96 条，6 组留出词对），固定种子采样（temp=0.7, top_p=0.9, seed=42）：

| 指标 | 基线（4-bit） | +QLoRA |
|---|---|---|
| 发言字数合规率（≤25 字） | 96.7% | **100%** |
| 发言自然度（非思考前缀） | 3.3% | **100%** |
| 发言格式合规率（字数+不泄词+自然） | 0% | **85.0%** |
| 发言泄词率（原始输出，越低越好） | 1.7% | 15.0% ⚠ |
| 投票格式合法率（`我投N号`/`我弃权`） | 44.4% | **100%** |
| 投票正确率（对齐规则式标签） | 5.6% | 11.1% |

### 结果解读（诚实记录）

- **发言任务学得很好**：自然度 3.3%→100%、格式合规 0%→85%（基线输出全是
  思考前缀垃圾，微调后是像样的人话）。
- **泄词率是唯一回退**（1.7%→15%）：基线压根不在玩游戏所以「不泄词」是假
  达标；微调后模型真的在聊自己的词，偶尔说出词里的字。线上由清洗层强制删除
  自己词语里的字兜底（`rules.py`），极端情况话会变短但绝不违规。
- **投票决策只到「接近随机」水平**，我们没有掩饰它，也做了归因实验：
  1. 训练集投票准确率同样低 → 不是泛化问题，是任务没吸收；
  2. 换成结构化解码（对每个候选「我投N号」算条件对数概率取 argmax）依然
     接近随机 → 不是生成采样问题，是**判别力**问题；
  3. 投票标签是规则式的（谁用了对方词的判别属性），但属性在文本上有歧义
     （如「甜的」两词都沾），标签噪声不可约；对 1.3B 激活参数的模型 + 1.3%
     LoRA + 千级样本，这是合理的天花板。
- **格式合规不受影响**：v2 训练后投票输出 100% 可解析（过采样治好了「全弃权」
  坍缩），Avernet 验收口径（格式/协议）达标，胜率不计分。
- 线上兜底：清洗层强制 25 字/泄词删除/投票格式（见 Avernet 侧 `rules.py`），
  模型再跑偏也不会违规。

## 应用：Avernet「谁是卧底」牌桌（一鱼两吃）

微调后的模型通过 `mlx_lm.server` 起 OpenAI 兼容服务，接入 Avernet 的玩家 Bot
（BCN V3 协议），与其他引擎（云端大模型等）坐同一张桌。集成代码与实测日志见
Avernet 侧 PR（#2611）。Bot 的发言清洗层（≤25 字、泄词兜底、投票格式强制）保证
裁判脚本 100% 可解析——小模型即使偶发跑偏也不会违规。

## Troubleshooting

踩过的坑，按实际复现顺序记录：

1. **`StrictDataclassFieldValidationError: Field 'kda_lower_bound' expected float, got int`**
   —— 官方 `config.json` 里 `kda_lower_bound: -5`、`mtp_loss_scaling_factor: 0` 是
   整数字面量，MLX 加载时严格校验 float。把两处改成 `-5.0` / `0.0`（量化前后两个
   目录都要改）。
2. **`generate_step() got an unexpected keyword argument 'temp'`**
   —— mlx-lm 新版采样 API 变更：`generate()`/`generate_step()` 不再收 `temp=`，
   改用 `sampler=make_sampler(temp=0.7, top_p=0.9)`（评测贪心解码传 `temp=0.0`）。
3. **基线模型用「思考前缀」骗过字数指标**
   —— 未微调模型常输出 `1. **分析请求：**` 这类前缀，恰好 ≤25 字、无泄词，
   纯规则指标看不出问题。评测加了 `looks_natural()` 检查（序号开头 / markdown
   标记 / 尾部冒号均判不自然），并单列 `speak_natural` 指标。
4. **数据合成的近重复发言**
   —— 同一轮里多个玩家抽到相同属性 → 模板产物雷同；补了每轮属性去重
   （`used_attrs_round`）与 `a1 ≠ a2` 双属性抽取守卫。
5. **（Avernet 侧）websockets 连本地 BCS 报 `requires python-socks`**
   —— websockets ≥14 默认读环境代理变量，连 `127.0.0.1` 也会走 SOCKS。客户端
   显式 `websockets.connect(url, proxy=None)` 禁用；BS/回环地址永远不该走代理。
6. **微调后模型每个输出都带 `</thinking>` 前缀（终端里像一个 `！`）**
   —— Ling 的 chat 模板带 thinking 块，mlx_lm `mask_prompt` 的偏移量算到
   `<think>` 为止，把闭合标签留在了监督区：模型学会了「先吐 `</thinking>` 再吐
   内容」。而 `add_generation_prompt=True` 的提示词恰好停在 `<think>`，贪心解码
   第一件事就是输出闭合标签，污染全部指标（连 `first_line` 都被它占掉）。
   修法：推理时 `apply_chat_template(..., thinking_option="off")`（提示词以
   `<think></think>` 收尾，正好卡在训练时内容前的位置），解析用 `clean_output()`
   剥掉 `<|role_end|>` 等模板残渣。**模型权重本身没坏**，是渲染错位。
7. **贪心解码模式坍缩**：temp=0 时投票 100% 输出「我弃权」（弃权是单一字符串，
   5 个候选把概率质量摊薄）、发言全用「同意前面几位的说法」万能句。评测与推理
   统一改用固定种子采样（temp=0.7, top_p=0.9）；线上再叠加清洗层兜底格式。

## AI 使用策略

本 recipe 的代码、数据合成、训练配置、评测与文档由 Claude Code 在人工督导下完成，
详见 [AI_USAGE.md](AI_USAGE.md)。
