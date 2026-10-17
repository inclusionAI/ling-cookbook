<p align="right">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

# 全双工对话大脑：LoRA 微调 Ling-3.0-tiny 做话轮管理

**作者：** [Dayuxiaoshui](https://github.com/Dayuxiaoshui)

全双工语音助手可以同时听和说。除了"说什么"，它还要不停地判断"什么时候行动"：用户句中停顿时继续等，用户长篇讲述时给一句"嗯嗯"，用户说完时接话，被打断时停下，对不是说给它的话保持不理会。

本 recipe 用 LoRA 把 [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny)（MoE，总参 7.9B / 激活 1.3B）微调成这样一个**对话大脑**。在每个决策时刻，模型读取实时 ASR 文本，输出一个动作标签；需要开口时，再跟一句简短回复：

| 动作 | 触发场景 |
| :--- | :--- |
| `[LISTEN]` | 用户还没说完（句中停顿、犹豫、以"然后……"结尾）。保持沉默。 |
| `[BACKCHANNEL]` | 用户在长段讲述中。给一个简短应答，不抢话轮。 |
| `[SPEAK]` | 用户说完了一段对助手说的话。接话回复。 |
| `[STOP]` | 助手正在说话，用户插话提出新的或修改后的请求。停下并回应新请求。 |
| `[IGNORE]` | 这段语音不需要回应：简短的"嗯/对"、用户在和别人说话、背景人声。 |

评测集由 [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) v1.0 / v1.5 的标注转换为决策点。最佳 checkpoint 的效果如下：

- **准确率：** 0.581 → **0.885**
- **macro F1：** 0.374 → **0.737**
- **对应忽略语音的误回应率：** **33.6% → 0.7%**
- **被打断时的停止率：** **62.5% → 96.5%**

训练全程未使用任何 Full-Duplex-Bench 数据。

---

## 目录

- [任务定义](#任务定义)
- [文件结构](#文件结构)
- [环境](#环境)
- [复现步骤](#复现步骤)
- [结果](#结果)
- [分析](#分析)
- [局限](#局限)
- [权重与数据](#权重与数据)
- [AI 使用策略报告](#ai-使用策略报告)

---

## 任务定义

一次请求对应一个决策时刻。输入包括三部分：助手当前状态、用户到目前为止说的话（ASR 文本）、用户最后一个词之后的静音时长：

```text
Assistant state: speaking (currently answering: "Can you help me plan a weekend trip to Kyoto?")
New user speech: "wait, actually make it Osaka instead"
Silence since the last word: 0.3s
Action:
```

输出是一个动作标签。`[SPEAK]` / `[STOP]` 后接回复，`[BACKCHANNEL]` 后接应答语：

```text
[STOP] No problem, Osaka is a great choice too.
```

System prompt 和输入格式定义在 [`fdb_brain/prompt.py`](fdb_brain/prompt.py)，数据构造、评测、推理三处共用这一份。动作标签放在最前面，下游控制器读到前几个 token 就能执行，不必等回复生成完。

**静音时长固定**，训练和评测都一样：助手沉默时为 0.8s，助手说话时为 0.3s。这样模型无法靠停顿长短直接猜出标签，只能从*用户说了什么*来区分"句中停顿"和"说完了"。这也正是纯文本 LLM 在 VAD 之上能额外贡献的信息。

## 文件结构

```text
full-duplex-dialogue-brain/
├── fdb_brain/prompt.py        # 共用的 system prompt、输入渲染、动作解析
├── data/
│   ├── build_fdb_eval.py      # Full-Duplex-Bench 标注 -> 1862 个评测决策点
│   ├── synth_scenarios.py     # 用基座模型生成对话场景
│   └── build_train.py         # 场景 -> 带标签、预分词的 SFT 数据
├── train/
│   ├── train_lora.sh          # AReno LoRA SFT（训练配置）
│   └── sft_loader.py          # 传给 `areno train` 的数据加载函数
├── scripts/serve.sh           # 启动 OpenAI 兼容的 `areno serve`（可挂 LoRA）
├── eval/run_eval.py           # 请求 + 打分；输出逐条 jsonl 和 report json
├── infer.py                   # 推理示例（单次决策或脚本化对话）
└── results/*.report.json      # 下文各表对应的全部评测报告
```

## 环境

| 项目 | 使用版本 |
| :--- | :--- |
| GPU | 8× NVIDIA H20 96GB（训练用 8 卡，推理用 4 卡） |
| 训练 / 推理框架 | AReno，commit `3cfa480`（`areno train`、`areno serve`） |
| 基座模型 | `inclusionAI/Ling-3.0-tiny` |
| 评测数据 | Full-Duplex-Bench v1.0 + v1.5（只用标注，不用音频） |
| 数据 / 评测脚本依赖 | `httpx`、`transformers`（只用 tokenizer） |

如果 `areno` 不在 `PATH` 中，请设置 `ARENO_VENV=/path/to/venv`。以下所有命令都在本目录下执行。

```bash
export M=/path/to/Ling-3.0-tiny        # inclusionAI/Ling-3.0-tiny 本地快照
export FDB=/path/to/Full-Duplex-Bench  # 包含 v1.0/ 和 v1.5/ 的目录
```

## 复现步骤

### 1. 构造评测集（Full-Duplex-Bench → 决策点）

```bash
python data/build_fdb_eval.py --fdb-root $FDB --out data/fdb_eval.jsonl
```

每条 FDB 样本会转换成一个或多个文本级决策点：

| FDB 子集 | 决策点 | 标签 | 数量 |
| :--- | :--- | :--- | ---: |
| v1.0 `candor_pause_handling` | 截至每个标注 `[PAUSE]` 的文本 | LISTEN | 277 |
| v1.0 `synthetic_pause_handling` | 截至每个停顿的文本 / 完整话轮 | LISTEN / SPEAK | 137 / 137 |
| v1.0 `icc_backchannel` | 长独白中截至每个 ≥0.3s 间隙的文本 | BACKCHANNEL（LISTEN 也算对） | 315 |
| v1.5 `user_interruption` | 请求（沉默）/ 插话（说话中） | SPEAK / STOP | 200 / 200 |
| v1.5 `user_backchannel` | 请求 / "嗯嗯"类应答 | SPEAK / IGNORE | 98 / 98 |
| v1.5 `talking_to_other` | 请求 / 用户和他人说话 | SPEAK / IGNORE | 100 / 100 |
| v1.5 `background_speech` | 请求 / 背景人声 | SPEAK / IGNORE | 100 / 100 |

合计 **1862** 个决策点。

### 2. 用基座模型生成训练场景

训练数据完全是合成的。先以关闭思考的方式部署**基座**模型，再让它写对话场景。每个场景对应一个"话题 × 人设"组合，共 14 个字段，包括请求与回复、插话与回复、旁人对话、背景人声、长故事、话题切换等，详见 [`data/synth_scenarios.py`](data/synth_scenarios.py) 中的 `GEN_PROMPT`。

```bash
bash scripts/serve.sh $M 0,1,2,3 8100 /tmp/serve-base.log --disable-thinking
python data/synth_scenarios.py --base-url http://127.0.0.1:8100 --n 1500 --seed 1 --out data/scenarios_r2.jsonl
python data/synth_scenarios.py --base-url http://127.0.0.1:8100 --n 800  --seed 2 --out data/scenarios_r2b.jsonl
```

格式不合格的生成结果会被丢弃，最终分别保留 984 条和 498 条。发布的模型还用到了 1200 条早期场景（`data/scenarios.jsonl`，seed 0），它们来自 8 字段的旧版 prompt。这些场景已包含在发布的数据集中，可以直接下载，无需重新生成。

### 3. 场景展开为带标签的决策点

```bash
python data/build_train.py \
  --scenarios data/scenarios.jsonl data/scenarios_r2.jsonl data/scenarios_r2b.jsonl \
  --fdb-eval data/fdb_eval.jsonl --tokenizer $M --out-dir data/
```

每个场景最多展开为 18 个决策点，覆盖全部五种动作。几个例子：

- 请求在句中截断 → LISTEN
- 完整请求 → SPEAK + 回复
- 助手说话时用户插话 → STOP + 回复
- 助手说话时用户和旁人说话 → IGNORE
- 故事在完整句子之后截断 → BACKCHANNEL

完整列表见 [`data/build_train.py`](data/build_train.py) 的 docstring。有三个关键设计：

- **ASR 风格噪声。** 用户文本随机去掉标点、插入 `[UM]` / `[UH]` 填充词（各 35% 概率），避免模型依赖结尾的 `?` 或 `.`。
- **句中难负例。** 每个故事取 2 个窗口，结尾处听起来像一句完整的话，但实际还没讲完（有时窗口从句中开始），标签为 LISTEN。没有这类样本时，模型会学成"句子完整 ⇒ SPEAK"，在停顿处抢话（见下文 r2 → r3）。
- **防止评测泄漏。** 与 FDB 评测文本有重叠的场景全部剔除。

数据以**预分词**格式写出（`tokens` + `prompt_mask`），chat template 与 `areno serve --disable-thinking` 渲染的完全一致，保证训练和推理看到相同的 prompt。之所以要预分词，是因为 AReno 的原始文本 SFT 路径不识别 Ling 的 `<role>` 标记，会把 prompt 再包一层。

产出：

- `train.jsonl`：32,423 条（LISTEN 12,521 / IGNORE 7,635 / SPEAK 7,192 / STOP 3,951 / BACKCHANNEL 1,124）
- `val.jsonl`：1,662 条

### 4. LoRA SFT

```bash
EPOCHS=2 TP=4 SAVE_INTERVAL=200 bash train/train_lora.sh $M data/train.jsonl ckpt/lora-r3 0,1,2,3,4,5,6,7
```

训练配置见 [`train/train_lora.sh`](train/train_lora.sh)：

| 超参 | 取值 |
| :--- | :--- |
| LoRA rank / alpha | 32 / 64（q、k、v、o、gate、up、down 投影） |
| Batch size / micro batch | 64 / 16 |
| Epochs | 2（1014 步） |
| 学习率 | 2e-4 → 2e-5，cosine |
| 最大 prompt / response 长度 | 512 / 96 |
| 并行 | 8× H20 上 TP=4 × DP=2，约 4.7 s/步（总计约 80 分钟） |

Loss 只在 response 部分（`[ACTION] 回复`）上计算，最终 loss 约 0.32。AReno 的 SFT trainer 只在 `(step + 1) % save_interval == 0` 时保存。设置 `SAVE_INTERVAL=200` 会保留 `step_000200 … step_001000`，用于挑选 checkpoint。**发布的 adapter 是 `step_000800`。**

### 5. 部署与评测

```bash
# 基线
bash scripts/serve.sh $M 0,1,2,3 8100 /tmp/serve-base.log --disable-thinking
python eval/run_eval.py --base-url http://127.0.0.1:8100 --data data/fdb_eval.jsonl --out results/baseline_nothink.jsonl

# 微调后（必须显式传 LoRA rank：areno serve 默认是 8）
bash scripts/serve.sh $M 4,5,6,7 8101 /tmp/serve-lora.log --disable-thinking \
  --lora-rank 32 --lora-alpha 64 --lora-adapter-path ckpt/lora-r3/step_000800
python eval/run_eval.py --base-url http://127.0.0.1:8101 --data data/fdb_eval.jsonl --out results/lora_r3_step_000800.jsonl
```

`run_eval.py` 使用贪心解码（temperature 0，最多 128 token），逐条预测写入 `*.jsonl`，指标写入 `*.report.json`。加 `--score-only` 可以对已有的 jsonl 重新打分，不再发请求。

### 6. 推理示例

```bash
python infer.py --base-url http://127.0.0.1:8101 --demo
```

以下是发布 adapter（`step_000800`）的真实输出：

```text
[silent  ] user: 'Hey, can you help me plan a'
           -> LISTEN
[silent  ] user: 'Hey, can you help me plan a weekend trip to Kyoto?'
           -> SPEAK  "Sure, I can help you plan a trip to Kyoto. Let's start by picking the dates and your budget."
[speaking] user: 'yeah'
           -> IGNORE
[speaking] user: 'honey, can you grab my keys?'
           -> IGNORE
[speaking] user: 'wait, actually make it Osaka instead'
           -> STOP  'No problem, Osaka is a great choice too.'
[silent  ] user: 'So last time I went to Japan I got completely lost in the subway, and then'
           -> LISTEN
[silent  ] user: 'So last time I went to Japan I got completely lost in the subway, and then a stranger walked me all the way to my hotel.'
           -> SPEAK  "That's a great story, I bet you had a lot of fun getting back on the right train."
```

单次决策用 `python infer.py --agent-state speaking --history "..." --user "..." --silence 0.3`。任何 OpenAI 兼容的推理服务，只要按 Ling 的 chat template 关闭思考，并加载这个 PEFT adapter，用法都一样。

## 结果

### 指标

| 指标 | 定义 | 方向 |
| :--- | :--- | :---: |
| accuracy | 预测 ∈ 可接受标签 | ↑ |
| macro F1 | 5 个动作 F1 的平均（严格：BACKCHANNEL 必须预测为 BACKCHANNEL） | ↑ |
| 抢话率（句中） | 标签为 LISTEN/BACKCHANNEL，却预测 SPEAK/STOP：助手插嘴 | ↓ |
| 误回应率 | 标签为 IGNORE，却预测 SPEAK/STOP：回应了不是说给它的话 | ↓ |
| 话轮结束回应率 | 标签为 SPEAK，预测 SPEAK/STOP | ↑ |
| 打断停止率 | 标签为 STOP，预测 STOP | ↑ |
| 延迟 | 每次请求的平均耗时（动作 + 回复），4× H20、16 并发 | ↓ |

### 基线 vs. 微调（Full-Duplex-Bench，1862 个决策点）

| 指标 | 基座（关思考） | 基座（开思考） | LoRA r1 | LoRA r2 | **LoRA r3 @800（发布）** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| accuracy ↑ | 0.581 | 0.445 | 0.806 | 0.857 | **0.885** |
| macro F1 ↑ | 0.374 | 0.380 | 0.707 | **0.810** | 0.737 |
| 抢话率（句中）↓ | **0.023** | 0.088 | 0.089 | 0.155 | 0.081 |
| 误回应率 ↓ | 0.336 | 0.530 | **0.000** | 0.007 | 0.007 |
| 话轮结束回应率 ↑ | 0.394 | 0.301 | 0.735 | **0.811** | 0.770 |
| 打断停止率 ↑ | 0.625 | 0.430 | 0.530 | **0.980** | 0.965 |
| 解析失败率 ↓ | 0.001 | 0.343 | 0 | 0 | 0 |
| 平均输出 token | 39.5 | 172.4 | 10.7 | 11.5 | 11.4 |
| 延迟（s）↓ | 1.92 | 3.49 | **1.20** | 1.43 | 1.34 |

各轮实验：

- **r1：** 1,200 个场景（8 字段），训练 3 epoch。
- **r2：** 新增 1,482 个含话题切换、陈述句、短故事的场景，并加入 ASR 风格噪声。
- **r3：** 加入句中难负例，训练 2 epoch，每 200 步保存一次。

### 分来源准确率

| 来源（标签） | 数量 | 基座 | r1 | r2 | r3 @800 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| candor_pause_handling (LISTEN) | 277 | 0.953 | 0.913 | 0.722 | 0.924 |
| synthetic_pause_handling (LISTEN) | 137 | 0.964 | 0.985 | 0.964 | 0.978 |
| synthetic_pause_handling (SPEAK) | 137 | 0.051 | 0.190 | 0.533 | 0.453 |
| icc_backchannel (BACKCHANNEL / LISTEN) | 315 | 0.984 | 0.768 | 0.813 | 0.886 |
| v1.5 user_interruption (STOP) | 200 | 0.625 | 0.530 | 0.980 | 0.965 |
| v1.5 user_backchannel (IGNORE) | 98 | 0.010 | 1.000 | 1.000 | 1.000 |
| v1.5 talking_to_other (IGNORE) | 100 | 0.000 | 1.000 | 1.000 | 1.000 |
| v1.5 background_speech (IGNORE) | 100 | 0.000 | 1.000 | 0.980 | 0.980 |
| v1.5 各子集的请求 (SPEAK) | 498 | 0.43–0.67 | 0.74–0.99 | 0.76–1.00 | 0.76–0.99 |

### r3 checkpoint 扫描

| 指标 | @200 | @400 | @600 | **@800** | @1000 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| accuracy | 0.840 | 0.850 | 0.873 | **0.885** | 0.882 |
| macro F1 | 0.709 | **0.796** | 0.731 | 0.737 | 0.736 |
| 抢话率 ↓ | 0.218 | 0.115 | **0.053** | 0.081 | 0.060 |
| 话轮结束回应率 ↑ | **0.833** | 0.753 | 0.701 | 0.770 | 0.735 |
| 打断停止率 ↑ | 0.875 | **1.000** | 0.985 | 0.965 | 0.985 |

全部报告见 [`results/`](results/)。

## 分析

- **基座模型"忍不住"。** 关闭思考时，它几乎对所有语音都回应，包括旁人对话和背景人声：误回应率 33.6%，IGNORE 召回只有 0.3%。同时，大多数已说完的话轮它又不接话（回应率 39.4%）。它在 CANDOR / ICC 上准确率高，是因为助手沉默时它默认输出 LISTEN，并不是真的理解了话轮。
- **开思考反而更差。** 它更慢（3.5s，172 token），而且 34% 的输出在 128 token 内没给出动作标签。话轮决策必须快，所以本 recipe 训练和推理都关闭思考。
- **微调解决了"这句话是不是对我说的"。** 每一轮在三个 IGNORE 子集上都接近 100%。r2 和 r3@800 对插话的停止率 ≥96.5%。输出缩短到约 11 token，延迟降低 25–37%。
- **难点在"说完了"和"句中停顿"之间，而且两者此消彼长。**
  - r2 学会了接已说完的话轮（回应率 0.811），但开始在 CANDOR 停顿处抢话（LISTEN 0.722，抢话率 0.155）。
  - r3 加入难负例后（实际未结束、但听起来完整的子句 → LISTEN），CANDOR 回到 0.924，抢话率减半，回应率只小幅下降。
  - checkpoint 扫描呈现的是同一个权衡：@600 抢话率最低（0.053），@200 回应率最高（0.833）。
  - 我们发布 @800：总体准确率最高，抢话率和回应率也比较均衡。如果更看重 macro F1 和响应积极性、能容忍一些插嘴，可以选 @400 或 r2。
- **r2 → r3 的 macro F1 为什么下降。** r3@800 在 ICC 上从不输出 BACKCHANNEL，而是输出 LISTEN（279 / 315）。这在 accuracy 中算对，但在严格 macro F1 中算错。BACKCHANNEL 只占训练数据的 3.5%，而难负例又在非常相似的输入上教模型"继续听"。剩余错误主要是 `synthetic_pause_handling` 中已说完的话轮被判为 LISTEN（129 条 SPEAK → LISTEN）。

## 局限

- **只用文本，静音时长固定。** 模型能看到 ASR 文本和静音时长，但训练和评测中静音都是固定值，所以它没有学会利用停顿长度、韵律或音频。在真实系统中，应把它放在 VAD / endpointer 之后，由后者决定*何时*调用它。
- **这是 Full-Duplex-Bench 的文本级代理评测。** 数字来自由 FDB 标注转换出的决策点，不是 FDB 官方的音频级指标，不应与语音到语音模型的榜单结果直接比较。
- **checkpoint 是在评测集上选的。** r3 的 checkpoint 按 FDB 准确率挑选，因此 r3@800 的数字略偏乐观。r1 / r2 用的都是最终 checkpoint。
- **训练数据为合成的纯英文数据**，由基座模型自己生成，话题和表达的多样性有限，例如用户开启话题切换的说法比较单一。
- **BACKCHANNEL 输出偏少**（见分析）。如果需要主动应答，可以在 `build_train.py` 中调高 `P_BACKCHANNEL`，或者使用 r2。

## 权重与数据

| 产物 | 链接 |
| :--- | :--- |
| LoRA adapter（r3 `step_000800`，PEFT，约 2.4 GB） | [Dayuxiaoshui/Ling-3.0-tiny-Full-Duplex-Brain-LoRA](https://huggingface.co/Dayuxiaoshui/Ling-3.0-tiny-Full-Duplex-Brain-LoRA) |
| 训练数据（场景 + `train.jsonl` / `val.jsonl`） | [Dayuxiaoshui/Ling-3.0-tiny-Full-Duplex-Brain-SFT](https://huggingface.co/datasets/Dayuxiaoshui/Ling-3.0-tiny-Full-Duplex-Brain-SFT) |
| 基座模型 | [inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) |
| 评测来源 | [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)；用步骤 1 重建 `fdb_eval.jsonl` |

## AI 使用策略报告

整个项目是和 AI 编程 agent（Claude Code）协作完成的，agent 通过 SSH 直接在训练服务器上工作。本目录的大部分代码由 agent 编写，包括 FDB → 决策点转换、场景生成、数据展开、训练 / 部署封装和评测脚本。它还负责启动和监控训练与评测，并起草了本文档。人负责定义任务、在每轮实验后决定方向（r2 补数据；r3 加难负例并做 checkpoint 扫描），以及核查结果。最有价值的贡献，都建立在实际阅读代码和数据之上，而不是依赖泛泛的常识：

- 通过阅读 AReno 源码，发现原始文本 SFT 路径不识别 Ling 的 `<role>` 标记，由此改用预分词数据格式。
- 发现 SFT trainer 训练结束时不会自动保存。
- 读 r2 的混淆矩阵时发现 CANDOR 停顿被判成 SPEAK，并提出用句中难负例来修复。
- 提出用 checkpoint 扫描把抢话率 / 回应率的权衡显式呈现出来。
- 提出固定静音时长，防止模型靠停顿长短走捷径。

也有一些建议或操作不可靠，需要人工发现并纠正：

- **示例被照抄。** 最初的生成 prompt 带了具体示例句，基座模型在大量场景里原样照抄。后来删掉了示例，并要求变换开头说法。
- **误杀自身进程。** 它曾两次因为 `pkill -f` 的匹配模式过宽，杀掉了自己的 SSH 会话。更早的一个会话还卡在 20 分钟的 `sleep` 上，失去了响应。
- **只看 accuracy 会掩盖问题。** 如果只报告 accuracy，就看不出 r3 从不输出 BACKCHANNEL。为此补充了严格 macro F1 和混淆矩阵。
- **在评测集上选 checkpoint。** 按 FDB 准确率挑 r3 的 checkpoint，本质上是在测试集上调参。我们在[局限](#局限)中如实说明，而不是隐去。

我们的体会：agent 能大幅加快工程迭代，但每一个关于模型效果的结论，都仍需对照分来源指标和原始预测逐条核查。
