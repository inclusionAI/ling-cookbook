---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

- 感谢 [@Dayuxiaoshui](https://github.com/Dayuxiaoshui) 提供此微调用例。

Licensed under the MIT License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://opensource.org/licenses/MIT

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

+++

# 全双工语音对话：LoRA 微调 Ling-3.0-tiny 做话轮管理

<table align="left">
  <td>
    <a target="_blank" href="https://www.ant-ling.com/"><img src="https://img.shields.io/badge/Official_Website-ant--ling.com-6366f1?style=flat-square" alt="Website" /></a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/inclusionAI/ling-cookbook"><img src="https://img.shields.io/badge/GitHub-Ling_Cookbook-181717?style=flat-square&logo=github" alt="GitHub" /></a>
  </td>
  <td>
    <a target="_blank" href="https://huggingface.co/inclusionAI"><img src="https://img.shields.io/badge/Hugging_Face-inclusionAI-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face" /></a>
  </td>
  <td>
    <a target="_blank" href="https://www.modelscope.cn/organization/inclusionAI"><img src="https://img.shields.io/badge/ModelScope-inclusionAI-624AFF?style=flat-square" alt="ModelScope" /></a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/inclusionAI/AReno"><img src="https://img.shields.io/badge/GitHub-AReno-181717?style=flat-square&logo=github" alt="AReno" /></a>
  </td>
</table>

<br><br>

+++

## 1. 这个任务会做什么

全双工语音助手需要同时处理语音输入与输出。除了生成回复内容，系统还需要实时判断话轮动作：用户句中停顿时继续等待，用户长篇讲述时给出简短应答，用户说完时接话，被打断时停止当前输出，对非指向助手的语音保持忽略。

本教程使用 LoRA 将总参数量 7.9B、单 token 激活参数量 1.3B 的 MoE 模型 [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) 微调为话轮决策与回复生成模型。在每个决策时刻，模型读取实时 ASR 文本，输出一个动作标签；当需要接话或应答时，再跟一句简短回复：

| 动作 | 触发场景 |
| :--- | :--- |
| LISTEN | 用户出现句中停顿、犹豫或话未说完时，保持沉默。 |
| BACKCHANNEL | 用户在长段讲述中，给出简短应答而不打断话轮。 |
| SPEAK | 用户说完了一段对助手说的话，接话回复。 |
| STOP | 助手正在说话时用户插话提出新请求或修改请求，停止当前播报并回应新请求。 |
| IGNORE | 简短应答词、用户与旁人交谈或背景人声，不需要回应。 |

评测集由 [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) v1.0 / v1.5 的标注转换为决策点。最佳 checkpoint 的效果如下：

- 准确率从 0.581 提升至 0.885
- macro F1 从 0.374 提升至 0.737
- 应忽略语音的误回应率从 33.6% 降至 0.7%
- 被打断时的停止率从 62.5% 提升至 96.5%

训练全程未使用任何 Full-Duplex-Bench 数据。

+++

## 2. 注意事项

在运行或复用本教程的训练与评测流程前，请注意以下任务边界与实验局限：

- 模型仅接收 ASR 文本和固定静音时长，未学习停顿长短、韵律或音频特征。在实际系统中应将其置于 VAD 或端点检测模块之后，由前置模块决定何时触发调用。
- 文中评测数据是由 Full-Duplex-Bench 标注转换出的文本级决策点，并非官方音频级指标，不宜与端到端语音模型的榜单结果直接横向比较。
- 第三轮实验 r3 的最佳权重是按评测集准确率挑选的第 800 步检查点，因此指标略偏乐观；前两轮 r1 和 r2 均直接采用训练结束时的最终权重。
- 训练数据全部由基座模型合成且仅为英文，话题与表达的多样性有限，例如用户开启话题切换的句式较为单一。
- 模型较少主动输出 BACKCHANNEL。如果业务场景需要更频繁的简短附和，可以在 `data/build_train.py` 中调高 `P_BACKCHANNEL` 参数，或改用 r2 权重。

+++

## 3. 模型与微调框架介绍

本任务基于 Ling-3.0-tiny 与百灵后训练框架 [AReno](https://github.com/inclusionAI/AReno) 完成训练、推理服务部署与评测。相关的基座模型、微调适配器权重、训练数据集与评测来源汇总如下：

| 产物 | 链接 |
| :--- | :--- |
| LoRA 适配器权重，PEFT 格式，约 2.4 GB | [Dayuxiaoshui/Ling-3.0-tiny-Full-Duplex-Brain-LoRA](https://huggingface.co/Dayuxiaoshui/Ling-3.0-tiny-Full-Duplex-Brain-LoRA) |
| 合成场景与训练验证集 | [Dayuxiaoshui/Ling-3.0-tiny-Full-Duplex-Brain-SFT](https://huggingface.co/datasets/Dayuxiaoshui/Ling-3.0-tiny-Full-Duplex-Brain-SFT) |
| 基座模型 | [inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) |
| 评测集来源 | [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)，可通过第 6 节步骤 1 生成 `fdb_eval.jsonl` |

+++

## 4. 设备要求

复现本任务的完整训练与并发评测所使用的软硬件环境如下：

| 项目 | 使用版本 |
| :--- | :--- |
| GPU | 8× NVIDIA H20 96GB，训练使用 8 卡，推理使用 4 卡 |
| 训练与推理框架 | AReno，commit 3cfa480 |
| 基座模型 | inclusionAI/Ling-3.0-tiny |
| 评测数据 | Full-Duplex-Bench v1.0 与 v1.5 文本标注 |
| 脚本依赖 | httpx、transformers |

表格中的 8× H20 是作者原始实验服务器配置。如果你希望在消费级设备上进行实验，可以参考最后一节的「[轻量级设备要求](#lightweight-hardware)」。

如果 `areno` 不在系统 PATH 中，请先设置环境变量 `ARENO_VENV=/path/to/venv`。以下所有命令均在本目录下执行。

```{code-cell} ipython3
%%bash
export M=/path/to/Ling-3.0-tiny        # local snapshot of inclusionAI/Ling-3.0-tiny
export FDB=/path/to/Full-Duplex-Bench  # directory containing v1.0/ and v1.5/
```

+++

## 5. 任务分析

下面从单次决策时刻的输入输出格式与本目录配套源码结构两方面说明任务设计。

### 任务定义

一次请求对应一个决策时刻。输入包含三部分：助手当前状态、用户到目前为止的 ASR 文本，以及用户最后一个词之后的静音时长：

```text
Assistant state: speaking (currently answering: "Can you help me plan a weekend trip to Kyoto?")
New user speech: "wait, actually make it Osaka instead"
Silence since the last word: 0.3s
Action:
```

输出以动作标签开头；当动作为 SPEAK 或 STOP 时后接具体回复，当动作为 BACKCHANNEL 时后接简短应答语：

```text
[STOP] No problem, Osaka is a great choice too.
```

系统提示词和输入格式定义在 [`fdb_brain/prompt.py`](fdb_brain/prompt.py) 中，由数据构造、评测和推理阶段共用。将动作标签放在输出最前面，便于下游控制器在生成前几个 token 时立即执行动作，无需等待完整回复生成结束。

训练和评测阶段均采用固定静音时长：助手沉默时设为 0.8s，助手说话时设为 0.3s。这样模型无法根据停顿时长直接判断标签，必须依据用户输入的文本语义来区分句中停顿与话轮结束。这也是文本模型在 VAD 之上所能提供的语义判别信息。

### 文件结构

```text
full-duplex-turn-taking-lora/
├── fdb_brain/prompt.py        # shared system prompt, input rendering, action parser
├── data/
│   ├── build_fdb_eval.py      # Full-Duplex-Bench annotations -> 1862 eval decision points
│   ├── synth_scenarios.py     # generate dialogue scenarios with the base model
│   └── build_train.py         # scenarios -> labelled, pre-tokenized SFT rows
├── train/
│   ├── train_lora.sh          # AReno LoRA SFT (training config)
│   └── sft_loader.py          # dataset loader passed to `areno train`
├── scripts/serve.sh           # start an OpenAI-compatible `areno serve` (optionally with LoRA)
├── eval/run_eval.py           # query + score; writes per-point jsonl and a report json
├── infer.py                   # inference example (single decision or scripted call)
└── results/*.report.json      # evaluation reports of every run in the tables below
```

+++

## 6. 操作步骤

按照以下步骤可依次完成评测集构建、合成训练场景生成、带标签决策点展开、LoRA SFT 训练、推理服务部署评测以及单次推理测试。

### 步骤 1：构造评测集

```{code-cell} ipython3
%%bash
python data/build_fdb_eval.py --fdb-root $FDB --out data/fdb_eval.jsonl
```

每条 Full-Duplex-Bench 样本会转换为一个或多个文本级决策点，共计 1862 条：

| FDB 子集 | 决策点 | 标签 | 数量 |
| :--- | :--- | :--- | ---: |
| v1.0 candor_pause_handling | 截至每个标注 PAUSE 处的文本 | LISTEN | 277 |
| v1.0 synthetic_pause_handling | 截至每个停顿处的文本 / 完整话轮 | LISTEN / SPEAK | 137 / 137 |
| v1.0 icc_backchannel | 长独白中截至每个 ≥0.3s 间隙处的文本 | BACKCHANNEL 或 LISTEN | 315 |
| v1.5 user_interruption | 助手沉默时的请求 / 助手说话时的插话 | SPEAK / STOP | 200 / 200 |
| v1.5 user_backchannel | 正常请求 / 简短应答 | SPEAK / IGNORE | 98 / 98 |
| v1.5 talking_to_other | 正常请求 / 用户与旁人交谈 | SPEAK / IGNORE | 100 / 100 |
| v1.5 background_speech | 正常请求 / 背景人声 | SPEAK / IGNORE | 100 / 100 |

+++

### 步骤 2：用基座模型生成训练场景

训练数据全部由模型合成。先以关闭思考模式部署基座模型，再调用该服务生成对话场景。每个场景对应一组话题与人设组合，包含请求与回复、插话与回复、旁人对话、背景人声、长故事、话题切换等 14 个字段，具体定义见 [`data/synth_scenarios.py`](data/synth_scenarios.py)。

```{code-cell} ipython3
%%bash
bash scripts/serve.sh $M 0,1,2,3 8100 /tmp/serve-base.log --disable-thinking
python data/synth_scenarios.py --base-url http://127.0.0.1:8100 --n 1500 --seed 1 --out data/scenarios_r2.jsonl
python data/synth_scenarios.py --base-url http://127.0.0.1:8100 --n 800  --seed 2 --out data/scenarios_r2b.jsonl
```

格式不合规范的生成结果会被过滤，两次采样最终分别保留 984 条和 498 条。发布的模型还结合了早期 8 字段提示词生成的 1200 条场景 `data/scenarios.jsonl`。上述场景均已包含在公开数据集中，可直接下载使用。

+++

### 步骤 3：场景展开为带标签的决策点

```{code-cell} ipython3
%%bash
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

完整展开规则见 [`data/build_train.py`](data/build_train.py) 的文件注释。数据构造包含三项关键处理：

- 引入 ASR 风格噪声：用户文本以各 35% 的概率随机去除标点或插入 UM、UH 等填充词，避免模型单纯依赖句尾标点符号判断话轮结束。
- 构造句中难负例：从每个长故事中截取 2 个片段，使片段结尾在语法上看似完整、但实际叙述尚未结束，并将标签设为 LISTEN。若缺少此类样本，模型容易将语法完整的子句直接判为 SPEAK，导致在用户句中停顿处提前抢话。
- 过滤评测集重叠文本：剔除所有与 Full-Duplex-Bench 评测文本存在重叠的合成场景，防止评测泄漏。

由于 AReno 的原始文本 SFT 路径不识别 Ling 模型的角色标记，直接传入文本会导致模板被重复包裹。因此脚本按 `areno serve --disable-thinking` 一致的对话模板预先分词，输出包含 token 序列与掩码的预分词文件，最终生成 32,423 条训练样本 `train.jsonl`（其中 LISTEN 12,521 条、IGNORE 7,635 条、SPEAK 7,192 条、STOP 3,951 条、BACKCHANNEL 1,124 条）和 1,662 条验证样本 `val.jsonl`。

+++

### 步骤 4：LoRA SFT

```{code-cell} ipython3
%%bash
EPOCHS=2 TP=4 SAVE_INTERVAL=200 bash train/train_lora.sh $M data/train.jsonl ckpt/lora-r3 0,1,2,3,4,5,6,7
```

训练配置见 [`train/train_lora.sh`](train/train_lora.sh)：

| 超参 | 取值 |
| :--- | :--- |
| LoRA rank / alpha | 32 / 64，作用于 q、k、v、o、gate、up、down 投影 |
| Batch size / micro batch | 64 / 16 |
| Epochs | 2，共 1014 步 |
| 学习率 | 2e-4 → 2e-5，cosine |
| 最大 prompt / response 长度 | 512 / 96 |
| 并行配置 | 8× H20 上采用 TP=4、DP=2，单步约 4.7 秒，总计约 80 分钟 |

训练损失仅在动作标签与回复部分上计算，最终收敛至约 0.32。AReno 的训练器仅在步数满足保存间隔时写出检查点，设置 `SAVE_INTERVAL=200` 会依次保留第 200 至 1000 步的权重。最终发布的权重为第 800 步检查点 `step_000800`。

+++

### 步骤 5：部署与评测

```{code-cell} ipython3
%%bash
# baseline
bash scripts/serve.sh $M 0,1,2,3 8100 /tmp/serve-base.log --disable-thinking
python eval/run_eval.py --base-url http://127.0.0.1:8100 --data data/fdb_eval.jsonl --out results/baseline_nothink.jsonl

# fine-tuned (pass the LoRA rank explicitly: areno serve defaults to 8)
bash scripts/serve.sh $M 4,5,6,7 8101 /tmp/serve-lora.log --disable-thinking \
  --lora-rank 32 --lora-alpha 64 --lora-adapter-path ckpt/lora-r3/step_000800
python eval/run_eval.py --base-url http://127.0.0.1:8101 --data data/fdb_eval.jsonl --out results/lora_r3_step_000800.jsonl
```

评测脚本 `eval/run_eval.py` 采用 temperature 为 0、最大生成长度 128 token 的贪心解码，将逐条预测写入 jsonl 文件，并将汇总指标写入对应的 report.json 文件。若仅需对已有预测结果重新计算指标，可附加 `--score-only` 参数。

+++

### 步骤 6：推理示例

```{code-cell} ipython3
%%bash
python infer.py --base-url http://127.0.0.1:8101 --demo
```

以下是加载 `step_000800` 权重后的实际输出：

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

若需执行单次决策，可运行 `python infer.py --agent-state speaking --history "..." --user "..." --silence 0.3`。任何兼容 OpenAI 协议的推理服务，只要按 Ling 对话模板关闭思考并加载该 PEFT 权重，均可按相同方式调用。

+++

## 7. 典型结果

在 Full-Duplex-Bench 转换出的 1862 个决策点上，基座模型与各轮 LoRA 微调实验的评测指标定义、对比结果与行为分析如下。

### 指标

| 指标 | 定义 | 方向 |
| :--- | :--- | :---: |
| accuracy | 预测属于可接受标签 | ↑ |
| macro F1 | 5 个动作 F1 的平均值，BACKCHANNEL 必须严格预测为 BACKCHANNEL | ↑ |
| 抢话率 | 标准标签为 LISTEN 或 BACKCHANNEL，却预测为 SPEAK 或 STOP | ↓ |
| 误回应率 | 标准标签为 IGNORE，却预测为 SPEAK 或 STOP | ↓ |
| 话轮结束回应率 | 标准标签为 SPEAK，预测为 SPEAK 或 STOP | ↑ |
| 打断停止率 | 标准标签为 STOP，预测为 STOP | ↑ |
| 延迟 | 4× H20、16 并发下每次请求的平均耗时 | ↓ |

### 基线与微调效果对比

| 指标 | 基座（关思考） | 基座（开思考） | LoRA r1 | LoRA r2 | LoRA r3 @800（发布） |
| :--- | ---: | ---: | ---: | ---: | ---: |
| accuracy ↑ | 0.581 | 0.445 | 0.806 | 0.857 | 0.885 |
| macro F1 ↑ | 0.374 | 0.380 | 0.707 | 0.810 | 0.737 |
| 抢话率 ↓ | 0.023 | 0.088 | 0.089 | 0.155 | 0.081 |
| 误回应率 ↓ | 0.336 | 0.530 | 0.000 | 0.007 | 0.007 |
| 话轮结束回应率 ↑ | 0.394 | 0.301 | 0.735 | 0.811 | 0.770 |
| 打断停止率 ↑ | 0.625 | 0.430 | 0.530 | 0.980 | 0.965 |
| 解析失败率 ↓ | 0.001 | 0.343 | 0 | 0 | 0 |
| 平均输出 token | 39.5 | 172.4 | 10.7 | 11.5 | 11.4 |
| 延迟（s）↓ | 1.92 | 3.49 | 1.20 | 1.43 | 1.34 |

各轮实验配置：

- r1：使用 1,200 个 8 字段场景，训练 3 个 epoch。
- r2：新增 1,482 个包含话题切换、陈述句和短故事的场景，并加入 ASR 风格噪声。
- r3：加入句中难负例，训练 2 个 epoch，每 200 步保存一次检查点。

### 分来源准确率

| 来源与标准标签 | 数量 | 基座 | r1 | r2 | r3 @800 |
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

### r3 检查点扫描

| 指标 | @200 | @400 | @600 | @800 | @1000 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| accuracy | 0.840 | 0.850 | 0.873 | 0.885 | 0.882 |
| macro F1 | 0.709 | 0.796 | 0.731 | 0.737 | 0.736 |
| 抢话率 ↓ | 0.218 | 0.115 | 0.053 | 0.081 | 0.060 |
| 话轮结束回应率 ↑ | 0.833 | 0.753 | 0.701 | 0.770 | 0.735 |
| 打断停止率 ↑ | 0.875 | 1.000 | 0.985 | 0.965 | 0.985 |

全部报告见 [`results/`](results/)。

### 分析

- 在未经微调时，基座模型在这个任务上倾向于过度触发回应，且缺乏话轮边界判别能力。关闭思考时，模型对旁人对话和背景人声等非指向性语音也会触发回复，误回应率达 33.6%，IGNORE 召回率仅为 0.3%；同时对大多数已结束的话轮又未能及时接话，话轮结束回应率仅为 39.4%。基座模型在 CANDOR 和 ICC 子集上准确率较高，主要是因为助手处于沉默状态时默认偏向输出 LISTEN，而非真正识别了话轮状态。
- 关闭思考反而综合效果更好。开启思考时平均输出长度达 172 token、单次耗时 3.49s，且有 34.3% 的输出在 128 token 内未能给出动作标签。话轮决策要求低延迟，因此本任务在训练和推理阶段均关闭思考。
- 微调显著改善了非指向性语音过滤与打断响应。各轮微调模型在三个 IGNORE 子集上的准确率均接近 100%，r2 和 r3@800 对插话的停止率达到 96.5% 以上，平均输出缩短至约 11 token，端到端延迟降低 25% 至 37%。
- 主要难点在于区分话轮结束与句中停顿，且两项指标存在权衡关系：
  - r2 提升了话轮结束回应率至 0.811，但在 CANDOR 句中停顿处的准确率降至 0.722，抢话率升至 0.155。
  - r3 引入句中难负例后，CANDOR 准确率恢复至 0.924，抢话率减半至 0.081，话轮结束回应率仅小幅回落至 0.770。
  - 不同检查点呈现出相同的权衡趋势：第 600 步抢话率最低，为 0.053；第 200 步回应率最高，为 0.833。最终发布的第 800 步检查点总体准确率最高，且在抢话率与回应率之间较为均衡；若更看重 macro F1 和响应积极性，可选用第 400 步检查点或 r2。
- 难负例对 BACKCHANNEL 与严格 macro F1 的影响：r3@800 在 ICC 子集的 315 条样本上有 279 条输出 LISTEN 而非 BACKCHANNEL。这在宽松准确率下计为正确，但在严格 macro F1 下计为错误。原因在于 BACKCHANNEL 仅占训练数据的 3.5%，而新增的句中难负例在相似的未完结语境下将模型监督为输出 LISTEN。其余误差主要集中在 synthetic_pause_handling 子集中，有 129 条已结束话轮被保守预测为 LISTEN。

+++

## 8. 常见问题与进阶方向

结合本任务的复现过程与实验记录，以下汇总消费级硬件适配方案与常见调优注意事项。

<a id="lightweight-hardware"></a>

### 轻量级设备要求

作者原始实验使用 8× H20 是为了在多卡并行与微批次 16 的配置下以约 80 分钟跑完两轮共 6.48 万条样本。由于本任务单次输入上限仅 512 token、平均输出仅 11.4 token，激活值与 KV Cache 开销很小，在单卡消费级 GPU 或 Apple Silicon Mac 上均可按以下两档方案复现：

- 复现全精度实验所需的硬件与改动：
  - 硬件要求：BF16 基座约占 15.8GB，Rank 32 LoRA 权重约占 1.18GB，短序列 KV Cache 小于 0.5GB，因此全精度推理与评测需要至少 24GB 统一内存的 Mac 或单张 24GB 显存的 GPU。全精度训练除冻结基座外，还需约 8.3GB 存放 5.92 亿可训练参数的梯度与优化器状态，以及约 2GB 存放微批次激活值，总内存峰值约 25GB 至 27GB，需要至少 36GB 统一内存的 Mac 或单张 32GB 以上显存的 GPU。
  - 必要改动：推理时将 `scripts/serve.sh` 中的 `--tp-size` 和 `--world-size` 改为 1，在 Mac 上指定 `--backend mlx`，并将并发数 `--max-running-prompts` 调小为 2 或 4。训练时在 `train/train_lora.sh` 中将 `--tp-size` 和 `--world-size` 改为 1，并将微批次 `--mini-bs` 从 16 调小为 1 或 2，保持 `--batch-size 64` 由梯度累积完成，即可在不改变有效批次大小与模型精度的前提下运行。

- 复现实验所需的最轻量级硬件与改动：
  - 硬件要求：4-bit 量化后模型权重约占 4.5GB，推理与评测仅需 8GB 至 16GB 统一内存的 Mac 或 8GB 显存的 GPU。采用 4-bit QLoRA 与 Rank 8 训练时，参数与优化器状态约占 2.1GB，单样本激活值约占 2.5GB，总内存峰值约 9GB 至 10GB，在 16GB 统一内存的 Mac 或 12GB 至 16GB 显存的 GPU 上即可运行。
  - 必要改动：在 `areno train` 中加入 `--q4` 将冻结基座量化为 4-bit，将 LoRA 参数下调为 `--lora-rank 8 --lora-alpha 16`，推理服务部署时也传入相同的 rank 和 alpha，并将批次参数改为 `--batch-size 1 --mini-bs 1`，Mac MLX 环境下可同时加上 `--no-mlx-compile-train-step`。由于完整训练集包含 32,423 条样本，单机训练时建议按动作类别分层抽样 3,000 至 5,000 条样本或设置 `--max-steps 1000`，将单机训练时间控制在 2 至 4 小时内。

### 常见问题与调优方向

- 启动 `areno serve` 加载 LoRA 时必须显式传入 `--lora-rank 32 --lora-alpha 64`：`areno serve` 默认的 LoRA rank 为 8，若未显式指定，将与训练时的 rank 32 配置不匹配。
- 为什么训练数据需要预分词写出：AReno 的原始文本 SFT 路径不识别 Ling 模型的角色标记，直接传入原始文本会把提示词重复包裹，因此 `data/build_train.py` 先按 `areno serve --disable-thinking` 一致的对话模板渲染并写出预分词序列。
- 如何提高 BACKCHANNEL 触发率或调整抢话与响应的权衡：若业务场景需要更积极的应答，可在 `data/build_train.py` 中调高 `P_BACKCHANNEL` 重新构建训练集，或直接选用第 400 步检查点及 r2 权重。
