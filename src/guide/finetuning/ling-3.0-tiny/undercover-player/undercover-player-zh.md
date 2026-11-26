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

##### Copyright 2026 Ant Group.

感谢 [@lplalbert](https://github.com/lplalbert) 提供此微调用例。

+++

# 用 MLX 端侧 QLoRA 微调 Ling-3.0-tiny 玩「谁是卧底」

在 Apple Silicon（验证机型：M4 MacBook Air 16GB）上把 `inclusionAI/Ling-3.0-tiny`
微调为社交推理桌游「谁是卧底」的**玩家决策组件**：输入身份、词语与轮次历史，
输出 ≤25 字发言与 `我投N号`/`我弃权` 投票。微调产物同时是
[Avernet #2611](https://github.com/inclusionAI/Avernet/issues/2611)「谁是卧底」
牌桌的 **Apple MLX 本地端侧模型引擎** 玩家 Bot 的推理内核——一个项目，两边交付。

+++

## 1. 这个任务会做什么

- **数据**：30 组词对 × 属性池 × 模板**程序化合成**千余条对局样本（可复现、无泄露，
  6 组留出词对测泛化），并支持两条增量轨道：实战对局飞轮（卧底身份 = 上帝视角
  真值标签）与教师大模型离线蒸馏（**绝不进游戏循环**）。
- **训练**：4-bit 底座 + LoRA（rank 8，后 16/24 层，MoE 专家冻结，可训练 104M/1.3%），
  全程 16GB 内存内完成。
- **评测**：微调前后对比——发言字数合规率、泄词率、自然度、投票格式合法率、
  投票规则标签匹配率、**自对弈胜率**（6 席全模型 + 脚本裁判）。
- **应用**：`mlx_lm.server` 起 OpenAI 兼容服务，经 BCN V3 协议接入 Avernet
  牌桌；运行时清洗保障输出格式，不代表策略质量。

+++

## 2. 注意事项

1. **量化必须本地做**：ModelScope 的 int4 预量化是 vLLM 专用，MLX 用不了；
   用 `mlx_lm.convert` 从 BF16 本地量化。
2. **16G 的训练边界**：QLoRA 连续训练约 625 步后会确定性 `Metal Insufficient
   Memory`——**分段续训**（每段 ≤600 步，`--resume-adapter-file` 接力）。
3. **别开量化 KV**（`--kv-bits 8`）：mlx-lm 的 MLA 实现曾在 >2048 token 提示词上
   崩溃（上游 [#2058](https://github.com/ml-explore/mlx-lm/pull/2058) 已修），
   16G 单实例显存够用，不开也罢。
4. **权重不入 Git**：adapters 托管在
   [GitHub Release](https://github.com/lplalbert/ling-cookbook/releases/tag/undercover-player-adapter-v2)。
5. 贪心解码会让模型坍缩（发言全万能句、投票全弃权）——评测与推理统一用
   固定种子采样（temp=0.7, top_p=0.9）。

+++

## 3. 模型与微调框架介绍

- **底座**：`inclusionAI/Ling-3.0-tiny`（MoE，激活 1.3B，`bailing_hybrid` 架构，
  24 层）。
- **框架**：Apple [MLX](https://github.com/ml-explore/mlx) + `mlx-lm`
  （QLoRA 训练与推理一体，`mask_prompt` 只对输出算 loss，梯度检查点省内存）。
- **LoRA 拓扑**：只挂后 16/24 层的注意力投影，MoE 专家权重全冻结——
  可训练 104M / 1.3%，16G 统一内存跑得动训练也跑得动牌桌。

+++

## 4. 设备要求

| 项 | 版本/值 |
|---|---|
| 硬件 | Apple Silicon Mac（验证机型：M4 Air 16GB） |
| Python | ≥ 3.10（验证：3.13） |
| mlx-lm | ≥ 0.32.0 |
| 磁盘 | 模型 BF16 15G + 量化 4.8G + adapter ~400M |
| 内存 | 训练峰值约 12GB；推理单实例约 5.3GB |

+++

## 5. 任务分析

「谁是卧底」每局 6 人：1 卧底 5 平民，各自拿到相近的词。每轮先轮流发言
（≤25 字描述自己的词、不泄词），再投票（`我投N号`/`我弃权`），出局到只剩
≤2 人或抓出卧底为止。玩家决策拆成两个动作头：

1. **发言**（speak）：给定身份/词语/历史 → 一句 ≤25 字、不泄词、有信息量的描述；
2. **投票**（vote）：给定同样上下文 → 票面。

提示词**窗口化**：历史只保留第 1 轮 + 最近 2 轮（中间折叠成「（第x-y轮略）」），
单次推理输入有界（≤420 字符），训练/推理/飞轮三处同咽喉（`data/prompting.py`），
Avernet 玩家 Bot 内嵌同款渲染、逐字节一致。合成对局深度 1~6 轮对齐实战上限，
让折叠标记在训练集里有覆盖（626/1212 条）——**凡是推理时会折叠/截断输入的规则，
训练数据必须覆盖该变换的全部取值**。

+++

## 6. 操作步骤

<a id="setup"></a>

### 步骤 1：环境准备与模型量化

```{code-cell}
%%bash
# 克隆仓库并进入 recipe 目录
git clone https://github.com/inclusionAI/ling-cookbook.git
cd ling-cookbook/guide/finetuning/ling-3.0-tiny/undercover-player

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 下载 Ling-3.0-tiny（BF16 约 15G）到 ~/models/Ling-3.0-tiny（ModelScope / HF），
# 然后本地量化为 4-bit——MLX 训练/推理都用这份：
python -m mlx_lm.convert --model ~/models/Ling-3.0-tiny \
    --q-bits 4 --mlx-path ~/models/Ling-3.0-tiny-4bit --trust-remote-code
```

> 量化后若报 `StrictDataclassFieldValidationError: Field 'kda_lower_bound' expected float`，
> 把 `config.json` 里的 `-5` / `0` 改成 `-5.0` / `0.0`（官方配置的整数字面量踩了
> MLX 严格校验，详见 README Troubleshooting #1）。

<a id="data"></a>

### 步骤 2：生成训练数据集（程序化合成）

```{code-cell}
%%bash
python data/synthesize.py   # 固定种子，生成 train/valid/test.jsonl（1212/76/96 条）
```

合成规则：随机配词对 → 分配身份（1 卧底 / 5 平民）→ 按属性池模板生成每轮发言
（卧底刻意模糊，平民用判别属性）→ 规则式打投票标签（平民投嫌疑最大的卧底、
卧底甩锅/弃权）。投票样本过采样 3 倍平衡任务难度（train 1212 = 432 speak + 780 vote），
valid/test 保持自然分布。

<a id="train"></a>

### 步骤 3：QLoRA 微调

```{code-cell}
%%bash
cd train && bash train.sh    # QLoRA 1500 步 ≈ 2.5 小时（M4 16GB），产物在 adapters/
# 等价：python -m mlx_lm lora -c train/lora_config.yaml
```

关键配置：4-bit 底座 + LoRA rank 8、只挂后 16/24 层、batch 1 × 梯度累积 4、
lr 1e-4、`mask_prompt`、梯度检查点。

**16G 分段续训**（连续跑 >~625 步必 Metal OOM）：

```bash
python -m mlx_lm lora ... --iters 600 --adapter-path ../adapters-seg1
python -m mlx_lm lora ... --iters 300 \
    --resume-adapter-file ../adapters-seg1/0000600_adapters.safetensors \
    --adapter-path ../adapters-seg2
```

注意检查点文件名是 **7 位**零填充（`0000600_adapters.safetensors`）；段间优化器态
重置，因此不等价于连续训练；需重新评测，不能直接接受 loss 尖峰。

<a id="eval"></a>

### 步骤 4：微调前后对比评测（含自对弈）

```{code-cell}
%%bash
# 96 条测试集（6 组留出词对），固定种子采样 temp=0.7 top_p=0.9 seed=42
python eval/evaluate.py --model ~/models/Ling-3.0-tiny-4bit \
    --data data/test.jsonl --out eval/base.json
python eval/evaluate.py --model ~/models/Ling-3.0-tiny-4bit --adapter-path adapters \
    --data data/test.jsonl --out eval/lora.json

# 自对弈胜率：6 席全模型 + 脚本裁判（20 局；随机基线 200 局做参照）
python eval/selfplay.py --model ~/models/Ling-3.0-tiny-4bit --adapter-path adapters \
    --games 20 --out eval/selfplay.json
python eval/selfplay.py --model ~/models/Ling-3.0-tiny-4bit --policy random \
    --games 200 --out eval/selfplay-random.json
```

典型输出与对比表见下文「7. 典型结果」；逐轮指标记进 `eval/metrics-history.jsonl`。

### 步骤 5：单条交互 demo

```{code-cell}
%%bash
python infer/chat.py --model ~/models/Ling-3.0-tiny-4bit --adapter-path adapters
```

<a id="flywheel"></a>

### 步骤 6：数据飞轮（可选进阶）

```{code-cell}
%%bash
# 每攒够 ≥5 局新对局跑一轮：
python flywheel.py build --out data/live-vN.jsonl          # 对局状态 -> 实战样本（真值票）
python flywheel.py merge --live data/live-vN.jsonl data/api-distill-vN.jsonl \
    --out data/train-vN.jsonl                              # 实战 + 蒸馏增量一次合并（≤30% 限流）
python data/rerender.py data/live-vN.jsonl data/api-distill-vN.jsonl  # 格式变更后无损重渲染
```

三条采样纪律：真值投票标签是核心增量（DAgger 式）；自我模仿（speak）限流；
留出词对的整局跳过。**评测门槛**（防自我训练坍缩）：`vote_format_ok = 1.0`、
`speak_format_ok` 不低于上一轮、泄词率不升超 2pt——达标才换线上 adapter，
投票规则标签匹配率只观测不设门槛。详见 README「数据飞轮」。

<a id="apply"></a>

### 步骤 7：接入 Avernet「谁是卧底」牌桌（应用侧）

+++

微调模型经 `mlx_lm.server` 起 OpenAI 兼容服务（单实例双角色：底座/LoRA 靠请求级
`adapters` 字段切换），玩家 Bot 以纯 Python BCN V3 WebSocket 适配层入桌。集成代码、
运维手册见 Avernet 侧
[PR #3970](https://github.com/inclusionAI/Avernet/pull/3970) 与
[接入核验报告](eval/verification/INTEGRATION.md)；推理权重见
[Release `undercover-player-adapter-v2`](https://github.com/lplalbert/ling-cookbook/releases/tag/undercover-player-adapter-v2)。

+++

<a id="results"></a>

## 7. 典型结果
> 历史结果：以下数字使用旧版解析与推理协议，不能作为当前默认思考模式的验收结果。当前严格解析的三种子重评见[最终核验报告](eval/verification/REPORT.md)。


test.jsonl（96 条，6 组留出词对），固定种子采样（temp=0.7, top_p=0.9, seed=42）。
窗口化 test 三方对比（48 条触发历史折叠）：

| 指标 | 基线（4-bit） | v2 adapter（收养件） | v3 adapter（窗口化重训） |
|---|---|---|---|
| 发言字数合规率（≤25 字） | 93.3% | **100%** | 88.3% |
| 发言自然度（非思考前缀） | 8.3% | 100% | 100% |
| 发言格式合规率（字数+不泄词+自然） | 3.3% | **93.3%** | 78.3% |
| 发言泄词率（原始输出，越低越好） | 5.0% | **6.7%** | 11.7% |
| 投票格式合法率（`我投N号`/`我弃权`） | 52.8% | **100%** | **100%** |
| 投票规则标签匹配率（观测项） | 11.1% | 25.0% | 19.4% |

门槛判定：v3 未达标（speak_format 0.783、泄词 +5pt），**v2 继续收养**——
门槛机制第一次真实拦截回退。该小样本中 v2 在折叠与未折叠子集上表现接近（折叠/未折叠
speak_format 0.939/0.926），该历史配置中 v2 通过了当时的门槛。

上述自对弈命令为可选实验；本文不主张当前自对弈胜率有所提升。三种子验证及不确定区间见[最终核验报告](eval/verification/REPORT.md)。

+++

## 8. 常见问题与进阶方向

- **原始输出带 `</think>` 前缀**：默认训练会学习闭合空思考块，这不证明训练器偏移错误。
  推理统一 `thinking_option="on"`，用 `clean_output()` 提取思考结束后的正文。
- **贪心解码模式坍缩**（投票全弃权/发言全万能句）：固定种子采样 + 线上清洗层兜底。
- **投票规则标签匹配率天花板**：规则式标签与输入可观测性均需进一步核查（三步归因实验见 README）；
  进阶方向 = 数据飞轮真值票 + 教师蒸馏票（谁买到判别力？见 README 消融）。
- **训练/推理分布对齐**：凡是推理时会变换/折叠/截断输入的规则，训练数据必须
  覆盖该变换的全部取值（Troubleshooting #9 的 OOD 教训）。
- 完整踩坑记录（9 条）见 [README.md](README.md) 的 Troubleshooting。

+++

## 9. 致谢

- 感谢蚂蚁开源轻训营与 `inclusionAI/ling-cookbook`、`inclusionAI/Avernet` 维护者；
- 「谁是卧底」事实层规则参考 Avernet `undercover-game-referee`；
- 本 recipe 的代码、数据合成、训练配置、评测与文档由
  [Claude Code](https://claude.com/claude-code) 在人工督导下完成（披露见
  [AI_USAGE.md](AI_USAGE.md)）。


## Final verification / 最终核验

完整验证包含 12 个任务、3 个解码种子、1,056 条输出，保留 v2。候选在重复平民投票上为 11/48，v2 为 10/48；独立状态只有 16 个，尚未证明稳定的决策质量提升。详见[核验报告](eval/verification/REPORT.md)与[接入测量](eval/verification/INTEGRATION.md)。
