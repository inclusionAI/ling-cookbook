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
├── undercover-player-zh.ipynb  # 中文 walkthrough notebook（快速上手主入口）
├── undercover-player.ipynb     # English walkthrough notebook
├── requirements.txt      # 运行依赖（mlx-lm ≥ 0.32）
├── data/
│   ├── word_pairs.py     # 30 组词对 + 属性池（含 6 组留出词对用于泛化测试）
│   ├── prompting.py      # 训练/推理统一提示词格式（历史窗口化）
│   ├── synthesize.py     # 程序化数据合成（生成 train/valid/test.jsonl）
│   ├── rerender.py       # 格式迁移：旧渲染增量样本 → 当前窗口格式（标签不变）
│   ├── live-v1.jsonl     # 实战对局增量（flywheel.py 产出，带真值投票标签）
│   ├── api-distill-v1.jsonl  # 教师蒸馏增量（distill.py 产出，离线标注）
│   ├── train-v3.jsonl    # 当前合并产物（1212 合成 + 156 增量，窗口化渲染）
│   ├── train-v2.jsonl    # 上一轮合并产物（1208 + 156，全量历史渲染，存档）
│   └── *.jsonl           # 已生成数据集：1212 / 76 / 96 条
├── flywheel.py           # 数据飞轮：对局状态 → 增量样本 → 按比例合并（见「数据飞轮」）
├── distill.py            # 蒸馏轨道：教师大模型离线出动作标签（可选进阶，见「数据飞轮」）
├── train/
│   ├── lora_config.yaml  # QLoRA 超参（rank 8，后 16/24 层）
│   └── train.sh          # 训练入口
├── eval/
│   ├── evaluate.py       # 微调前后自动评测 → JSON 对比（--seed 换种子出误差棒）
│   ├── selfplay.py       # 自对弈胜率：6 席全模型 + 脚本裁判（含随机基线）
│   ├── base-on-v3test.json    # 窗口化 test 三方对比：底座基线
│   ├── lora-v2-on-v3test.json # 同 test 复测 v2（当前收养件）
│   ├── lora-v3.json           # 窗口化重训 v3 实测（门槛未过）
│   ├── lora-v2.json           # v2 上一轮实测（历史 test 存档）
│   └── metrics-history.jsonl  # 每轮迭代的指标账本
├── infer/
│   └── chat.py           # 单条交互 demo（发言/投票）
└── adapters/             # 训练输出（git 不跟踪权重，见 Releases）
```

权重不入 Git：[GitHub Release `undercover-player-adapter-v2`](https://github.com/lplalbert/ling-cookbook/releases/tag/undercover-player-adapter-v2)
托管当前收养件（`adapters.safetensors` ~397M + `adapter_config.json`，不含训练中间 checkpoints）。

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
对局深度 1~6 轮随机（对齐实战「最多 6 轮」）。

**提示词格式（窗口化）**：历史渲染只保留**第 1 轮（初始描述）+ 最近 2 轮**，
中间轮折叠成一行「（第x-y轮略）」（`prompting.render_history`）。这样 prompt
随对局深度有界（实测 ≤420 字符），且 ≥4 轮样本携带折叠标记进入训练集
（训练集中 626/1212 条触发折叠），保证实战长对局的输入不越出训练分布。
Avernet 玩家 Bot 内嵌同款渲染（逐字节一致），训练/推理分布严格对齐。

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
把训练集投票样本过采样 3 倍（train 1212 条 = 432 speak + 780 vote）；
valid/test 保持自然分布，评测口径可比。

### 3. 评测（微调前后对比）

```bash
python eval/evaluate.py --model ~/models/Ling-3.0-tiny-4bit \
    --data data/test.jsonl --out eval/base.json
python eval/evaluate.py --model ~/models/Ling-3.0-tiny-4bit --adapter-path adapters \
    --data data/test.jsonl --out eval/lora.json
# 自对弈胜率（6 席全模型 + 脚本裁判）+ 随机基线参照
python eval/selfplay.py --model ~/models/Ling-3.0-tiny-4bit --adapter-path adapters \
    --games 20 --out eval/selfplay.json
python eval/selfplay.py --model ~/models/Ling-3.0-tiny-4bit --policy random \
    --games 200 --out eval/selfplay-random.json
```

指标：发言字数合规率、泄词率（输出含自己词语任一字）、自然度（排除
`1. **分析请求：**` 之类的思考前缀）、格式合规率、投票格式合法率、投票正确率、
**自对弈胜率**（平民/卧底胜率，对照纯随机投票基线）。`evaluate.py --seed` 换
随机种子重跑即可出误差棒（同一 test 多种子对比，见 metrics-history）。

当前输出预算默认 256 tokens（含思考与正文）；评测会记录思考是否完成，未闭合的输出判为无有效答案。

解码协议：固定种子采样（`temp=0.7, top_p=0.9, seed=42`），并用
`thinking_option="on"`（Ling 默认模式，与训练一致）渲染提示词，
再用 `clean_output()` 提取思考结束后的正文（原因见
Troubleshooting #6/#7）。贪心解码会让模型坍缩到单一安全输出（发言全用万能句、
投票全弃权），不能反映决策质量。

### 4. 交互 demo

```bash
python infer/chat.py --model ~/models/Ling-3.0-tiny-4bit --adapter-path adapters
```

## 评测结果

指标说明：`vote_accuracy` 是历史字段名，表示与规则标签完全匹配，并非游戏策略正确率。
新结果另记 `vote_label_match_rate` 与平民识别真实卧底的 `civilian_undercover_hit_rate`；
卧底策略标签包含随机弃权，不能把不匹配直接视为错误决策。

### 2026-11-14 完整验证结论

四种条件 × 三个采样种子的完整验证已结束，**保留现用 v2，不收养新候选**。
座位重排候选的平民命中为 11/48 次采样，现用 v2 为 10/48；这里仅有 16 个独立状态，
按状态重采样的命中差区间跨零（约 -8.3 至 +12.5 个百分点）。
候选的合法重编号一致性仅 1/48，与原编号重复训练组相同，未证明修复座位依赖。
发言格式没有观察到平均退化，但本轮不支持“投票能力得到稳定提升”的结论。
新协议保持显式可选，不继续无条件扩训；完整结果见`eval/verification/REPORT.md`。

### state-v2 输入协议实验

新增实验显式提供本人座位与存活玩家；旧 adapter 的默认调用保留 legacy 输入。
两种协议都使用 Ling 默认思考模式，协议变更不是关闭思考。

```bash
python data/build_state_dataset.py --source data \
  --out data-v5-state --control-out data-v5-control
python infer/chat.py --model ~/models/Ling-3.0-tiny-4bit \
  --adapter-path adapters-v2-run3 --prompt-protocol state-v2 --interactive
python eval/selfplay.py --model ~/models/Ling-3.0-tiny-4bit \
  --adapter-path adapters-v2-run3 --prompt-protocol state-v2 --games 20
```

交互 JSON 在原有字段之外必须提供 `own_seat`（整数）和 `alive_seats`（整数列表），
例如 `"own_seat": 2, "alive_seats": [1, 2, 4, 6]`。自对弈传入实时存活名单。
离线评测直接使用数据里的完整 messages，不再另外猜测或添加状态。

独立数据为 train 1212 / valid 76 / test 96 条；对照与实验样本、顺序、标签均相同，
仅实验组增加状态字段。原合成器没有淘汰过程，因此这些样本全员存活，不代表已覆盖淘汰后的训练分布。
转换器拒绝处理实战增量，不能把初始席位当成真实对局的存活名单。
生成时校验留出词对隔离、训练/验证输入不重叠、投票目标合法；manifest 保存数据哈希。
当前检查的完整模板最长 349 tokens，低于 1024-token 训练上限。

`eval/state-v2-pilot/` 保存现有 v2 的 16 条配对输入诊断（8 发言、8 投票）。
使用相同逐样本种子、思考模式与输出预算；它只能衡量旧 adapter 对新输入的响应，
不能证明新版数据重训后的收益，也不用于自动替换现有 adapter。

> 以下为旧版推理／解析协议的历史结果。当前默认思考模式与严格解析协议需重新评测，旧表不代表新协议验收通过。

固定种子采样（temp=0.7, top_p=0.9, seed=42）。

### 窗口化 test 三方对比（当前 test.jsonl，96 条中 48 条触发历史折叠）

| 指标 | 基线（4-bit） | v2 adapter（收养件） | v3 adapter（窗口化重训） |
|---|---|---|---|
| 发言字数合规率（≤25 字） | 93.3% | **100%** | 88.3% |
| 发言自然度（非思考前缀） | 8.3% | 100% | 100% |
| 发言格式合规率（字数+不泄词+自然） | 3.3% | **93.3%** | 78.3% |
| 发言泄词率（原始输出，越低越好） | 5.0% | **6.7%** | 11.7% ⚠ |
| 投票格式合法率（`我投N号`/`我弃权`） | 52.8% | **100%** | **100%** |
| 投票规则标签匹配率（对齐规则式标签，观测项） | 11.1% | 25.0% | 19.4% |

**门槛判定：v3 未达标，v2 继续收养**（vote_format=1.0 ✓、speak_format
0.783 < 0.90 ✗、泄词 0.117 较上轮 +5pt ✗）。折叠子集归因（各 48 条）：
**该小样本中 v2 在折叠／未折叠子集上的表现接近**（折叠/未折叠 speak_format 0.939/0.926、泄词
0.061/0.074）——v2 在历史协议下通过了当时的门槛；v3 的回退**集中在未折叠短局**
（speak_format 0.630、泄词 0.222；折叠子集反而 0.909/0.030），成因尚未验证；优化器重置与数据分布变化仅为候选解释。
全部数字见 `eval/base-on-v3test.json` / `eval/lora-v2-on-v3test.json` /
`eval/lora-v3.json` 与 `eval/metrics-history.jsonl`。

### v1 初版结果（历史存档：旧版 test，全量历史渲染、对局深度 ≤3 轮）

| 指标 | 基线（4-bit） | +QLoRA |
|---|---|---|
| 发言字数合规率（≤25 字） | 96.7% | **100%** |
| 发言自然度（非思考前缀） | 3.3% | **100%** |
| 发言格式合规率（字数+不泄词+自然） | 0% | **85.0%** |
| 发言泄词率（原始输出，越低越好） | 1.7% | 15.0% ⚠ |
| 投票格式合法率（`我投N号`/`我弃权`） | 44.4% | **100%** |
| 投票规则标签匹配率（对齐规则式标签） | 5.6% | 11.1% |

（v1/v2 当轮实测在旧版 test 上，与上方窗口化 test 数字口径不同、不可直接比；
同 test 三方对比以上表为准。）

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
     （如「甜的」两词都沾），可能存在输入信息不足、标签歧义及训练不足等原因；尚不能证明不可约噪声或模型能力上限。
- **格式合规不受影响**：v2 训练后投票输出 100% 可解析（过采样治好了「全弃权」
  坍缩），Avernet 验收口径（格式/协议）达标，胜率不计分。
- 线上兜底：清洗层强制 25 字/泄词删除/投票格式（见 Avernet 侧 `rules.py`），
  这些规则的保证范围需以应用侧测试与日志核实。
- **v3 窗口化重训门槛未过，v2 继续收养**（门槛机制第一次真实拦截回退）：
  折叠子集上 v3 反而更好、回退集中在短局样本；不做「训了就换」，收养只认
  门槛数字（见上表与 `metrics-history.jsonl`）。

## 应用：Avernet「谁是卧底」牌桌（一鱼两吃）

本地服务应使用玩家项目的 `serve.py` 包装，并在 Bot 启动时显式传入同一个 v2 adapter 路径。
模型别名不代表选择微调权重；未传 adapter 时只允许显式选择底座模式。
实现见 Avernet PR #3970（认领 issue #2611），本轮已同步两个本地玩家工作树，但未推送。

本轮完成了独立本地服务、协议回环与缓存隔离验证，未启动真实 BCS 对局或发送牌桌消息。
默认思考开启，Bot 采样配置与 recipe 一致；输出层剥离思考、限制发言并拒绝已知非法票。
PK 的合法动作依赖上游提供有效候选；不能无条件承诺“模型怎么输出都不会违规”。
当前入口以平民提示兼容旧模型，真实牌桌并不提供隐藏身份，因此离线身份已知的评测不能直接代表实战质量。
历史窗口只限制保留轮数，不能保证任意上游文本都不超过固定字符数。

三轮缓存测试的 `cached_tokens` 均为零，重复与冷请求约 4.9/5.2 秒，没有证实缓存加速。
当前 Ling 混合缓存不能任意回退；`--prompt-cache-size 2` 是容量设置，不是性能收益保证。
底座与 LoRA 缓存命名空间已隔离。详见`eval/verification/INTEGRATION.md`。

## 数据飞轮：从实战对局迭代（可选进阶）

合成数据有个天花板（见上：投票标签是规则启发式，噪声与输入信息不足待核查）。而**实战对局
里卧底身份是上帝视角真值**——`flywheel.py` 把牌桌对局状态转成无标签噪声的
新训练样本，形成「训练 → 实战 → 采样 → 再训练」的自我迭代：

```bash
# 每攒够 ≥5 局新对局跑一轮：
python flywheel.py build --out data/live-vN.jsonl     # 对局状态 -> 实战样本
python flywheel.py merge --live data/live-vN.jsonl data/api-distill-vN.jsonl \
    --out data/train-vN.jsonl                        # 增量可多个，实战+蒸馏一次合并
mkdir -p data-v2 && cp data/train-vN.jsonl data-v2/train.jsonl \
    && cp data/valid.jsonl data-v2/valid.jsonl        # mlx_lm 认 train/valid 布局
# 然后照常重训（QLoRA，同超参，--data data-v2 --adapter-path adapters-v2，
# 16G 上分段续训见「内存与拓扑」）→ 评测门槛 → 达标才换线上 adapter
```

提示词格式若改动（如历史窗口化），已采集的增量样本用 `python data/rerender.py
data/live-vN.jsonl data/api-distill-vN.jsonl` 就地重渲染——旧 prompt 解析回
rounds 后按当前格式重画 user 部分，动作标签不动（解析先过无损往返校验才落盘）。

三条采样规则（详见 `flywheel.py` 模块注释）：

1. **vote 样本**（核心增量）：每个存活平民座位 × 每个已完成投票轮，标签 =
   `我投<真实卧底>号`（上帝视角专家，DAgger 式状态取实战分布、动作取真值）；
2. **speak 样本**：只模仿自己座位的干净发言（无违规/≤25 字/非思考前缀），
   己方获胜局全收、否则 0.3 概率下采样——自我模仿是弱信号，必须限流；
3. **留出纪律**：词对命中 `test.jsonl` 留出集的整局跳过；merge 默认实战样本
   ≤ 原集 30%（防自我模仿淹没原分布）。

实现细节里最容易踩的一个坑：**去重键必须含 user 历史**（task+词+完整历史+答案）。
只按 (task, 词, 答案) 去重会把「不同历史、同标签」的 DAgger 样本折叠掉——那正是
真值标签的价值所在，9 局实战会只剩 15 条。历史是多样状态，不是重复。

**蒸馏轨道（可选进阶，`distill.py`）**：把同款学生提示词发给大模型教师
（Anthropic 兼容 API，如 `mimo-v2.6-pro`）拿发言/投票动作作标签，攻击
「规则投票标签噪声与输入信息不足待核查」的瓶颈（合成对局的状态 + 专家判断）。四条防标签
冲突纪律：自己座位让给 flywheel、平民真值票不让教师抢、状态按学生提示词哈希
去重、留出词对跳过。**只做离线数据，绝不进游戏循环**（对局中的大脑始终是
本机微调小模型），凭据只存 `~/.distill-api.env`，详见 `distill.py` 模块注释
与 [AI_USAGE.md](AI_USAGE.md) 披露。合并与评测门槛同上。

**评测门槛（每轮迭代的收养标准，防自我训练坍缩）**：
`vote_format_ok = 1.0`、`speak_format_ok` 不低于上一轮、泄词率不升超 2pt——
达标才替换线上 adapter；投票正确率只观测不设门槛（天花板见归因实验）。
每轮的指标记进 `eval/metrics-history.jsonl`，「慢慢提高」必须可测量。

实测第一轮（2026-11-13，39 条实战 + 117 条蒸馏并入 1208 条合成，重训 1500 步）：
三门槛全达标，泄词率 0.15→**0.05**、speak_format_ok 0.85→**0.90**、
vote_accuracy 0.11→0.17（观测项）；代价是字数合规 1.00→0.95（约 3/60 句超长）
与 val loss 上行（自模仿样本少、有过拟合迹象）——收养以门槛为准，全部数字
见 `eval/lora-v2.json` 与 `eval/metrics-history.jsonl`。

第二轮（2026-11-13，格式窗口化迁移：重合成深度 1-6 + 156 条增量经 `rerender.py`
无损重渲染，train-v3=1368 条，重训 1500 步）**门槛未过、v2 继续收养**——
speak_format_ok 0.783、泄词率 0.117 双双回退（折叠子集反而更好，回退集中在
短局样本），三方对比与归因见「评测结果」。门槛第一次真实拦截回退，
这正是它存在的意义。

诚实预期：主要增益在①真实对手文本带来的历史鲁棒性 ②真值投票标签的判别力
信号；发言风格是自我模仿，增益有限且有坍缩风险，所以限流+门槛双保险。

## M4 16GB 统一内存预算（实测）

在 16GB M4 Air 上跑「模型服务 + 6 玩家牌桌 + 裁判」的实测账本，供同配置同学排预算：

### 磁盘

| 项 | 大小 | 说明 |
|---|---|---|
| `Ling-3.0-tiny-4bit` | 4.8G | `mlx_lm.convert` 量化产物，推理用这份 |
| `Ling-3.0-tiny-3bit` | 4.0G | 早期量化实验，可删 |
| `adapters/` | 3.1G | 其中**推理只用 `adapters.safetensors`（397M）+ `adapter_config.json`**；`checkpoints/` 是训练中间产物，托管/分发时不用带 |

### 内存与拓扑（关键结论：16G 只能单实例）

| 拓扑 | 进程 | RSS | 结论 |
|---|---|---|---|
| **单实例双角色**（推荐） | `serve.py` 4bit+QLoRA（GPU） | **5.3G** | 底座/LoRA 靠请求级 `adapters` 字段切换（LoRALinear.scale 置零=底座），模型只加载一次 |
| | 玩家 bot `main.py` | 14M | |
| | 裁判 `referee_driver.py` | 5M | |
| 双实例（已废弃） | 3bit base + 4bit LoRA 两个 serve | 两份权重 ≈9G+ | 挤爆压缩器（swap 5.1G 满载、compressor 4.7G）→ **推理变慢 10 倍** → 节点 420s 超时 |

16G 机器的护栏：`--prompt-cache-size 2`；**别开** `--kv-bits 8 --quantized-kv-start 2048`
（mlx-lm 对 Ling 的 MoE 实现不兼容，见 Troubleshooting #8）；`MLX_CPU=1` 只为双模型
共存设计（CPU 权重可被系统换页，Metal 缓冲不行），单实例直接用 GPU。

**训练侧的 16G 约束（2026-11-13 实测）**：QLoRA 1000+ iters 连续跑会在 ~625 步后
确定性 `Metal Insufficient Memory`（同种子两次死在同一步，`mx.eval` 优化器状态
处的分配峰值 + 碎片化；v1 当夜一次性跑完属于运气边界）。对策 = **分段续训**：
每段 ≤600 步，段间用 `--resume-adapter-file <上段 0000XXX_adapters.safetensors>`
续权重（优化器态会重置；不等价于无中断续训，需单独评估影响）。检查点每 200 步落盘，断在哪
都能接。若续训中 val loss 明显回升（实测累计 1200 步时 1.61→2.14）是过拟合
信号，收养前用固定 test 集门槛说话，别只看步数。

### 时延（单节点，提示词 2~4k token）

| 模式 | 单次发言/投票 | 备注 |
|---|---|---|
| 单实例 GPU | 秒级 | 短提示实测 <1s；牌桌节点含长上下文，见 Avernet PR 的对局日志 |
| 双实例 CPU（`MLX_CPU=1`） | 30~90s | 内存挤压期 180s+，直接打穿 420s 节点预算 |

混档请求（底座/微调）不能进同一个 batch（scale 是模型级状态），serve.py 强制
`is_batchable=False` 串行——牌桌节点本就链式派发，吞吐不受影响。

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
6. **原始输出以 `</think>` 开头**
   —— 默认训练目标包含空思考块的闭合标签，默认推理提示词停在 `<think>`，
   因此模型先生成 `</think>` 再回答可以是正常行为，不证明 `mask_prompt` 算错。
   交互、评测、自对弈统一使用 `thinking_option="on"`，与默认训练模板一致；
   `clean_output()` 提取闭合标签后的正文并移除回合结束标记。
   历史上关闭思考的评测改变了提示条件，旧指标不能直接当作默认模式结果。
7. **贪心解码模式坍缩**：temp=0 时投票 100% 输出「我弃权」（弃权是单一字符串，
   5 个候选把概率质量摊薄）、发言全用「同意前面几位的说法」万能句。评测与推理
   统一改用固定种子采样（temp=0.7, top_p=0.9）；线上再叠加清洗层兜底格式。
8. **`--kv-bits 8 --quantized-kv-start 2048` 在 >2048 token 提示词上崩溃**：
   `AttributeError: 'list' object has no attribute 'swapaxes'`，位置
   `mlx_lm/models/bailing_moe_v3.py` 的 `pe_scores = (q_rope * self.scale) @
   k_rope.swapaxes(-1, -2)`——量化 KV cache 切换后 `k_rope` 是 list，模型代码
   没处理。短提示（<2048 token）永远踩不到，冒烟测试测不出；**2k+ 真实提示词
   一打就炸**（当时表现为玩家 Bot "empty response" 全灭）。规避：不开量化 KV
   （16G 单实例显存够用）；根治需上游修（见 Avernet PR 的上游 issue 链接）。
9. **训练分布的对局深度覆盖不到实战上限，窗口折叠标记就是 OOD**：历史窗口化后
   ≥4 轮才会出现「（第x-y轮略）」折叠标记，而合成器原本只生成 1~3 轮对局——
   训练集里折叠标记零覆盖，实战 6 轮对局一上来全是没见过的输入形态。修法：
   合成对局深度拉齐实战上限（`n_rounds = rng.randint(1, 6)`），让 626/1212 条
   训练样本携带折叠标记。教训通用：**凡是推理时会变换/折叠/截断输入的规则，
   训练数据必须覆盖该变换的全部取值**，否则「训练/推理同格式」只是部分成立。

## AI 使用策略

本 recipe 的代码、数据合成、训练配置、评测与文档由 Claude Code 在人工督导下完成，
详见 [AI_USAGE.md](AI_USAGE.md)。
