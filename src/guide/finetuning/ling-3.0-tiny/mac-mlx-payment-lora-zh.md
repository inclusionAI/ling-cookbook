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

# 基于 AReno 在 Mac 上微调 Ling-3.0-tiny 模拟组合支付决策

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

本教程演示如何在 Apple Silicon Mac 本地，使用 [AReno](https://github.com/inclusionAI/AReno) 后训练框架，对百灵的轻量级 MoE 模型 [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) 进行 4-bit QLoRA 监督微调，让模型模拟一个「组合支付计算与推荐服务」这一传统上由确定性代码实现的业务逻辑。

在这个任务中，微调后的模型需要接收包含「订单金额」、「首选支付渠道」、「各渠道可用余额」以及「优惠券规则」的 JSON 请求，逐步推演满减门槛、优惠叠加与余额拆单逻辑，最终输出分配好的各支付渠道结算单与「推荐用户使用的更优支付渠道」。

通过本教程，你将依次完成环境安装、数据集与任务规则解析、端侧 LoRA 训练、训练看板查看、OpenAI 兼容推理服务拉起，以及测试数据集的自动化评测。这是一个典型的模型后训练和微调任务，无论是想了解和学习模型训练，还是有具体任务希望评估，都可以作为参考。

+++

## 2. 注意事项

我们选择「组合支付与优惠计算」作为微调示例，是因为其结果结果可由程序客观判定，这种任务特性适合用来观察和理解微调如何塑造模型的行为。你可以将本任务视为理解端侧微调机制、数据构造与评估方法的一次实验、或者学习资料，但未经评估，不要在生产任务中使用 LLM 直接替代此类高风险计算逻辑。

+++

## 3. 模型与微调框架介绍

- 底座模型：`inclusionAI/Ling-3.0-tiny` 是百灵系列的轻量级 MoE 模型，总参数量 7.9B，单 Token 激活参数量 1.3B，适合本地部署和自定义。
- 后训练框架与后端：[AReno](https://asystem-ai.io/docs/areno/getting-started/installation.html) 是百灵团队开发的，原生支持 Apple Silicon 的后训练框架。

+++

## 4. 设备要求

运行本教程的端侧训练与评测，需要满足以下软硬件条件：

- 硬件设备：搭载 Apple Silicon（M 系列芯片）的 Mac 设备。
- 统一内存：至少 16GB 统一内存（实测在 16GB 内存的 MacBook Air M5 上，500 步训练期间 MLX 分配器峰值约为 10GB）。
- 磁盘空间：至少预留约 25GB 可用磁盘空间，用于存放 `inclusionAI/Ling-3.0-tiny` 的 BF16 权重（约 15GB）、数据集缓存及训练检查点。
- 软件环境：macOS 系统，安装 Git 与 Python 虚拟环境（推荐使用 Python 3.11 或 3.12）。

+++

## 5. 任务分析

本任务使用公开数据集 [AsystemAReno/Ling-Ting-SFT-Augment](https://huggingface.co/datasets/AsystemAReno/Ling-Ting-SFT-Augment)，其中包含 1,115 条训练样本（`train.jsonl`）、100 条验证样本（`valid.jsonl`）和 500 道测试题（`eval500.jsonl`）。

在每条样本中，模型需要同时完成两项决策：
1. 按用户当前选定的首选渠道执行结算（若首选渠道余额不足，则用备选渠道补足差额）
2. 分析所有可用渠道与优惠规则，判断是否存在比当前选择更省钱的单一渠道支付方案，并给出推荐。

下面以一笔 0.99 元的订单为例说明输入与输出逻辑。

输入场景设定如下：

- 首选数字人民币钱包余额为 0.21 元，备选浦发银行卡余额为 2.97 元。
- 存在两条优惠规则：a) 全场通用优惠，满 0.24 元减 0.02 元； b) 浦发银行卡专享优惠，满 0.33 元减 0.19 元，且可与通用优惠叠加。

对应的输入 JSON 节选：

```jsonc
{
  "order": {"order_amount": 99}, // 金额单位为分：99分=0.99元
  "selected_channel_id": "DIGITAL_RMB_WALLET_8752",
  "available_channels": [
    {"channel_id": "DIGITAL_RMB_WALLET_8752", "balance": 21},
    {"channel_id": "BANK_SPDB_CARD_2565", "balance": 297}
  ],
  "promotions": [
    // 通用减2分；使用浦发卡时还能叠加19分优惠
    {"applicable_channel_id": null, "threshold": 24, "discount_amount": 2},
    {"applicable_channel_id": "BANK_SPDB_CARD_2565", "threshold": 33, "discount_amount": 19}
  ]
}
```

模型需要推导出的第一部分是按首选钱包拆单结算（对应标准答案中的 `cashier_receipt`）：

- 订单总额 99 分满足通用优惠门槛（24 分），减免 2 分后应付 97 分。
- 首选钱包余额仅 21 分，全部扣完后仍差 76 分，由备选浦发卡补足 76 分。

```jsonc
{
  "status": "SUCCESS",
  "cashier_receipt": {
    "primary_deduct_amount": 21,   // 首选钱包
    "secondary_deduct_amount": 76, // 浦发卡补足
    "total_discount_amount": 2,
    "payable_amount": 97
  }
}
```

模型需要推导出的第二部分是更优支付方式推荐（对应标准答案中的 `recommended_option`）：

- 若改用浦发卡单独支付整笔订单，可同时命中通用优惠（2 分）与浦发卡专享优惠（19 分），共优惠 21 分。
- 实付金额降至 78 分（0.78 元），比拆单方案少付 19 分。

```jsonc
{
  "recommended_option": {
    "recommended_channel_id": "BANK_SPDB_CARD_2565",
    "payable_amount": 78,         // 浦发卡单卡支付
    "total_discount_amount": 21
  }
}
```

+++

<a id="setup"></a>

## 6. 操作步骤

### 步骤 1：安装 AReno

在 Apple Silicon Mac 上克隆 AReno 仓库并从源码安装：

```{code-cell} ipython3
%%bash
git clone https://github.com/inclusionAI/AReno.git AReno
pip install -e AReno
```

更多环境细节可参考 [AReno 安装文档](https://asystem-ai.io/docs/areno/getting-started/installation.html) 。

+++

<a id="train"></a>

### 步骤 2：启动 QLoRA 训练与指标看板

运行以下命令启动 SFT 训练。

首次运行时会自动从 Hugging Face 下载 `inclusionAI/Ling-3.0-tiny` 权重与 `AsystemAReno/Ling-Ting-SFT-Augment` 数据集，下载耗时取决于网络状况，请耐心等待。

训练配置在加载时先基座量化为 4-bit，对全部 24 层的 Attention 和 MLP 模块挂载 LoRA，微批次大小设为 1，共训练 500 步，每 100 步保存一次适配器权重。

```{code-cell} ipython3
%%bash
train_args=(
  --algo sft --model-hub hf
  --ckpt inclusionAI/Ling-3.0-tiny
  --dataset-path AsystemAReno/Ling-Ting-SFT-Augment::train
  --dataset-loader-fn AReno/examples/sft/ling_payment/dataset_loader.py
  --validation-dataset-path AsystemAReno/Ling-Ting-SFT-Augment::validation
  # 在计划的更新前做验证，按受监督 token 数加权计算 loss。
  --eval-batches 25 --eval-before-step
  --shuffle --seed 0
  --q4  # 加载时将冻结的基座量化为 4bit。
  --lora-rank 4 --lora-alpha 32 --lora-dropout 0.05
  # 在全部 24 层的 Attention 和 MLP 中训练适配器，基座权重保持冻结。
  --lora-target-modules q_proj,k_proj,v_proj,f_proj,o_proj,g_proj,b_proj,q_a_proj,q_b_proj,kv_a_proj_with_mqa,dense,gate_proj,up_proj,down_proj
  # 每个微批次用一条样本，默认开启激活检查点。
  --batch-size 1 --mini-bs 1 --score-micro-bs 1
  --epochs 1 --max-steps 500
  --max-new-tokens 1024
  --lr 2e-5 --min-lr 2e-5 --lr-decay-style constant --weight-decay 0
  --no-adam-bias-correction  # 跳过 Adam 对零初始化矩估计的偏差修正。
  --no-grad-clip  
  --no-mlx-compile-train-step  # 不用 mx.compile 编译损失与梯度计算函数。
  --save-path runs/q4-cli/adapters  # 默认每 100 步保存一次适配器。
  --metrics-log-dir runs/q4-cli/events
)
areno train "${train_args[@]}"
```

训练过程中、或训练完成后，你都可通过 `--save-path` 指定的目录查看导出的适配器权重。第 500 步的最终产物位于 `runs/q4-cli/adapters/step_000500/`。

你可以启动 AReno Dashboard，直观地查看训练损失曲线、每步耗时与运行日志：

```{code-cell} ipython3
%%bash
areno dashboard --start
```

启动后在浏览器访问 [http://127.0.0.1:8765](http://127.0.0.1:8765)，在 Jobs 页面中查看训练记录：

![AReno Dashboard 中的 Ling-3.0-tiny 500 步训练记录](../../../resources/assets/mac-mlx-payment-lora/areno-dashboard.png)

*图 1：500 步训练在 AReno Dashboard 中的记录。*

+++

<a id="inference"></a>

### 步骤 3：启动推理服务

使用 `areno serve` 加载底座模型与第 500 步保存的 LoRA 适配器（`runs/q4-cli/adapters/step_000500`），在本地 `8000` 端口启动兼容 OpenAI 协议的 HTTP 推理服务：

```{code-cell} ipython3
%%bash --bg
areno serve \
  --backend mlx \
  --model-hub hf \
  --model-path inclusionAI/Ling-3.0-tiny \
  --lora-adapter-path runs/q4-cli/adapters/step_000500 \
  --host 127.0.0.1 --port 8000 \
  --max-running-prompts 2 --default-max-tokens 4096
```

+++

### 步骤 4：运行 500 题自动评测

等待后台推理服务完成模型加载并监听 `8000` 端口后，运行以下评测脚本对测试集 `eval500.jsonl` 中的 500 道题进行批量推理，并将生成结果保存到本地 `eval_results.jsonl`。

评分规则：提取模型输出中 `</think>` 之后的最终 JSON 文本，与标准答案 `ground_truth` 逐项比对。所有字段数值与数组顺序必须完全一致（JSON 对象键名顺序不限），无法解析为合法 JSON 或任意数值不符均计为错误：

```{code-cell} ipython3
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local", timeout=600)
model = client.models.list().data[0].id
eval_path = Path(hf_hub_download(
    repo_id="AsystemAReno/Ling-Ting-SFT-Augment",
    filename="eval500.jsonl",
    repo_type="dataset",
))
cases = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
results = []

for i, case in enumerate(cases, 1):
    response = client.chat.completions.create(
        model=model,
        # 只发送规则和问题，不发送样本中的 assistant 答案
        messages=[m for m in case["messages"] if m["role"] != "assistant"],
        temperature=0.1,
        max_tokens=4096,
    )
    answer = response.choices[0].message.content or ""
    try:
        prediction = json.loads(answer.rsplit("</think>", 1)[-1].strip())
        correct = prediction == case["ground_truth"]
    except json.JSONDecodeError:
        correct = False
    results.append({"case_id": case["case_id"], "correct": correct, "output": answer})
    print(f"已完成 {i}/{len(cases)}", end="\r")

Path("eval_results.jsonl").write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n",
    encoding="utf-8",
)
correct = sum(row["correct"] for row in results)
print(f"\n测试集正确率： {correct}/{len(cases)} = {correct / len(cases):.2%}")
```

+++

<a id="results"></a>

## 7. 典型结果

在 MacBook Air（M5，16GB 统一内存）上，我们测试得到的实测评测数据如下：

- 训练开销：共训练 500 步，可训练 LoRA 参数量为 74,947,584，总耗时 3 小时 15 分 55 秒，MLX 内存分配器记录的峰值约为 10.03GB。
- 评测表现：在 500 道测试题上，微调后的模型完全答对 322 题，严格 JSON 完全匹配正确率为 64.4%（对比微调前基线，微调前在相同严格匹配标准下为 0% -- 完全无法进行此任务）。

![Ling-3.0-tiny 微调前后的 500 题评测对比](../../../resources/assets/mac-mlx-payment-lora/evaluation-before-after-zh.png)

*图 2：左侧对比微调前后整体严格匹配正确率，右侧展示微调后在各细分场景下的答对题数 / 该场景总题数（场景分类沿用测试集原始标签）。*

+++

## 8. 常见问题与进阶方向

### 常见问题排查

- 执行 `pip install -e AReno` 时报错 `externally-managed-environment`
  - macOS 系统自带或 Homebrew 安装的 Python 默认开启了 PEP 668 保护。请先使用 `uv venv .venv --python 3.12 && source .venv/bin/activate`（或 `python3 -m venv .venv`）创建并激活虚拟环境，再执行安装命令。

- 运行评测脚本时报错 `Connection refused`
  - 步骤 3 中的 `areno serve` 使用 `%%bash --bg` 在后台异步启动，加载底座权重与注入 LoRA 适配器需要一定时间。若启动后立即执行步骤 4，服务尚未完成端口监听。可等待约 30~60 秒，或在终端通过 `lsof -i :8000` 确认端口已处于 `LISTEN` 状态后再启动评测。

- 16GB 统一内存机型在训练过程中出现明显卡顿
  - 训练期间 MLX 峰值内存约 10.03GB，留给 macOS 系统和其他应用的空间较为有限。建议在训练前关闭浏览器过多标签页、Docker Desktop 或其他本地大模型驻留进程，减少触发系统磁盘交换（Swap）。

- 为什么训练命令中开启了 `--no-mlx-compile-train-step`
  - 禁用 `mx.compile` 对训练单步损失与梯度函数的图编译，能够降低变长序列输入下的重编译开销与额外内存峰值，使 16GB 统一内存设备上的运行更加平稳。

### 还能做什么？

如果你想在此基线（64.4%）上继续探索后训练效果，可以从以下方向深入：

- 引入可验证奖励强化学习（RLVR）
  - 组合支付计算具有明确的客观评判标准。在模型通过 SFT 掌握输出格式与基础演算步骤后，可以利用 AReno 支持的强化学习算法（参见 [DPO / GRPO / GSPO / PPO 训练文档](https://asystem-ai.io/docs/areno/cli/training.html) 与 [Agentic RL 文档](https://asystem-ai.io/docs/areno/getting-started/welcome.html#agentic-rollout)），将 JSON 解析合法性、优惠互斥合规性以及实付金额精确匹配编写为规则奖励函数（Reward Function），通过同题多采样与强化学习进一步减少算术与条件分支失误。

- 针对薄弱的子场景扩充合成数据
  - 观察图 2 右侧的分场景得分，定位错误率较高的复杂情况（例如多张满减券门槛接近、跨渠道拆单边界等），编写确定性的 Python 结算脚本合成更多同类难例，加入训练集进行增量微调。

- 调整 LoRA Adapter 容量与训练步数
  - 本教程为兼顾 16GB 机型采用了 `--lora-rank 4` 和 500 步训练。如果在 24GB/36GB 以上内存的 Mac 或具备大显存的 Linux GPU 环境上运行，可以尝试将 `--lora-rank` 提高至 8 或 16，增加训练步数，或结合 [8-bit AdamW](https://asystem-ai.io/docs/areno/concepts/adamw-8bit.html) 优化器，进一步观察验证集损失与测试集准确率的收敛上限。

+++

## 9. 致谢

- 感谢 [@sofan110](https://github.com/sofan110) 提供此微调用例。

