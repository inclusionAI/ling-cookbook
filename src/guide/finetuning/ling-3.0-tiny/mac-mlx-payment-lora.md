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

# Fine-Tuning Ling-3.0-tiny on Mac with AReno for Split-Payment Decision Making

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

## 1. What This Task Does

This guide demonstrates how to run 4-bit QLoRA supervised fine-tuning (SFT) on the lightweight Mixture-of-Experts (MoE) model [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) locally on an Apple Silicon Mac using the [AReno](https://github.com/inclusionAI/AReno) post-training framework, training the model to simulate a split-payment calculation and recommendation service—business logic traditionally implemented with deterministic code.

In this task, the fine-tuned model receives a JSON request containing the order amount, preferred payment channel, available balances across channels, and promotion rules. It reasons step by step through discount thresholds, promotion stacking, and split-payment deductions, then outputs a settlement receipt across payment channels along with a recommended single-channel payment option that saves more money.

Through this guide, you will walk through environment installation, dataset and rule inspection, on-device LoRA training, training dashboard monitoring, launching an OpenAI-compatible inference server, and automated evaluation on the test set. It serves as a practical reference both for learning model post-training and fine-tuning workflows and for evaluating model adaptation on specific tasks.

+++

## 2. Important Notes

We chose "split-payment and discount calculation" as a fine-tuning example because its outputs can be verified objectively by a program, making it suitable for observing and understanding how fine-tuning shapes model behavior. You can treat this task as an experiment or learning resource on on-device fine-tuning mechanisms, dataset construction, and evaluation methods, but do not use an LLM to directly replace such high-risk calculation logic in production without thorough evaluation.

+++

## 3. Model and Fine-Tuning Framework

- Base Model: `inclusionAI/Ling-3.0-tiny` is a lightweight Mixture-of-Experts (MoE) model in the Ling family with 7.9B total parameters and 1.3B active parameters per token, well suited for local deployment and customization.
- Post-Training Framework and Backend: [AReno](https://asystem-ai.io/docs/areno/getting-started/installation.html) is a post-training framework developed by the Ling team with native support for Apple Silicon.

+++

## 4. Hardware and Software Requirements

Running the on-device training and evaluation in this guide requires the following hardware and software setup:

- Hardware: Mac equipped with Apple Silicon (M-series chip).
- Unified Memory: At least 16GB of unified memory (measured MLX allocator peak during the 500-step training run on a 16GB MacBook Air M5 is approximately 10GB).
- Disk Space: At least 25GB of free disk space for storing the `inclusionAI/Ling-3.0-tiny` BF16 weights (~15GB), dataset cache, and training checkpoints.
- Software Environment: macOS with Git and a Python virtual environment installed (Python 3.11 or 3.12 recommended).

+++

## 5. Task Analysis

This task uses the public dataset [AsystemAReno/Ling-Ting-SFT-Augment](https://huggingface.co/datasets/AsystemAReno/Ling-Ting-SFT-Augment), which contains 1,115 training examples (`train.jsonl`), 100 validation examples (`valid.jsonl`), and 500 test cases (`eval500.jsonl`).

For each example, the model must make two decisions simultaneously:
1. Settle the payment using the user's currently selected preferred channel (covering any shortfall with a backup channel if the preferred channel balance is insufficient).
2. Analyze all available channels and promotion rules to determine whether a cheaper single-channel payment option exists, and output the recommendation.

Below we illustrate the input and output logic using a CNY 0.99 order as an example.

The input scenario is defined as follows:

- The preferred digital RMB wallet balance is CNY 0.21, and the backup SPDB bank card balance is CNY 2.97.
- Two promotion rules apply: a) a general promotion offering CNY 0.02 off orders of at least CNY 0.24; b) an SPDB card-exclusive promotion offering CNY 0.19 off orders of at least CNY 0.33, which can stack with the general promotion.

Corresponding input JSON excerpt:

```jsonc
{
  "order": {"order_amount": 99}, // Amounts are in fen: 99 fen = CNY 0.99
  "selected_channel_id": "DIGITAL_RMB_WALLET_8752",
  "available_channels": [
    {"channel_id": "DIGITAL_RMB_WALLET_8752", "balance": 21},
    {"channel_id": "BANK_SPDB_CARD_2565", "balance": 297}
  ],
  "promotions": [
    // General 2-fen discount; paying with the SPDB card stacks an extra 19-fen discount
    {"applicable_channel_id": null, "threshold": 24, "discount_amount": 2},
    {"applicable_channel_id": "BANK_SPDB_CARD_2565", "threshold": 33, "discount_amount": 19}
  ]
}
```

The first part the model needs to derive is the split-payment settlement using the preferred wallet (`cashier_receipt` in the reference answer):

- The order total of 99 fen meets the general promotion threshold (24 fen), leaving 97 fen payable after the 2-fen discount.
- The preferred wallet has only 21 fen; after deducting all 21 fen, the remaining 76 fen is covered by the backup SPDB card.

```jsonc
{
  "status": "SUCCESS",
  "cashier_receipt": {
    "primary_deduct_amount": 21,   // Preferred wallet
    "secondary_deduct_amount": 76, // Covered by SPDB card
    "total_discount_amount": 2,
    "payable_amount": 97
  }
}
```

The second part the model needs to derive is the cheaper payment recommendation (`recommended_option` in the reference answer):

- Paying the entire order with the SPDB card alone qualifies for both the general promotion (2 fen) and the SPDB card-exclusive promotion (19 fen), for a total discount of 21 fen.
- The payable amount drops to 78 fen (CNY 0.78), saving 19 fen compared with the split-payment plan.

```jsonc
{
  "recommended_option": {
    "recommended_channel_id": "BANK_SPDB_CARD_2565",
    "payable_amount": 78,         // Paid with SPDB card alone
    "total_discount_amount": 21
  }
}
```

+++

<a id="setup"></a>

## 6. Step-by-Step Guide

### Step 1: Install AReno

Clone the AReno repository on your Apple Silicon Mac and install it from source:

```{code-cell} ipython3
%%bash
git clone https://github.com/inclusionAI/AReno.git AReno
pip install -e AReno
```

For additional environment details, refer to the [AReno Installation Guide](https://asystem-ai.io/docs/areno/getting-started/installation.html).

+++

<a id="train"></a>

### Step 2: Run QLoRA Training and Launch the Metrics Dashboard

Run the command below to start SFT training.

On the first run, the `inclusionAI/Ling-3.0-tiny` weights and the `AsystemAReno/Ling-Ting-SFT-Augment` dataset will be downloaded automatically from Hugging Face. Download time depends on your network connection; please be patient.

The training configuration quantizes the frozen base model to 4-bit at load time, attaches LoRA adapters to the Attention and MLP modules across all 24 layers, sets the micro-batch size to 1, trains for 500 steps in total, and saves adapter checkpoints every 100 steps.

```{code-cell} ipython3
%%bash
train_args=(
  --algo sft --model-hub hf
  --ckpt inclusionAI/Ling-3.0-tiny
  --dataset-path AsystemAReno/Ling-Ting-SFT-Augment::train
  --dataset-loader-fn AReno/examples/sft/ling_payment/dataset_loader.py
  --validation-dataset-path AsystemAReno/Ling-Ting-SFT-Augment::validation
  # Token-weighted validation before scheduled updates.
  --eval-batches 25 --eval-before-step
  --shuffle --seed 0
  --q4  # Quantize the frozen base model to 4-bit at load time.
  --lora-rank 4 --lora-alpha 32 --lora-dropout 0.05
  # Attention/MLP adapters across all 24 layers; base weights stay frozen.
  --lora-target-modules q_proj,k_proj,v_proj,f_proj,o_proj,g_proj,b_proj,q_a_proj,q_b_proj,kv_a_proj_with_mqa,dense,gate_proj,up_proj,down_proj
  # Single-example microbatches; activation checkpointing is on by default.
  --batch-size 1 --mini-bs 1 --score-micro-bs 1
  --epochs 1 --max-steps 500
  --max-new-tokens 1024
  --lr 2e-5 --min-lr 2e-5 --lr-decay-style constant --weight-decay 0
  --no-adam-bias-correction  # Skip correction for zero-initialized Adam moment estimates.
  --no-grad-clip  
  --no-mlx-compile-train-step  # Skip mx.compile for the loss/gradient function.
  --save-path runs/q4-cli/adapters  # Save adapters every 100 steps (default).
  --metrics-log-dir runs/q4-cli/events
)
areno train "${train_args[@]}"
```

During or after training, you can inspect the exported adapter weights in the directory specified by `--save-path`. The final checkpoint at step 500 is saved under `runs/q4-cli/adapters/step_000500/`.

You can launch the AReno Dashboard to view the training loss curve, step latency, and runtime logs:

```{code-cell} ipython3
%%bash
areno dashboard --start
```

Once started, open [http://127.0.0.1:8765](http://127.0.0.1:8765) in your browser and check the training run under Jobs:

![Ling-3.0-tiny 500-step training run in the AReno Dashboard](../../../resources/assets/mac-mlx-payment-lora/areno-dashboard.png)

*Figure 1: Record of the 500-step training run in the AReno Dashboard.*

+++

<a id="inference"></a>

### Step 3: Launch the Inference Service

Use `areno serve` to load the base model together with the LoRA adapter saved at step 500 (`runs/q4-cli/adapters/step_000500`), starting an OpenAI-compatible HTTP inference service on local port `8000`:

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

### Step 4: Run Automated Evaluation on 500 Test Cases

After the background inference server finishes loading the model and begins listening on port `8000`, run the evaluation script below to perform batch inference across the 500 cases in `eval500.jsonl` and save the outputs to `eval_results.jsonl`.

Scoring rule: extract the final JSON text after `</think>` from the model output and compare it field by field against `ground_truth`. All field values and array orders must match exactly (JSON object key order does not matter); any response that fails JSON parsing or differs in value is counted as incorrect:

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
        # Send only the rules and question, excluding the reference assistant answer
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
    print(f"Completed {i}/{len(cases)}", end="\r")

Path("eval_results.jsonl").write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n",
    encoding="utf-8",
)
correct = sum(row["correct"] for row in results)
print(f"\nTest accuracy: {correct}/{len(cases)} = {correct / len(cases):.2%}")
```

+++

<a id="results"></a>

## 7. Typical Results

Measured training and evaluation results on a MacBook Air (M5, 16GB unified memory) are as follows:

- Training Cost: 500 training steps with 74,947,584 trainable LoRA parameters completed in 3 hours, 15 minutes, and 55 seconds, with an MLX memory allocator peak of approximately 10.03GB.
- Evaluation Performance: Across the 500 test cases, the fine-tuned model answered 322 cases completely right, achieving a strict JSON exact-match accuracy of 64.4% (compared with the pre-fine-tuning baseline of 0% under the same strict matching criterion—where the base model could not perform this task at all).

![Ling-3.0-tiny evaluation before and after fine-tuning on 500 payment cases](../../../resources/assets/mac-mlx-payment-lora/evaluation-before-after.png)

*Figure 2: Overall strict exact-match accuracy before and after fine-tuning on the left, and post-fine-tuning correct counts / total cases by sub-scenario on the right (scenario labels follow the original test set).*

+++

## 8. Troubleshooting and Further Exploration

### Troubleshooting & FAQ

- Encountering `externally-managed-environment` when running `pip install -e AReno`
  - System Python and Homebrew Python on macOS enable PEP 668 protection by default. Create and activate a virtual environment first using `uv venv .venv --python 3.12 && source .venv/bin/activate` (or `python3 -m venv .venv`), then run the installation command.

- Encountering `Connection refused` when running the evaluation script
  - `areno serve` in Step 3 starts asynchronously in the background via `%%bash --bg`, and loading the base weights plus injecting the LoRA adapter takes some time. If Step 4 is executed immediately, the server is not yet listening on port `8000`. Wait about 30 to 60 seconds, or verify in the terminal via `lsof -i :8000` that the port is in `LISTEN` state before starting evaluation.

- Noticeable system lag during training on a 16GB unified memory Mac
  - Peak MLX memory during training is around 10.03GB, leaving limited headroom for macOS and other applications. Close excess browser tabs, Docker Desktop, or other resident local model processes before training to reduce disk swap activity.

- Why `--no-mlx-compile-train-step` is enabled in the training command
  - Disabling `mx.compile` graph compilation for the per-step loss and gradient function reduces recompilation overhead and extra memory spikes on variable-length sequences, keeping execution stable on 16GB unified memory devices.

### What Else Can You Explore?

If you want to push post-training performance further beyond this 64.4% baseline, consider the following directions:

- Introduce Reinforcement Learning with Verifiable Rewards (RLVR)
  - Split-payment calculation has deterministic verification criteria. Once the model learns the output format and basic reasoning steps through SFT, you can use the reinforcement learning algorithms supported by AReno (see [DPO / GRPO / GSPO / PPO Training Docs](https://asystem-ai.io/docs/areno/cli/training.html) and [Agentic RL Docs](https://asystem-ai.io/docs/areno/getting-started/welcome.html#agentic-rollout)) to encode JSON validity, promotion rule compliance, and exact payable amount matching as a rule-based reward function, reducing arithmetic and branching errors through multi-sample rollouts.

- Expand Synthetic Data for Weak Sub-Scenarios
  - Inspect the per-scenario breakdown on the right side of Figure 2 to identify complex cases with higher error rates (such as multiple promotions near threshold boundaries or cross-channel split edge cases), use a deterministic Python settlement script to synthesize more hard examples of those types, and include them for incremental fine-tuning.

- Increase LoRA Adapter Capacity and Training Steps
  - To fit within 16GB unified memory, this guide uses `--lora-rank 4` and 500 training steps. On a Mac with 24GB/36GB+ unified memory or a Linux GPU machine with larger VRAM, you can increase `--lora-rank` to 8 or 16, train for more steps, or combine training with the [8-bit AdamW](https://asystem-ai.io/docs/areno/concepts/adamw-8bit.html) optimizer to observe the convergence ceiling of validation loss and test accuracy.

+++

## 9. Acknowledgments

- Thanks to [@sofan110](https://github.com/sofan110) for providing this fine-tuning recipe.
