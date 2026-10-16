---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
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

# Fine-tuning Ling-3.0-tiny for Invoice Field Extraction (Kaggle / Colab · QLoRA)

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
    <a target="_blank" href="https://x.com/AntLingAGI"><img src="https://img.shields.io/badge/X-@AntLingAGI-000000?style=flat-square&logo=x" alt="X" /></a>
  </td>
  <td>
    <a target="_blank" href="https://discord.com/invite/GNaQc8WC5T"><img src="https://img.shields.io/badge/Discord-Ling_Community-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord" /></a>
  </td>
</table>

<br><br>

This recipe fine-tunes `Ling-3.0-tiny` (a 7.9B sparse MoE model with 1.3B active parameters) with **QLoRA (4-bit NF4)** on a **single free T4 (16GB)** GPU to solve a real downstream task: **structured field extraction from Chinese invoices** (turning invoice text into a fixed-schema JSON object).

Everything runs from one notebook plus two config files, reproducible on the free tiers of **Kaggle Notebooks** or **Google Colab**. The dataset is procedurally synthesized (no external annotation required), and we compare the **base model vs the fine-tuned model** at field level.

> [!NOTE]
> This recipe doubles as the reference implementation for the `guide/finetuning/` chapter of `ling-cookbook`, and can be adapted as a template for other structured-extraction / JSON-generation tasks.

---

### Overview

| Item | Description |
| :--- | :--- |
| Base model | `inclusionAI/Ling-3.0-tiny` (sparse MoE, 7.9B total / 1.3B active) |
| Method | QLoRA: 4-bit NF4 + LoRA (`r=8`, `alpha=16`), ~0.5% trainable params |
| Train precision | `fp16` (T4 is Turing — **no bf16 support**) |
| Task | Chinese invoice field extraction → structured JSON |
| Data | Procedural, 1000 samples by default (800 train / 200 test) |
| Environment | Kaggle / Colab free T4; GPU + Internet enabled |
| Deliverables | LoRA adapter (tens of MB) + evaluation report + inference sample |
| Runtime | Data synthesis < 1 min; training (3 epochs) ~15–25 min; eval ~3–5 min |

Field schema (10 fields):

```json
{
  "invoice_code": "Invoice code",
  "invoice_number": "Invoice number",
  "invoice_date": "Issue date (YYYY-MM-DD)",
  "seller_name": "Seller name",
  "seller_tax_id": "Seller tax ID",
  "buyer_name": "Buyer name",
  "buyer_tax_id": "Buyer tax ID",
  "amount": "Amount excluding tax (CNY, number)",
  "tax_amount": "Tax amount (CNY, number)",
  "total_amount": "Total incl. tax (CNY, number)"
}
```

> [!IMPORTANT]
> **T4 memory warning**: the 4-bit weights of `Ling-3.0-tiny` are ~5–6GB; with optimizer states and activations, use `batch_size=1` + `gradient_accumulation_steps=8` + `gradient_checkpointing=True`. If you hit OOM on a 16GB card, lower `max_seq_length` (e.g. 768).

+++

### Step 0: Environment Setup

On Kaggle, create a notebook and turn on **Accelerator = GPU T4 x1** and **Internet = On**; on Colab pick **Runtime → Change runtime type → T4 GPU**.

Install the fine-tuning dependencies (first install takes ~1–2 min):

```{code-cell}
# Install dependencies for QLoRA fine-tuning
!pip -q install -U "transformers>=4.45" "peft>=0.13" "trl>=0.11" "datasets>=2.20" \
    "accelerate>=0.34" "bitsandbytes>=0.43" "huggingface_hub" "sentencepiece" "protobuf"

import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "-")
```

Expected output (T4):

```text
torch: 2.x.x+cu121
CUDA available: True
GPU: Tesla T4
capability: (7, 5)
```

> [!NOTE]
> A `capability` of `(7, 5)` means Turing, which supports only `fp16`, not `bf16`. On Ampere (`(8, 0)` or higher) you can switch `fp16` to `bf16` for better stability.

+++

### Step 1: Procedurally Synthesize the Invoice Dataset

We use a template-based synthesizer to generate "invoice text → label JSON" pairs, covering multiple cities, sellers/buyers, tax rates, and layouts — avoiding any annotation cost.

```{code-cell}
import json, os, re, random
from datetime import date, timedelta

SEED = 42
random.seed(SEED)

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "重庆", "苏州", "天津"]
SELLER_SUFFIX = ["科技有限公司", "贸易有限公司", "信息技术有限公司", "实业有限公司", "供应链管理有限公司", "电子商务有限公司"]
BUYER_SUFFIX = ["集团有限公司", "股份有限公司", "文化传媒有限公司", "物流有限公司", "电子有限公司", "制造有限公司"]
ITEMS = ["办公用品", "技术服务费", "软件开发费", "咨询服务费", "设备采购", "云服务费", "广告服务费", "运输服务费"]
TAX_RATES = [0.01, 0.03, 0.06, 0.09, 0.13]
TAX_CHARS = "0123456789ABCDEFGHJKLMNPQRTUWXY"  # tax-ID charset (ambiguous chars removed)

def rand_tax_id(rng):
    return "".join(rng.choice(TAX_CHARS) for _ in range(18))

def rand_date(rng):
    d = date(2024, 1, 1) + timedelta(days=rng.randint(0, 900))
    return d.strftime("%Y-%m-%d")

# Multiple layouts (different field order / wording / header noise) to improve generalization
TEMPLATES = [
    "发票代码：{invoice_code}\n发票号码：{invoice_number}\n开票日期：{invoice_date}\n"
    "购买方名称：{buyer_name}\n购买方纳税人识别号：{buyer_tax_id}\n"
    "销售方名称：{seller_name}\n销售方纳税人识别号：{seller_tax_id}\n"
    "项目名称：{item}\n金额：{amount} 元\n税额：{tax_amount} 元\n价税合计：{total_amount} 元",

    "电子发票（普通发票）\n"
    "销售方：{seller_name}\n纳税人识别号：{seller_tax_id}\n"
    "购买方：{buyer_name}\n纳税人识别号：{buyer_tax_id}\n"
    "项目：{item}\n开票日期：{invoice_date}\n"
    "发票代码 {invoice_code}  发票号码 {invoice_number}\n"
    "金额（不含税）：{amount}\n税额：{tax_amount}\n价税合计：{total_amount}",

    "发票号码:{invoice_number} 发票代码:{invoice_code}\n"
    "开票日期 {invoice_date}\n"
    "购 买 方 名 称: {buyer_name}\n购 买 方 税 号: {buyer_tax_id}\n"
    "销 售 方 名 称: {seller_name}\n销 售 方 税 号: {seller_tax_id}\n"
    "货物或应税劳务名称: {item}\n合计金额: {amount}\n合计税额: {tax_amount}\n价税合计(大写略): {total_amount}",
]

SYSTEM_PROMPT = (
    "你是一名发票信息抽取助手。请阅读用户提供的发票文本，抽取字段后"
    "严格输出一个 JSON 对象，不要输出任何解释性文字。"
    "字段包括：invoice_code, invoice_number, invoice_date, seller_name, seller_tax_id, "
    "buyer_name, buyer_tax_id, amount, tax_amount, total_amount。"
)

def make_invoice(rng):
    amount = round(rng.uniform(50, 50000), 2)
    rate = rng.choice(TAX_RATES)
    tax = round(amount * rate, 2)
    total = round(amount + tax, 2)
    inv = {
        "invoice_code": str(rng.randint(1000000000, 4999999999)),
        "invoice_number": str(rng.randint(10000000, 99999999)),
        "invoice_date": rand_date(rng),
        "seller_name": rng.choice(CITIES) + rng.choice(SELLER_SUFFIX),
        "seller_tax_id": rand_tax_id(rng),
        "buyer_name": rng.choice(CITIES) + rng.choice(BUYER_SUFFIX),
        "buyer_tax_id": rand_tax_id(rng),
        "amount": amount,
        "tax_amount": tax,
        "total_amount": total,
    }
    item = rng.choice(ITEMS)
    text = rng.choice(TEMPLATES).format(item=item, **inv)
    return text, inv

def build_sample(rng):
    text, inv = make_invoice(rng)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"发票文本：\n{text}\n\n请抽取字段并以 JSON 输出。"},
            {"role": "assistant", "content": json.dumps(inv, ensure_ascii=False)},
        ]
    }

N_TOTAL, N_TEST = 1000, 200
rng = random.Random(SEED)
samples = [build_sample(rng) for _ in range(N_TOTAL)]
rng.shuffle(samples)

test_samples = samples[:N_TEST]
train_samples = samples[N_TEST:]

os.makedirs("data", exist_ok=True)
def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

dump_jsonl("data/train.jsonl", train_samples)
dump_jsonl("data/test.jsonl", test_samples)

print(f"train: {len(train_samples)}  test: {len(test_samples)}")
print("---- sample (user input) ----")
print(train_samples[0]["messages"][1]["content"])
print("---- sample (assistant label) ----")
print(train_samples[0]["messages"][2]["content"])
```

Expected output (truncated):

```text
train: 800  test: 200
---- sample (user input) ----
发票文本：
发票代码：21xxxxxxxx
发票号码：43xxxxxx
开票日期：2025-xx-xx
购买方名称：上海xx集团有限公司
...
---- sample (assistant label) ----
{"invoice_code": "21xxxxxxxx", "invoice_number": "43xxxxxx", ..., "total_amount": 12345.67}
```

+++

### Step 1.5: Define the Evaluation Utilities

Protocol: greedy-decode on the test set, extract JSON from the output with a regex, and compute **field-level micro Precision / Recall / F1** plus the **JSON validity rate**. Numeric fields are compared with a 0.01 tolerance. These pure functions are defined before any model is loaded, so both the base and fine-tuned runs share the exact same protocol.

```{code-cell}
FIELDS = ["invoice_code", "invoice_number", "invoice_date", "seller_name", "seller_tax_id",
          "buyer_name", "buyer_tax_id", "amount", "tax_amount", "total_amount"]

def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def norm(v):
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    if v is None:
        return None
    return str(v).strip()

def equal(a, b):
    if isinstance(b, float) and isinstance(a, float):
        return abs(a - b) < 0.01
    return a == b

def score(preds, golds):
    tp = fp = fn = valid = 0
    for p, g in zip(preds, golds):
        if p is None:
            fn += len(FIELDS)
            continue
        valid += 1
        for k in FIELDS:
            gv, pv = norm(g.get(k)), norm(p.get(k))
            if pv is not None and equal(pv, gv):
                tp += 1
            else:
                fn += 1
                if pv is not None:
                    fp += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "json_valid_rate": round(valid / len(golds), 4)}

gold_labels = [json.loads(s["messages"][2]["content"]) for s in test_samples]
print("gold labels:", len(gold_labels), "| sample key count:", len(gold_labels[0]))
```

```{code-cell}
def run_infer(model, max_new_tokens=256):
    model.eval()
    preds = []
    for s in test_samples:
        prompt = model_tokenizer.apply_chat_template(
            s["messages"][:2], tokenize=False, add_generation_prompt=True
        )
        inputs = model_tokenizer(prompt, return_tensors="pt").to(model.device)
        import torch
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=model_tokenizer.eos_token_id,
            )
        gen = model_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append(parse_json(gen))
    return preds
```

+++

### Step 2: Load the Base Model (4-bit NF4) and Confirm LoRA Targets

We first load the tokenizer and the 4-bit quantized model. Unlike a dense model, `Ling-3.0-tiny` is a **MoE** architecture whose expert layers (`gate_proj` / `up_proj` / `down_proj`) also live inside the FFN, so you **must print the module names first** to confirm which `target_modules` exist before attaching LoRA.

```{code-cell}
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "inclusionAI/Ling-3.0-tiny"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  # T4: fp16
)

model_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if model_tokenizer.pad_token is None:
    model_tokenizer.pad_token = model_tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
print("model loaded:", type(model).__name__)
print("num params (billions):", round(sum(p.numel() for p in model.parameters()) / 1e9, 2))
```

```{code-cell}
# Print the short names of all leaf modules to confirm attention / MoE expert naming
leaf_names = sorted({name.split(".")[-1] for name, _ in model.named_modules()})
print("leaf module names:", leaf_names)
```

```{code-cell}
# Robust matching based on the previous output: only matched layers get LoRA attached
CANDIDATE_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj", "gate", "w1", "w2", "w3"]
present = [n for n in CANDIDATE_TARGETS if n in leaf_names]
missing = [n for n in CANDIDATE_TARGETS if n not in leaf_names]
print("will attach to :", present)
print("not found      :", missing)
assert present, "No target layer matched; adjust CANDIDATE_TARGETS based on the names printed above"
```

#### Baseline evaluation (must run before LoRA is attached)

```{code-cell}
base_preds = run_infer(model)
base_score = score(base_preds, gold_labels)
print("Base (Ling-3.0-tiny):", base_score)
```

> [!WARNING]
> For MoE models, attaching LoRA only to `q_proj/k_proj/v_proj/o_proj` (attention only) often learns slowly or underperforms. This recipe covers both attention projections and FFN/expert projections (`gate_proj/up_proj/down_proj`). If your model uses `w1/w2/w3` or `gate` naming, adjust `CANDIDATE_TARGETS` accordingly.

+++

### Step 3: Configure LoRA

The config files `configs/lora_config.json` and `configs/train_config.json` live next to this notebook. For the Kaggle case where only a single ipynb is uploaded, a fallback to equivalent inline defaults is used when the files are missing.

```{code-cell}
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_json_with_fallback(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"loaded config: {path}")
        return cfg
    except FileNotFoundError:
        print(f"[warn] {path} not found, using inline defaults")
        return default

lora_default = {
    "r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
}
lora_cfg = load_json_with_fallback("configs/lora_config.json", lora_default)
# Override target_modules with the layers that actually exist, to avoid attach failures
lora_cfg["target_modules"] = present
print("lora target_modules:", lora_cfg["target_modules"])

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(**lora_cfg)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

Expected output:

```text
trainable params: ~6,000,000 || all params: ~8,000,000,000 || trainable%: ~0.07
```

+++

### Step 4: Training

We render `messages` into text (using the model's own chat template), then hand it to `SFTTrainer`.

```{code-cell}
from datasets import load_dataset

raw = load_dataset("json", data_files={"train": "data/train.jsonl", "test": "data/test.jsonl"})

def render(example):
    try:
        text = model_tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    except Exception:
        text = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in example["messages"])
    return {"text": text}

train_ds = raw["train"].map(render, remove_columns=raw["train"].column_names)
test_ds = raw["test"].map(render, remove_columns=raw["test"].column_names)
print(train_ds[0]["text"][:400])
```

```{code-cell}
from trl import SFTTrainer, SFTConfig

train_default = {
    "model_id": MODEL_ID, "max_seq_length": 1024,
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4, "num_train_epochs": 3, "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine", "logging_steps": 10, "save_strategy": "epoch",
    "fp16": True, "bf16": False, "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit", "seed": 42,
}
tcfg = load_json_with_fallback("configs/train_config.json", train_default)

sft_args = SFTConfig(
    output_dir="outputs/ling-tiny-json-extraction",
    max_seq_length=tcfg["max_seq_length"],
    per_device_train_batch_size=tcfg["per_device_train_batch_size"],
    gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
    learning_rate=float(tcfg["learning_rate"]),
    num_train_epochs=tcfg["num_train_epochs"],
    warmup_ratio=tcfg["warmup_ratio"],
    lr_scheduler_type=tcfg["lr_scheduler_type"],
    logging_steps=tcfg["logging_steps"],
    save_strategy=tcfg["save_strategy"],
    save_total_limit=tcfg.get("save_total_limit", 1),
    fp16=bool(tcfg["fp16"]),
    bf16=bool(tcfg["bf16"]),
    gradient_checkpointing=bool(tcfg["gradient_checkpointing"]),
    optim=tcfg["optim"],
    seed=tcfg["seed"],
    report_to="none",
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=model_tokenizer,
    args=sft_args,
    train_dataset=train_ds,
)

train_result = trainer.train()
print({k: round(v, 4) for k, v in train_result.metrics.items() if isinstance(v, float)})
```

> [!NOTE]
> On a T4, 3 epochs take roughly 15–25 minutes. On Ampere or newer, switch `fp16` to `bf16=True` for better stability, and you can raise `max_seq_length` to 2048.

+++

### Step 5: Post-fine-tuning Evaluation and Comparison

```{code-cell}
ft_preds = run_infer(model)
ft_score = score(ft_preds, gold_labels)
print("Base  (Ling-3.0-tiny):", base_score)
print("QLoRA (finetuned)    :", ft_score)

import pandas as pd
df = pd.DataFrame({"Base (Ling-3.0-tiny)": base_score, "QLoRA Finetuned": ft_score}).T
print(df)
```

Expected results (T4, synthetic dataset, for reference; actual numbers vary with the random seed):

| Model | Precision | Recall | F1 | JSON validity |
| :--- | :-: | :-: | :-: | :-: |
| Base (Ling-3.0-tiny) | ~0.7–0.9 | ~0.7–0.9 | ~0.7–0.9 | ~0.85–1.0 |
| QLoRA fine-tuned | ~0.95+ | ~0.95+ | ~0.95+ | ~1.0 |

+++

### Step 6: Inference Examples

Pick 2 test samples and eyeball the before/after outputs:

```{code-cell}
for i in range(2):
    print("=" * 60)
    print("[input]\n", test_samples[i]["messages"][1]["content"][:300])
    print("[label]\n", json.dumps(gold_labels[i], ensure_ascii=False))
    print("[finetuned output]\n", json.dumps(ft_preds[i], ensure_ascii=False) if ft_preds[i] else ft_preds[i])
```

+++

### Step 7: Save and Upload

```{code-cell}
SAVE_DIR = "lora-adapter"
trainer.save_model(SAVE_DIR)
model_tokenizer.save_pretrained(SAVE_DIR)
print("saved to", SAVE_DIR, "->", os.listdir(SAVE_DIR))
```

Upload to Hugging Face (configure `HF_TOKEN` in Kaggle Secrets / Colab first; **never hard-code the token in the notebook**):

```{code-cell}
# Uncomment and fill in your own repo names to upload
# from huggingface_hub import HfApi, login
# login(token=os.environ["HF_TOKEN"])
# api = HfApi()
# api.create_repo(repo_id="wanglin/ling-tiny-json-extraction-lora", repo_type="model", exist_ok=True)
# api.upload_folder(repo_id="wanglin/ling-tiny-json-extraction-lora", folder_path="lora-adapter")
# api.create_repo(repo_id="wanglin/ling-tiny-json-extraction-dataset", repo_type="dataset", exist_ok=True)
# for f in ["data/train.jsonl", "data/test.jsonl"]:
#     api.upload_file(repo_id="wanglin/ling-tiny-json-extraction-dataset",
#                     path_or_fileobj=f, path_in_repo=f.split("/")[-1], repo_type="dataset")
print("To upload, configure HF_TOKEN and uncomment the code above.")
```

**Assets**: LoRA adapter and dataset produced by this recipe.

| Asset | Link |
| :--- | :--- |
| LoRA Adapter | `https://huggingface.co/wanglin/ling-tiny-json-extraction-lora` |
| Dataset (train / test) | `https://huggingface.co/datasets/wanglin/ling-tiny-json-extraction-dataset` |

+++

### Step 8: AI Usage Strategy Report

In this recipe, **the human** defines the task boundary (invoice field extraction), specifies the field schema, reviews the plausibility of the synthesized fields, and decides whether the fine-tuning result is good enough. **The AI (coding assistant)** implements the data synthesizer, the training/evaluation scripts, and the documentation from the confirmed design, following a "human decides, AI executes" collaboration contract. On the data side, the training corpus is **procedurally synthesized** by a local script and contains no real invoices or personal information, mitigating privacy and compliance risk; on the model side, `Ling-3.0-tiny` weights come from a public Hugging Face repository, and fine-tuning only produces a task-specific LoRA adapter without altering the base distribution.

Throughout implementation and verification, every key piece of AI-generated code or conclusion is held to a "reproducible command + checkable output" standard: evaluation numbers must come from `score()` under a fixed seed, the field schema in the docs must match the synthesizer code verbatim, and the compiled artifact must pass `scripts/compile_notebooks.py --check`. Any deviation from the plan or new trade-off (such as the exact `target_modules` attachment points) is confirmed with the human before execution, keeping the final deliverable explainable, traceable, and reproducible.

+++

### FAQ

1. **OOM (CUDA out of memory)**: lower `max_seq_length` to 768, or raise `gradient_accumulation_steps` to 16 (keeping `per_device_train_batch_size=1`).
2. **`target_modules` not matched / barely trains after attaching**: revisit Step 2 and print `leaf_names`, then put the real layer names into `CANDIDATE_TARGETS`. MoE models usually need both attention and expert-FFN projections covered.
3. **Output is not valid JSON**: `parse_json()` already adds a regex fallback; a stable JSON field order in training data also helps convergence.
4. **bf16 error**: T4 does not support bf16 — always use `fp16=True`; only Ampere+ can switch to `bf16`.
5. **Kaggle has no network**: turn on **Internet** in the right-hand Settings — it is required, because the base model is downloaded from Hugging Face. For offline-only runs, pre-download the wheels on a machine with network access using the same spec as Step 0 (`pip download -d wheels "transformers>=4.45" "peft>=0.13" "trl>=0.11" "datasets>=2.20" "accelerate>=0.34" "bitsandbytes>=0.43" "huggingface_hub" "sentencepiece" "protobuf"`; add `--platform manylinux2014_x86_64 --only-binary=:all: --python-version 3.10` if the downloading machine is not Linux), upload `wheels/` as a Kaggle Dataset, then install with `!pip install --no-index --find-links=/kaggle/input/<your-dataset>/wheels <same spec>`.
6. **Kaggle sessions do not persist artifacts**: on Kaggle, `/kaggle/working` starts empty in every new session, and the outputs of a previous run are **not** restored into the working directory. Re-running is safe and deterministic — Step 1 re-synthesizes `data/train.jsonl` / `data/test.jsonl` from the fixed `SEED=42` — so **there is nothing to upload by hand**. To keep the adapter, upload it to Hugging Face in Step 7 (recommended), or run *Save Version* and download it from the notebook output.
7. **Single-file run**: when `configs/` is missing, the notebook automatically falls back to inline defaults and still runs end to end.
