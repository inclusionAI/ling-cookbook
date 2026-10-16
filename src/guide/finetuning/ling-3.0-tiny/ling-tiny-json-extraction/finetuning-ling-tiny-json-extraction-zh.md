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

# Ling-3.0-tiny 发票字段抽取微调实战（Kaggle / Colab · QLoRA）

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

本 Recipe 在**单块免费 T4（16GB 显存）** 上，用 **QLoRA（4-bit NF4）** 对 `Ling-3.0-tiny`（7.9B 稀疏 MoE / 1.3B 激活参数）做监督微调，使其胜任一个真实的下游任务：**中文发票/票据的结构化字段抽取**（把一段发票文本转成固定 schema 的 JSON）。

全程只依赖一个 Notebook 加两份配置文件，可在 **Kaggle Notebooks** 或 **Google Colab** 免费环境下复现。数据集由程序合成，不依赖任何外部标注数据；训练完成后对比「基座模型 vs 微调模型」的字段级抽取质量。

> [!NOTE]
> 本 Recipe 同时是 `ling-cookbook` 中 `guide/finetuning/` 章节的参考实现，可直接作为模板迁移到其他结构化抽取 / JSON 生成任务。

---

### 概览

| 项目 | 说明 |
| :--- | :--- |
| 基座模型 | `inclusionAI/Ling-3.0-tiny`（Sparse MoE，7.9B 总参 / 1.3B 激活） |
| 微调方法 | QLoRA：4-bit NF4 量化 + LoRA（`r=8`, `alpha=16`），仅训练约 0.5% 参数 |
| 训练精度 | `fp16`（T4 为 Turing 架构，**不支持 bf16**） |
| 任务 | 中文发票字段抽取 → 结构化 JSON |
| 数据 | 程序合成，默认 1000 条（800 训练 / 200 测试） |
| 运行环境 | Kaggle / Colab 免费 T4；开启 GPU 与 Internet |
| 交付物 | LoRA adapter（约几十 MB）+ 评测报告 + 推理示例 |
| 预计耗时 | 数据合成 < 1 分钟；训练（3 epoch）约 15–25 分钟；评测约 3–5 分钟 |

字段 schema（10 个字段）：

```json
{
  "invoice_code": "发票代码",
  "invoice_number": "发票号码",
  "invoice_date": "开票日期（YYYY-MM-DD）",
  "seller_name": "销售方名称",
  "seller_tax_id": "销售方纳税人识别号",
  "buyer_name": "购买方名称",
  "buyer_tax_id": "购买方纳税人识别号",
  "amount": "金额（不含税，元，数字）",
  "tax_amount": "税额（元，数字）",
  "total_amount": "价税合计（元，数字）"
}
```

> [!IMPORTANT]
> **T4 显存警告**：`Ling-3.0-tiny` 的 4-bit 量化权重约 5–6GB，加上优化器状态与激活值，建议 `batch_size=1` + `gradient_accumulation_steps=8` + `gradient_checkpointing=True`。若在 16GB 卡上出现 OOM，请下调 `max_seq_length`（如 768）。

+++

### Step 0：环境准备

在 Kaggle 上新建 Notebook，右侧 Settings 中开启 **Accelerator = GPU T4 x1** 与 **Internet = On**；Colab 请选择 **Runtime → Change runtime type → T4 GPU**。

安装微调依赖（首次安装约 1–2 分钟）：

```{code-cell}
# 安装 QLoRA 微调所需依赖
!pip -q install -U "transformers>=4.45" "peft>=0.13" "trl>=0.11" "datasets>=2.20" \
    "accelerate>=0.34" "bitsandbytes>=0.43" "huggingface_hub" "sentencepiece" "protobuf"

import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "-")
```

预期输出（T4）：

```text
torch: 2.x.x+cu121
CUDA available: True
GPU: Tesla T4
capability: (7, 5)
```

> [!NOTE]
> `capability` 为 `(7, 5)` 表示 Turing 架构，仅支持 `fp16`，不支持 `bf16`。若你的卡是 Ampere（`(8, 0)` 及以上），可把后续 `fp16` 改为 `bf16` 以获得更好的稳定性。

+++

### Step 1：程序合成发票数据集

我们用一个模板化合成器生成「发票文本 → 标签 JSON」样本，覆盖多城市、多销售/购买方、多税率与多版式，天然避免标注成本。

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
TAX_CHARS = "0123456789ABCDEFGHJKLMNPQRTUWXY"  # 纳税人识别号字符集（去除易混淆字符）

def rand_tax_id(rng):
    return "".join(rng.choice(TAX_CHARS) for _ in range(18))

def rand_date(rng):
    d = date(2024, 1, 1) + timedelta(days=rng.randint(0, 900))
    return d.strftime("%Y-%m-%d")

# 多种版式（字段顺序 / 措辞 / 表头噪声不同），提升泛化
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
print("---- 样例（user 输入）----")
print(train_samples[0]["messages"][1]["content"])
print("---- 样例（assistant 标签）----")
print(train_samples[0]["messages"][2]["content"])
```

预期输出（截断）：

```text
train: 800  test: 200
---- 样例（user 输入）----
发票文本：
发票代码：21xxxxxxxx
发票号码：43xxxxxx
开票日期：2025-xx-xx
购买方名称：上海xx集团有限公司
...
---- 样例（assistant 标签）----
{"invoice_code": "21xxxxxxxx", "invoice_number": "43xxxxxx", ..., "total_amount": 12345.67}
```

+++

### Step 1.5：定义评测工具

评测口径：对测试集做贪心解码，用正则提取输出中的 JSON；统计**字段级 micro Precision / Recall / F1** 与 **JSON 合法率**。金额类字段以 0.01 容差比较。这些纯函数在任何模型加载之前定义好，便于后续对「基座」与「微调」复用同一套口径。

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

### Step 2：加载基座模型（4-bit NF4）并确认 LoRA 目标层

先加载 tokenizer 与 4-bit 量化模型。与普通稠密模型不同，`Ling-3.0-tiny` 是 **MoE** 架构，其专家层（`gate_proj` / `up_proj` / `down_proj`）也在 FFN 内，因此**必须先打印层名确认 `target_modules` 是否存在**，再决定 LoRA 挂载点。

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
# 打印所有叶子模块的短名，确认 attention / MoE 专家层的真实命名
leaf_names = sorted({name.split(".")[-1] for name, _ in model.named_modules()})
print("leaf module names:", leaf_names)
```

```{code-cell}
# 依据上一步输出做健壮匹配：命中的层才会被 LoRA 挂载
CANDIDATE_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj", "gate", "w1", "w2", "w3"]
present = [n for n in CANDIDATE_TARGETS if n in leaf_names]
missing = [n for n in CANDIDATE_TARGETS if n not in leaf_names]
print("will attach to :", present)
print("not found      :", missing)
assert present, "未匹配到任何目标层，请根据上一步的层名调整 CANDIDATE_TARGETS"
```

#### 基线评测（务必在挂载 LoRA 之前执行）

```{code-cell}
base_preds = run_infer(model)
base_score = score(base_preds, gold_labels)
print("Base (Ling-3.0-tiny):", base_score)
```

> [!WARNING]
> MoE 模型若只挂 `q_proj/k_proj/v_proj/o_proj`（仅注意力），往往「学得慢」或效果不佳。本 Recipe 默认同时覆盖注意力投影与 FFN/专家投影（`gate_proj/up_proj/down_proj`）。若你的模型层名是 `w1/w2/w3` 或 `gate` 等，请据实修改 `CANDIDATE_TARGETS`。

+++

### Step 3：配置 LoRA

配置文件 `configs/lora_config.json` 与 `configs/train_config.json` 与本 Notebook 同目录。为兼容 Kaggle 上「只上传单个 ipynb」的场景，读取失败时回退到文件内的等价默认值。

```{code-cell}
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_json_with_fallback(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"loaded config: {path}")
        return cfg
    except FileNotFoundError:
        print(f"[warn] {path} 未找到，使用内置默认值")
        return default

lora_default = {
    "r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
}
lora_cfg = load_json_with_fallback("configs/lora_config.json", lora_default)
# 用「实际存在的层名」覆盖配置里的 target_modules，避免挂载失败
lora_cfg["target_modules"] = present
print("lora target_modules:", lora_cfg["target_modules"])

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(**lora_cfg)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

预期输出：

```text
trainable params: ~6,000,000 || all params: ~8,000,000,000 || trainable%: ~0.07
```

+++

### Step 4：训练

先把 `messages` 渲染成文本（使用模型自带的 chat template），再交给 `SFTTrainer`。

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
> T4 上 3 epoch 大约 15–25 分钟。若使用 Ampere 及以上显卡，把 `fp16` 改为 `bf16=True` 通常更稳定；同时可把 `max_seq_length` 提升到 2048。

+++

### Step 5：微调后评测与对比

```{code-cell}
ft_preds = run_infer(model)
ft_score = score(ft_preds, gold_labels)
print("Base  (Ling-3.0-tiny):", base_score)
print("QLoRA (finetuned)    :", ft_score)

import pandas as pd
df = pd.DataFrame({"Base (Ling-3.0-tiny)": base_score, "QLoRA Finetuned": ft_score}).T
print(df)
```

预期结果（T4，合成数据集，供参考；实际数值随随机种子波动）：

| 模型 | Precision | Recall | F1 | JSON 合法率 |
| :--- | :-: | :-: | :-: | :-: |
| Base（Ling-3.0-tiny） | ~0.7–0.9 | ~0.7–0.9 | ~0.7–0.9 | ~0.85–1.0 |
| QLoRA 微调 | ~0.95+ | ~0.95+ | ~0.95+ | ~1.0 |

+++

### Step 6：推理示例

挑 2 条测试样本，直观对比微调前后的输出：

```{code-cell}
for i in range(2):
    print("=" * 60)
    print("[输入]\n", test_samples[i]["messages"][1]["content"][:300])
    print("[标签]\n", json.dumps(gold_labels[i], ensure_ascii=False))
    print("[微调输出]\n", json.dumps(ft_preds[i], ensure_ascii=False) if ft_preds[i] else ft_preds[i])
```

+++

### Step 7：保存与上传

```{code-cell}
SAVE_DIR = "lora-adapter"
trainer.save_model(SAVE_DIR)
model_tokenizer.save_pretrained(SAVE_DIR)
print("saved to", SAVE_DIR, "->", os.listdir(SAVE_DIR))
```

上传到 Hugging Face（需先在 Kaggle Secrets / Colab 中配置 `HF_TOKEN`；**请勿把令牌写进 Notebook**）：

```{code-cell}
# 取消注释并填入你的仓库名即可上传
# from huggingface_hub import HfApi, login
# login(token=os.environ["HF_TOKEN"])
# api = HfApi()
# api.create_repo(repo_id="<your-name>/ling-tiny-json-extraction-lora", repo_type="model", exist_ok=True)
# api.upload_folder(repo_id="<your-name>/ling-tiny-json-extraction-lora", folder_path="lora-adapter")
# api.upload_file(repo_id="<your-name>/ling-tiny-json-extraction-dataset", path_or_fileobj="data/train.jsonl",
#                 path_in_repo="train.jsonl", repo_type="dataset")
print("如需上传，请在配置 HF_TOKEN 后取消上面代码的注释。")
```

+++

### Step 8：AI 使用策略报告

本 Recipe 中，**人类负责**：确定任务边界（发票字段抽取）、定义字段 schema、审核合成数据的字段合理性、检查微调效果是否达标。**AI（编码助手）负责**：根据已确认的设计编写数据合成器、训练/评测脚本与文档，并在执行过程中遵循「人决策、AI 执行」的协作约定。数据层面，训练语料由本地脚本**程序合成**，不含任何真实票据或个人信息，规避了隐私与合规风险；模型层面，`Ling-3.0-tiny` 的权重来自公开的 Hugging Face 仓库，微调仅产出与任务相关的 LoRA adapter，不改变基座分发。

在整个实现与验证过程中，AI 生成的每一处关键代码或结论都以「可复现命令 + 可核对输出」为验收标准：例如评测结果需由 `score()` 在固定种子下产生、文档中的字段 schema 需与合成器代码逐字一致、编译产物需通过 `scripts/compile_notebooks.py --check`。任何偏离方案或新增的取舍（如 `target_modules` 的具体挂载点），均先与人类确认后再执行，确保最终交付物可解释、可追溯、可复现。

+++

### 常见问题（FAQ）

1. **OOM（CUDA out of memory）**：将 `max_seq_length` 降至 768，或将 `gradient_accumulation_steps` 提高到 16（同时保持 `per_device_train_batch_size=1`）。
2. **`target_modules` 未命中 / 挂载后几乎不训练**：回到 Step 2 打印 `leaf_names`，把实际层名填进 `CANDIDATE_TARGETS`；MoE 模型通常需要同时覆盖注意力与专家 FFN 投影。
3. **输出不是合法 JSON**：`parse_json()` 已做正则兜底；训练数据里 JSON 字段顺序稳定也有助于模型收敛。
4. **bf16 报错**：T4 不支持 bf16，务必使用 `fp16=True`；Ampere+ 才可切 `bf16`。
5. **Kaggle 无网**：请在右侧 Settings 打开 **Internet**；若只需离线运行，可预先 `pip download` 依赖并上传为 Kaggle Dataset。
6. **只想单文件运行**：`configs/` 读取失败时会自动回退到内置默认值，Notebook 可直接跑通。
