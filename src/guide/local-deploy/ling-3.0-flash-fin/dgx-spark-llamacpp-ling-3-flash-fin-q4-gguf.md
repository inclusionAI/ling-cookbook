---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  name: python3
  language: python
  display_name: Python 3 (ipykernel)
---

##### Copyright 2026 Ant Group.

+++

# Ling-3.0-flash-Fin on DGX Spark (llama.cpp Q4_K_M GGUF) Deployment Guide

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

+++

Ling-3.0-flash-Fin is the first domain-specialized financial foundation model in the Ling model family. Built upon Ling-3.0-flash with 124B total parameters and 5.1B activated parameters per token, it natively supports a 256K long-context window and is specifically enhanced for financial report analysis, metric reconciliation, multi-table alignment, and valuation modeling.

This notebook demonstrates how to deploy the 4-bit quantized (Q4_K_M GGUF) version of Ling-3.0-flash-Fin on an `NVIDIA DGX Spark` (GB10 121GB Unified Memory) system using `llama.cpp`. After quantization, the model weights occupy ~72 GB (~71.5 GB VRAM), leaving nearly 50 GB of memory for ultra-long context and multi-concurrency serving on a single device, achieving high throughput and low Time-to-First-Token (TTFT).

> [!TIP]
> **Environment Recommendations**:
> - **Python 3.11 / 3.12** is recommended.
> - Using **uv** to manage an isolated virtual environment ensures toolchain isolation and clean dependency management.

+++

### Step 1: Set Up Python Virtual Environment (uv) and Clone llama.cpp

Create an isolated virtual environment with `uv` and install the OpenAI Python library for endpoint testing. Upstream `llama.cpp` has integrated support for the Ling-3.0 architecture:

```{code-cell}
!pip install -U uv
!uv venv --python 3.12 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
!git clone https://github.com/ggerganov/llama.cpp.git
```

Typical output:
```text
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtualenv at: .venv
Cloning into 'llama.cpp'...
remote: Enumerating objects: 45210, done.
remote: Counting objects: 100% (210/210), done.
```

+++

### Step 2: Build llama.cpp

Build `llama.cpp` with CUDA acceleration enabled on DGX Spark:

```{code-cell}
!cd llama.cpp && cmake -B build -DGGML_CUDA=ON . && cmake --build build --parallel 8
```

Typical output:
```text
-- The CXX compiler identification is GNU 11.4.0
-- The CUDA compiler identification is NVIDIA 12.8.55
-- Building with CUDA architecture: native
[100%] Built target llama-server
[100%] Built target llama-quantize
```

+++

### Step 3: Obtain Ling-3.0-flash-Fin Model Weights

Download the original weights. If downloading within mainland China, the [ModelScope CLI](https://github.com/modelscope/modelscope/blob/master/README_zh.md) is recommended; alternatively, use the [Hugging Face CLI](https://huggingface.co/docs/hub/agents-cli):

- [Ling-3.0-flash-Fin on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-Fin)
- [Ling-3.0-flash-Fin on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-Fin)

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-Fin --local-dir ~/models/Ling-3.0-flash-Fin
```

Typical output:
```text
Downloading shard 1/24: 100%|██████████| 4.98G/4.98G [00:15<00:00, 332MB/s]
...
Successfully downloaded Ling-3.0-flash-Fin to ~/models/Ling-3.0-flash-Fin
```

+++

### Step 4: Convert Model to Full-Precision GGUF Format

Install the conversion dependencies and convert Safetensors weights into full-precision BF16 GGUF format using `convert_hf_to_gguf.py`:

```{code-cell}
!source .venv/bin/activate && uv pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
!source .venv/bin/activate && python3 llama.cpp/convert_hf_to_gguf.py ~/models/Ling-3.0-flash-Fin \
  --outfile ~/models/Ling-3.0-flash-Fin-bf16.gguf \
  --outtype bf16 --model-name Ling-3.0-flash-Fin
```

Typical output:
```text
INFO:hf-to-gguf:Loading model: Ling-3.0-flash-Fin
INFO:hf-to-gguf:Set model parameters
INFO:hf-to-gguf:Writing tensors to /home/squall/models/Ling-3.0-flash-Fin-bf16.gguf
INFO:hf-to-gguf:Done writing 124B model tensors (bf16).
```

+++

### Step 5: Quantize Model to Q4_K_M GGUF Format

The unquantized BF16 weights require ~248GB, which exceeds the memory capacity of a single DGX Spark node. Use `llama-quantize` to compress the model to 4-bit (Q4_K_M) format, reducing file size to ~60.5GB:

```{code-cell}
!./llama.cpp/build/bin/llama-quantize ~/models/Ling-3.0-flash-Fin-bf16.gguf ~/models/Ling-3.0-flash-Fin-Q4_K_M.gguf Q4_K_M
```

Typical output:
```text
llama_model_quantize_impl: model size  = 243267.57 MiB (16.01 BPW)
llama_model_quantize_impl: quant size  = 73436.36 MiB (4.83 BPW)
llama_model_quantize_impl: WARNING: 8 of 938 tensor(s) required fallback quantization

llama_quantize: quantize time = 533907.62 ms
llama_quantize:    total time = 533907.62 ms
=== [Step 4/4] Cleaning intermediate BF16 GGUF ===
=== All Done! Generated: /home/squall/models/Ling-3.0-flash-Fin-Q4_K_M.gguf ===
-rw-rw-r-- 1 squall squall 72G 10月 10 15:49 /home/squall/models/Ling-3.0-flash-Fin-Q4_K_M.gguf
```

+++

### Step 6: Start llama-server Inference Service

Launch `llama-server` to provide an OpenAI-compatible HTTP interface:

Key parameters:
- `-m ~/models/Ling-3.0-flash-Fin-Q4_K_M.gguf`: Model weight path
- `--alias ling-3.0-flash-fin-q4`: Registered model alias
- `-ngl all`: Offload all layers to GPU
- `-fa on`: Enable FlashAttention acceleration
- `-c 262144`: 256K full context window
- `-cb`: Enable continuous batching
- `-np 4`: Support 4 concurrent slots
- `--port 9102`: Default llama.cpp port convention

> [!WARNING]
> `llama-server` runs as a foreground process. Running the cell directly inside a notebook will block execution. Launch it in a separate terminal session before testing.

```{code-cell}
!./llama.cpp/build/bin/llama-server \
  -m ~/models/Ling-3.0-flash-Fin-Q4_K_M.gguf \
  --alias ling-3.0-flash-fin-q4 \
  -ngl all -fa on -c 262144 -cb -np 4 \
  --host 0.0.0.0 --port 9102 --api-key sk-ling-cookbook-test
```

Typical startup log:
```text
llama_server: HTTP server listening on 0.0.0.0:9102
llama_server: model loaded successfully, n_ctx = 262144, offload = 100% (GPU)
```

+++

### Step 7: Verify Service and Execute Inference Tests

After launching the service, you can query it via any OpenAI-compatible client. Ling-3.0-flash-Fin natively supports thinking mode, with recommended sampling parameters: `temperature=1.0`, `top_p=0.95`, `top_k=20`.

This section covers 3 verification steps:
1. **Health Check**: Confirm service availability and registered models.
2. **Financial Reasoning & Streaming Benchmark**: Perform Free Cash Flow to Firm (FCFF) discounted valuation, stream reasoning traces (`<think>`), and calculate TTFT and Decode TPS.
3. **Financial Tool Calling**: Test structured parameter extraction against a financial statement query schema.

+++

#### Step 7.1: Service Health Check

Query `GET /v1/models` to inspect active models:

```{code-cell}
import urllib.request
import json

url = "http://localhost:9102/v1/models"
headers = {"Authorization": "Bearer sk-ling-cookbook-test"}
try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        status_code = response.getcode()
        body = response.read().decode('utf-8')
        print(f"Health check status: {status_code}")
        print(f"Models response: {body}")
except Exception as e:
    print(f"Health check error: {e}")
```

Typical output:
```json
{
  "object": "list",
  "data": [
    {
      "id": "ling-3.0-flash-fin-q4",
      "aliases": [
        "ling-3.0-flash-fin-q4"
      ],
      "tags": [],
      "object": "model",
      "created": 1791619693,
      "owned_by": "llamacpp",
      "meta": {
        "vocab_type": 2,
        "n_vocab": 157184,
        "n_ctx": 65536,
        "n_ctx_train": 262144,
        "n_embd": 2560,
        "n_params": 127486405600,
        "size": 77003601792,
        "ftype": "Q4_K - Medium"
      }
    }
  ]
}
```

+++

#### Step 7.2: Financial Reasoning, Thinking Trace, and Speed Measurement

Send an FCFF DCF valuation prompt for a high-end semiconductor manufacturing firm, extract the `<think>` chain, and measure Time-to-First-Token (TTFT) and Decode Tokens Per Second (TPS) using exact server token usage metrics:

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9102/v1",
    api_key="sk-ling-cookbook-test"
)

def verify_financial_reasoning_and_speed():
    prompt = (
        "请对一家高端半导体制造企业进行自由现金流（FCFF）折现估值简要测算：\n"
        "1. 第1年EBIT 120亿元，年复合增长率15%，所得税率15%；\n"
        "2. WACC为9.5%，永续增长率为2.5%。\n"
        "请展示推导逻辑与主要估值结论。"
    )
    print("Sending financial prompt to model 'ling-3.0-flash-fin-q4'...")

    start_time = time.time()
    first_token_time = None
    chunk_count = 0
    exact_completion_tokens = None
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model="ling-3.0-flash-fin-q4",
            messages=[
                {"role": "system", "content": "你是由蚂蚁集团训练的金融专业级大模型 Ling-3.0-flash-Fin。在回答复杂专业问题时，请展示清晰、规范的金融推导逻辑。"},
                {"role": "user", "content": prompt}
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            temperature=1.0,
            top_p=0.95,
            stream=True,
            stream_options={"include_usage": True}
        )

        for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                exact_completion_tokens = chunk.usage.completion_tokens

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            reasoning_piece = (
                getattr(delta, 'reasoning', None)
                or getattr(delta, 'reasoning_content', None)
                or (delta.model_extra.get('reasoning') if hasattr(delta, 'model_extra') and delta.model_extra else None)
            )
            content_piece = delta.content

            if reasoning_piece:
                if first_token_time is None:
                    first_token_time = time.time()
                reasoning_text += reasoning_piece
                chunk_count += 1

            if content_piece:
                if first_token_time is None:
                    first_token_time = time.time()
                content_text += content_piece
                chunk_count += 1

        end_time = time.time()

        ttft = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
        decode_duration = end_time - first_token_time if first_token_time else 0.001

        total_tokens = exact_completion_tokens if exact_completion_tokens is not None else chunk_count
        tps = total_tokens / decode_duration

        print("\n=== Latency & Throughput Metrics ===")
        print(f"TTFT (Time to First Token): {ttft:.2f} ms")
        print(f"Decode TPS (Tokens/s): {tps:.2f} t/s")
        print(f"Total Generated Tokens: {total_tokens} (Exact tokens from usage)")
        print(f"Total Duration: {end_time - start_time:.2f} s")

        print("\n=== Extracted Reasoning Chain (<think>) ===")
        print(reasoning_text if reasoning_text else "[Note: Reasoning output]")

        print("\n=== Final Response Content ===")
        print(content_text)

    except Exception as e:
        print(f"Streaming verification failed: {e}")

if __name__ == "__main__":
    verify_financial_reasoning_and_speed()
```

Typical output:
```text
Sending financial prompt to model 'ling-3.0-flash-fin-q4'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 800.97 ms
Decode TPS (Tokens/s): 44.63 t/s
Total Generated Tokens: 1833
Total Duration: 41.87 s

--- Extracted Reasoning Chain (<think>) ---
1. 整理已知条件与财务要素：
   - 第1年 EBIT = 120 亿元，年复合增长率 = 15%，所得税率 = 15%
   - 加权平均资本成本 (WACC) = 9.5%，永续增长率 (g_n) = 2.5%
2. 确定估值模型框架：
   - 在未提供折旧、资本支出和营运资本变动细项的情况下，采用 EBIT 折现法，将 FCFF 近似为税后净营业利润 NOPAT = EBIT × (1 - T)。
   - 构建两阶段 FCFF 折现模型：明确预测期（10年高增长期，复合增长率 15%）+ 永续成熟期（永续增长率 2.5%）。
3. 验证合理性：
   - 半导体制造资本密集型行业，WACC 9.5% 处于合理区间（无风险利率约3%，市场风险溢价约6%，β约1.2-1.5）。
   - 永续增长率 2.5% 略高于长期 GDP 稳态增速。

--- Response Content ---
# 高端半导体制造企业 FCFF 折现估值测算

## 一、核心参数设定

| 参数 | 数值 | 依据 |
|------|------|------|
| 第1年 EBIT | 120 亿元 | 给定 |
| EBIT 复合增长率 (g) | 15% | 高端半导体国产替代+产能扩张 |
| 所得税率 (T) | 15% | 高新技术企业优惠税率 |
| WACC | 9.5% | 半导体制造资本密集型特征 |
| 永续增长率 (g_n) | 2.5% | 略高于长期GDP增速 |

---

## 二、推导逻辑与逐年预测（10年明确预测期）

### Step 1：计算第1年 FCFF (NOPAT)
$$FCFF_1 = EBIT_1 \times (1 - T) = 120 \times (1 - 0.15) = 102 \text{ 亿元}$$

### Step 2：逐年 FCFF 现值测算
| 年份 | EBIT (亿元) | FCFF = EBIT×0.85 (亿元) | 折现因子 (1.095)⁻ᵗ | 现值 (亿元) |
|------|------------|------------------------|---------------------|------------|
| 1 | 120.0 | 102.0 | 0.9132 | 93.15 |
| 2 | 138.0 | 117.3 | 0.8340 | 97.83 |
| 3 | 158.7 | 134.9 | 0.7617 | 102.75 |
| 4 | 182.5 | 155.1 | 0.6956 | 107.89 |
| 5 | 209.9 | 178.4 | 0.6353 | 113.35 |
| 6 | 241.4 | 205.2 | 0.5802 | 119.06 |
| 7 | 277.6 | 236.0 | 0.5298 | 125.03 |
| 8 | 319.2 | 271.3 | 0.4839 | 131.28 |
| 9 | 367.1 | 312.0 | 0.4419 | 137.87 |
| 10 | 422.2 | 358.9 | 0.4036 | 144.85 |

> **预测期现值合计 ≈ 1,173 亿元**

### Step 3：计算永续终值与企业价值
- 终值 $TV_{10} = \frac{FCFF_{10} \times (1 + g_n)}{WACC - g_n} = \frac{358.9 \times 1.025}{0.095 - 0.025} \approx 5,255.7 \text{ 亿元}$
- 终值折现现值 $PV(TV_{10}) = 5,255.7 \times 0.4036 \approx 2,121.2 \text{ 亿元}$
- **企业价值 (Enterprise Value, EV) = 预测期现值 + 终值现值 = 1,173 + 2,121 ≈ 3,294 亿元**
```

+++

#### Step 7.3: Financial Tool Calling Verification

Test structured financial metric extraction with a schema targeting financial statement lookups (`query_financial_statement`):

```{code-cell}
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9102/v1",
    api_key="sk-ling-cookbook-test"
)

financial_tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "query_financial_statement",
            "description": "查询上市公司指定报告期的资产负债表、利润表或现金流量表指标",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "上市公司股票代码或名称，例如 贵州茅台、600519、AAPL"
                    },
                    "report_period": {
                        "type": "string",
                        "description": "报告期，例如 2025Q3、2024FY"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "待查询财务科目列表，如 ['营业收入', '归母净利润']"
                    }
                },
                "required": ["ticker", "report_period", "metrics"]
            }
        }
    }
]

def verify_financial_tool_calling():
    print(f"Testing Financial Tool Calling with model 'ling-3.0-flash-fin-q4'...")
    try:
        response = client.chat.completions.create(
            model="ling-3.0-flash-fin-q4",
            messages=[
                {"role": "user", "content": "帮我查询贵州茅台2025年第三季度的营业收入和归母净利润。"}
            ],
            tools=financial_tools_schema,
            tool_choice="auto",
            temperature=0.1
        )

        message = response.choices[0].message
        if message.tool_calls:
            print("\n=== Financial Tool Call Output Detected ===")
            for tool_call in message.tool_calls:
                print(f"Tool Call ID: {tool_call.id}")
                print(f"Function Name: {tool_call.function.name}")
                print(f"Arguments JSON: {tool_call.function.arguments}")
        else:
            print("\n=== Direct Content Output ===")
            print(message.content)

    except Exception as e:
        print(f"Tool calling verification failed: {e}")

if __name__ == "__main__":
    verify_financial_tool_calling()
```

Typical output:
```text
Testing Financial Tool Calling with model 'ling-3.0-flash-fin-q4'...

=== Financial Tool Call Output Detected ===
Tool Call ID: 1DqqNndH1re5aq8whZG1e2jKo7Pjkk9H
Function Name: query_financial_statement
Arguments JSON: {"ticker":"贵州茅台","report_period":"2025Q3","metrics":["营业收入", "归母净利润"]}
```

+++

### Step 8: Troubleshooting & FAQ

1. **Conversion Dependencies Missing**:
   - Symptom: `ModuleNotFoundError` when running `convert_hf_to_gguf.py`.
   - Fix: Ensure `uv pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt` has been run inside `.venv`.

2. **KV Cache Out of Memory Under High Concurrency**:
   - Symptom: OOM error when allocating large context (`-c 262144`) with high slots (`-np 4`).
   - Fix: Reduce slots (`-np 2`) or adjust context length (`-c 131072`). A single DGX Spark node provides 121GB unified memory, leaving ~72GB after loading Q4_K_M weights.

3. **Port Conflict (Port 9102 occupied)**:
   - Symptom: `llama-server` errors out due to address already in use.
   - Fix: Identify and terminate the occupying process via `lsof -i :9102`, or bind to another port using `--port`.
