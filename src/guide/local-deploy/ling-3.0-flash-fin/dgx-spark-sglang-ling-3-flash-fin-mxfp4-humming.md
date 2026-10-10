---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

##### Copyright 2026 Ant Group and NVIDIA Corporation.

+++

# Ling-3.0-flash-Fin on DGX Spark (SGLang MXFP4 Humming Optimization) Deployment Guide

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

Special thanks to Nvidia team [@ly01325](https://github.com/ly01325) for performance optimization and operator support.

+++

`Ling-3.0-flash-Fin` is Ant Group's premier financial-enhanced large language model, built upon `Ling-3.0-flash` (124B total parameters, 5.1B activated per token, hybrid attention MoE architecture) with continued pre-training and alignment on extensive financial corpora and analyst research reports.

This notebook demonstrates deploying the MXFP4 quantized `Ling-3.0-flash-Fin` on a single **NVIDIA DGX Spark** (Grace Blackwell GB10, 121GB usable unified memory) leveraging three deep optimization mechanisms:

1. **Humming MoE Operator Backend**: Tailored specifically for Blackwell (SM121) hardware architecture (`--moe-runner-backend humming`);
2. **Online FP8 LM Head**: In-flight dynamic quantization of the BF16 LM Head into FP8 (`SGLANG_ENABLE_FP8_LM_HEAD=1`), reducing decoding memory bandwidth pressure;
3. **MTP Speculative Decoding**: Leveraging the native Multi-Token Prediction architecture with 3-step speculative decoding (`NEXTN`).

> [!TIP]
> **Recommended Setup**:
> - Python 3.11, CUDA 13.0, Ubuntu 24.04 ARM64 / DGX OS;
> - Use **uv** for clean virtual environment management;
> - Static memory fraction `--mem-fraction-static 0.68` to balance MTP draft cache and long financial context retention.

+++

### Step 1: Prepare Python Virtual Environment and Clone Optimized Branch

Clone the SGLang branch incorporating Humming MoE, online FP8 LM Head, and MTP draft patches (`inclusionAI/sglang` branch `ling_v3_support_mxfp4_humming`):

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
!git clone -b ling_v3_support_mxfp4_humming https://github.com/inclusionAI/sglang.git
```

Typical output:
```text
Using CPython 3.11.15
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
Resolved 16 packages in 709ms
Installed 16 packages in 21ms
Cloning into 'sglang'...
remote: Total 225775 (delta 182), reused 225600 (delta 180)
Receiving objects: 100% (225775/225775), 184.20 MiB | 28.50 MiB/s, done.
```

+++

### Step 2: Install SGLang and Dependencies

Build SGLang in editable mode, omitting optional Rust gRPC dependencies on ARM64:

```{code-cell}
!sed -i.bak '/\[\[tool.setuptools-rust.ext-modules\]\]/,+3d' ./sglang/python/pyproject.toml

!source .venv/bin/activate && \
J=$(python3 -c "import os; print(max(1, min(os.cpu_count() or 1, 8)))") && \
export MAX_JOBS="$J" NINJA_NUM_JOBS="$J" CMAKE_BUILD_PARALLEL_LEVEL="$J" && \
echo "Building with MAX_JOBS=$J..." && \
uv pip install -e "./sglang/python[all]"
```

Typical output:
```text
Building with MAX_JOBS=8...
Resolved 240 packages in 6.88s
      Built sglang @ file:///home/squall/sipan/sglang/python
Prepared 112 packages in 39.15s
Installed 225 packages in 174ms
```

+++

### Step 3: Download Model Weights

Download the pre-quantized MXFP4 weights (~60.5GB) from ModelScope or Hugging Face:

- [ModelScope Ling-3.0-flash-Fin-fp4](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-Fin-fp4)
- [Hugging Face Ling-3.0-flash-Fin-fp4](https://huggingface.co/inclusionAI/Ling-3.0-flash-Fin-fp4)

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-Fin-fp4 --local-dir ~/models/Ling-3.0-flash-Fin-fp4
```

+++

### Step 4: Verify or Pre-compile FlashInfer CUTLASS MXFP4 Kernels

Pre-compiling CUTLASS MoE kernels prevents memory spikes during runtime model loading:

```{code-cell}
import os
import sys
from pathlib import Path

ROOTS = [
    Path.home() / ".cache" / "flashinfer",
    Path.home() / ".cache" / "sglang" / ".cache" / "flashinfer",
]
PATTERN = "*/121a/cached_ops/fused_moe_120"

def newest(name):
    hits = [p for r in ROOTS for p in r.glob(f"{PATTERN}/{name}")]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None

so = newest("fused_moe_120.so")
if so:
    print(f"✅ FlashInfer CUTLASS kernel ready: {so}")
else:
    print("⏳ Pre-compiling kernels required before launch...")
```

+++

### Step 5: Launch Inference Server with Humming Optimizations

Launch the server with Humming MoE backend, online FP8 LM Head, and MTP 3-step speculative decoding:

```{code-cell}
!source .venv/bin/activate && \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_DSV4_FP4_DEQUANT=0 \
SGLANG_FP8_IGNORED_LAYERS="" \
SGLANG_ENABLE_FP8_LM_HEAD=1 \
HUMMING_COMPILER=nvrtc \
HUMMING_CACHE_DIR=~/.humming/cache \
python3 -m sglang.launch_server \
  --model-path ~/models/Ling-3.0-flash-Fin-fp4 \
  --served-model-name ling-3.0-flash-fin-fp4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 1 \
  --ep-size 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key sk-ling-cookbook-test \
  --max-running-requests 1 \
  --max-mamba-cache-size 64 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 \
  --page-size 64 \
  --context-length 262144 \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-bs-decode 1 \
  --cuda-graph-backend-prefill disabled \
  --random-seed 308534008 \
  --reasoning-parser ling3 \
  --tool-call-parser glm \
  --attention-backend flashinfer \
  --disable-flashinfer-autotune \
  --mem-fraction-static 0.65 \
  --fp8-gemm-backend cutlass \
  --moe-runner-backend humming \
  --flashinfer-mxfp4-moe-precision default \
  --disable-shared-experts-fusion \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path ~/models/Ling-3.0-flash-Fin-fp4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --json-model-override-args '{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}'
```

Server ready log:
```text
[2026-10-10 16:55:10] Online FP8 quantization enabled for lm_head.
[2026-10-10 16:56:02] Load weight end. elapsed=52.11 s, type=BailingMoeV3ForCausalLM
[2026-10-10 16:56:45] Capture draft decode CUDA graph begin...
[2026-10-10 16:57:30] The server is fired up and ready to roll!
```

+++

### Step 6: Functional Verification and Financial Benchmark

#### Step 6.1: Health Check

```{code-cell}
import urllib.request
import json

url = "http://127.0.0.1:30000/v1/models"
headers = {"Authorization": "Bearer sk-ling-cookbook-test"}

try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=5) as response:
        body = response.read().decode('utf-8')
        print("✅ Health check passed!")
        print(f"Models response: {body}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

+++

#### Step 6.2: Financial FCFF Reasoning and Speculative Benchmark

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="sk-ling-cookbook-test"
)

prompt = (
    "Please perform a Free Cash Flow to Firm (FCFF) discounted valuation model for an advanced semiconductor manufacturer:\n"
    "1. Year 1 EBIT: 12.0 billion RMB, compound growth rate: 15%, tax rate: 15%;\n"
    "2. WACC is 9.5%, perpetual growth rate is 2.5%.\n"
    "Please show your mathematical derivation and key valuation figures."
)

start_time = time.time()
first_token_time = None
chunk_count = 0
exact_completion_tokens = None
reasoning_text = ""
content_text = ""

try:
    response = client.chat.completions.create(
        model="ling-3.0-flash-fin-fp4",
        messages=[
            {"role": "system", "content": "You are Ling-3.0-flash-Fin, an expert financial model. Provide rigorous analysis."},
            {"role": "user", "content": prompt}
        ],
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
        now = time.time()

        r_piece = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        c_piece = delta.content

        if r_piece:
            if first_token_time is None:
                first_token_time = now
            reasoning_text += r_piece
            chunk_count += 1

        if c_piece:
            if first_token_time is None:
                first_token_time = now
            content_text += c_piece
            chunk_count += 1

    end_time = time.time()
    ttft = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
    decode_duration = end_time - first_token_time if first_token_time else 0.001
    total_tokens = exact_completion_tokens if exact_completion_tokens is not None else chunk_count
    tps = total_tokens / decode_duration

    print("================ Performance Metrics ================")
    print(f"  - TTFT (Time to First Token): {ttft:.2f} ms")
    print(f"  - Decode TPS (Tokens/s): {tps:.2f} tokens/s")
    print(f"  - Total Generated Tokens: {total_tokens}")
    print(f"  - Total Duration: {end_time - start_time:.2f} s")
    print("=====================================================")
    
    print("\n--- Reasoning Chain (<think>) ---")
    print(reasoning_text if reasoning_text else "[Reasoning in body]")
    
    print("\n--- Response Content ---")
    print(content_text)

except Exception as e:
    print(f"❌ Evaluation error: {e}")
```

Typical benchmark output on DGX Spark:
```text
================ Performance Metrics ================
  - TTFT (Time to First Token): 163.20 ms (steady state) / 36.48 s (cold graph capture)
  - Decode TPS (Tokens/s): 51.75 tokens/s (peak batch throughput up to 63.36 tokens/s)
  - Total Generated Tokens: 16,653 tokens
  - Speculative MTP Stats: MTP=3, average accept length 2.65 ~ 3.17, accept rate 53% ~ 72%
  - Total Duration: 321.81 s (generating complete 16k+ in-depth financial model)
=====================================================

--- Reasoning Chain (<think>) ---
## 1. Problem Formulation & Task Analysis
The user requests a Free Cash Flow to Firm (FCFF) discounted valuation model for an advanced semiconductor manufacturer...
**Given Parameters:**
- Year 1 EBIT = 12.0 billion RMB
- CAGR (g₁) = 15%
- Tax rate (T) = 15%
- Year 1 D&A = 3.5 billion RMB, CapEx = 4.5 billion RMB, ΔNWC = 1.0 billion RMB
- WACC = 9.5%, perpetual growth rate = 2.5%
... [Detailed 10-year explicit projection and valuation derivation]

--- Response Content ---
# FCFF Valuation Modeling Report

## 1. Core Model & Equations
FCFF_t = EBIT_t * (1 - T) + D&A_t - CapEx_t - ΔNWC_t
Enterprise Value (EV) = ∑ FCFF_t / (1+WACC)^t + Terminal Value / (1+WACC)^N
...
```

Server-side live speculative decoding metrics:
```text
[2026-10-10 17:18:22] Decode batch, #running-req: 1, #full token: 16320, full token usage: 0.63, accept len: 3.05, accept rate: 0.68, gen throughput (token/s): 58.50
[2026-10-10 17:18:26] Decode batch, #running-req: 1, #full token: 16512, full token usage: 0.63, accept len: 3.00, accept rate: 0.67, gen throughput (token/s): 57.85
[2026-10-10 17:18:34] Decode batch, #running-req: 1, #full token: 512, full token usage: 0.02, accept len: 3.17, accept rate: 0.72, gen throughput (token/s): 63.36
```

+++

#### Step 6.3: Financial Tool Calling

```{code-cell}
financial_tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "query_financial_statement",
            "description": "Query balance sheet, income statement, or cash flow metrics for listed companies",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker or company name, e.g. Kweichow Moutai, 600519, AAPL"
                    },
                    "report_period": {
                        "type": "string",
                        "description": "Reporting period, e.g. 2025Q3, 2024FY"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Target financial metrics list"
                    }
                },
                "required": ["ticker", "report_period", "metrics"]
            }
        }
    }
]

def verify_financial_tool_calling():
    print("=== Testing Financial Tool Calling ===")
    response = client.chat.completions.create(
        model="ling-3.0-flash-fin-fp4",
        messages=[
            {"role": "user", "content": "Please query revenue and net profit of Kweichow Moutai for 2025Q3."}
        ],
        tools=financial_tools_schema,
        tool_choice="auto",
        temperature=0.1
    )
    message = response.choices[0].message
    if message.tool_calls:
        print("✅ Tool call triggered successfully:")
        for tc in message.tool_calls:
            print(f"  - ID: {tc.id}")
            print(f"  - Func: {tc.function.name}")
            print(f"  - Args: {tc.function.arguments}")
    else:
        print("No tool call, direct response:", message.content)

if __name__ == "__main__":
    verify_financial_tool_calling()
```

Typical execution output:
```text
=== Testing Financial Tool Calling ===
✅ Tool call triggered successfully:
  - ID: call_6f9b1d45cde040ebadb2fd72
  - Func: query_financial_statement
  - Args: {"ticker": "贵州茅台", "report_period": "2025Q3", "metrics": ["营业收入", "扣非归母净利润"]}
```

+++

### Step 7: Troubleshooting

1. **Static Memory Fraction**: Keep `--mem-fraction-static 0.65` (~79 GB) to reserve adequate unified memory for CUDA graphs, MTP draft tokens, and long-context KV cache.
2. **Pre-compilation**: Ensure `fused_moe_120.so` is pre-compiled using Step 4 before running the server to prevent dynamic compilation overhead.
3. **Speculative Decoding Logs**: Verify `Capture draft decode CUDA graph begin...` and `Capture draft decode CUDA graph end` appear in the log to confirm speculative execution.
4. **Tool Call Parser**: Use `--tool-call-parser glm` to align with SGLang tool-calling specifications.

+++

### Step 8: Benchmark Performance Comparison

Benchmark results measured on an NVIDIA DGX Spark (Grace Blackwell GB10, 121 GB unified memory):

| Deployment Setup | MoE Backend | Speculative Decoding | Steady Decode TPS | Speedup | Steady TTFT | Static Memory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Marlin FP4 Baseline** | Marlin MoE | Disabled (Autoregressive) | 34.56 tokens/s | 1.00x (Baseline) | ~210 ms | ~74 GB |
| **SGLang MXFP4 Humming (Ours)** | **Humming MoE** | **Enabled (NEXTN MTP 3 steps)** | **51.75 tokens/s** (Peak 63.4 t/s) | **1.50x (+49.7%)** | **163.20 ms** | **~79 GB** |

**Performance Summary**:
- **Decode Speed**: With Humming MoE kernels and 3-step MTP (average accept length 2.65 ~ 3.17 tokens), decode speed increases from 34.56 t/s to **51.75 tokens/s** (1.50x speedup);
- **LM Head Memory Bandwidth**: `SGLANG_ENABLE_FP8_LM_HEAD=1` reduces memory bandwidth usage for the 150k vocabulary projection layer;
- **Reasoning and Function Calling**: Supports generating 16k+ tokens of financial reasoning chains (`<think>`) and extracting parameters for financial function calls under 50+ tokens/s throughput.

