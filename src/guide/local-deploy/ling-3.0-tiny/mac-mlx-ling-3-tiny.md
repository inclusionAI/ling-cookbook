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

##### Copyright 2026 Ant Group.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

+++

# Ling-3.0-tiny on Apple Silicon Mac (MLX-LM) Deployment Guide

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

`Ling-3.0-tiny` is a 7.9B lightweight Sparse MoE language model in the Ling model family with 1.3B active parameters per token, natively supporting a 128K context window.

`MLX` is Apple's open-source machine learning framework engineered specifically for Apple Silicon's unified memory architecture. It features minimal abstraction overhead and direct execution of native Metal kernels. Support for the Ling-3.0 architecture (`bailing_hybrid` / `BailingMoeV3ForCausalLM`) has been officially merged into upstream `mlx-lm` (PR #1711). With `mlx-lm`, developers can load native Safetensors weights directly on a Mac, perform on-device 4-bit / 8-bit / MXFP quantization, and serve an OpenAI-compatible high-performance HTTP REST server (`mlx_lm.server`).

This guide demonstrates how to deploy `Ling-3.0-tiny` across multiple precisions (BF16 full precision, 8-bit / MXFP8 high-fidelity quantization, and 4-bit compact quantization) using `mlx-lm` on Apple Silicon Macs (M1 / M2 / M3 / M4 / M5 series).

---

### Hardware & Precision Matrix

Select the suitable precision profile based on your Mac's unified memory capacity:

| Deployment Profile | Weight Format | Weights Disk Size | Recommended RAM (8K Context) | Recommended Mac Hardware | Typical Decode TPS (Empirical) |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **BF16 (Full Precision)** | Native Safetensors | **~14.72 GB** | **≥ 24 GB - 32 GB** | MacBook Pro 24GB / 36GB / 48GB+ | **~88.3 tok/s (Empirical)** |
| **8-bit / MXFP8 (High Fidelity)** | MLX Quantized Weights | **~8.06 - 8.27 GB** | **≥ 16 GB - 18 GB** | MacBook Pro 16GB / 18GB / 24GB | **~118.4 - 119.9 tok/s (Empirical)** |
| **4-bit (Lightweight Recommended)** | MLX Quantized Weights | **~4.83 GB** | **≥ 8 GB - 12 GB** | MacBook Air / Mac mini 8GB/16GB | **~150.5 tok/s (Empirical)** |

> [!TIP]
> **Environment & Dependency Requirements**:
> - **Python 3.11 / 3.12** is recommended;
> - **Upstream Architecture Requirement**: Ling-3.0 (`bailing_hybrid`) support was merged in `mlx-lm` upstream PR #1711. Install the latest development version (`git+https://github.com/ml-explore/mlx-lm.git` or `mlx-lm>=0.32.0`);
> - We recommend using **uv** for fast Python virtual environment and dependency management;
> - Default port convention: `mlx_lm.server` listens on port `8080` by default.

+++

### Step 1: Set Up Python Virtual Environment and Install MLX-LM

#### Step 1.1: Create Virtual Environment and Install MLX-LM

Use `uv` to create an isolated Python virtual environment and install `mlx-lm` with native Ling-3.0 architecture support:

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!mkdir -p ~/.cache/huggingface/hub
!source .venv/bin/activate && uv pip install --upgrade mlx "mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git" 'openai>=1.52.0,<2.0.0' 'modelscope>=1.18.0'
```

Typical installation output:
```text
Resolved 34 packages in 1.58s
Installed 24 packages in 95ms
 + mlx==0.32.2
 + mlx-lm==0.32.0 (from git+https://github.com/ml-explore/mlx-lm.git)
 + modelscope==1.39.1
 + openai==1.109.1
```

+++

### Step 2: Download Official Ling-3.0-tiny Base Weights (Safetensors)

Download the official `inclusionAI/Ling-3.0-tiny` Safetensors base weights (32 shards, ~15.8 GB total) from ModelScope or Hugging Face:
- [Ling-3.0-tiny on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny)
- [Ling-3.0-tiny on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-tiny)

We recommend using the `modelscope` CLI to download weights to `~/models/Ling-3.0-tiny`:

> [!NOTE]
> This step downloads approximately 15.8 GB of model weights. Download time depends on your network bandwidth.

```{code-cell}
!mkdir -p ~/models/Ling-3.0-tiny
!source .venv/bin/activate && uv run modelscope download \
  --model inclusionAI/Ling-3.0-tiny \
  --local_dir ~/models/Ling-3.0-tiny
```

Typical download output:
```text
Downloading shards: 100%|██████████| 32/32 [05:20<00:00, 51.2MB/s]
✓ Successfully downloaded Ling-3.0-tiny to /Users/sipan/models/Ling-3.0-tiny
```

+++

### Step 3: Prepare Multi-Precision Profiles (BF16 / 8-bit / 4-bit)

`mlx-lm` provides native weight conversion and quantization tools, allowing you to prepare the precision tier suited to your local hardware:

#### Option A: BF16 Native Full Precision

The downloaded Safetensors base weights are in native BF16 format. MLX can load them out-of-the-box **without any conversion**. Directly target the local path `~/models/Ling-3.0-tiny`.

#### Option B: 8-bit / MXFP8 High-Fidelity Quantization

Use `mlx_lm.convert` to quantize the model into 8-bit or MXFP8 (Microscaling FP8), reducing disk footprint to ~7.9 GB:

```{code-cell}
!source .venv/bin/activate && python3 -m mlx_lm.convert \
  --hf-path ~/models/Ling-3.0-tiny \
  --mlx-path ~/models/Ling-3.0-tiny-mxfp8 \
  --quantize \
  --q-mode mxfp8 \
  --trust-remote-code
```

Or convert to standard 8-bit affine quantization:
```bash
python3 -m mlx_lm.convert \
  --hf-path ~/models/Ling-3.0-tiny \
  --mlx-path ~/models/Ling-3.0-tiny-8bit \
  --quantize \
  --q-bits 8 \
  --trust-remote-code
```

#### Option C: 4-bit Compact Quantization (Recommended)

Use `mlx_lm.convert` to generate 4-bit quantized weights (~4.8 GB), optimized for 8GB / 16GB consumer-tier Macs:

```{code-cell}
!source .venv/bin/activate && python3 -m mlx_lm.convert \
  --hf-path ~/models/Ling-3.0-tiny \
  --mlx-path ~/models/Ling-3.0-tiny-4bit \
  --quantize \
  --q-bits 4 \
  --trust-remote-code
```

+++

### Step 4: Quick Generation Verification with MLX-LM CLI

`mlx-lm` includes a convenient CLI tool `mlx_lm.generate` to quickly verify model loading, generation quality, and decoding speed directly from the command line. Run the command matching your prepared precision:

#### Option A: Verify BF16 Native
```{code-cell}
!source .venv/bin/activate && python3 -m mlx_lm.generate \
  --model ~/models/Ling-3.0-tiny \
  --prompt "Calculate 17 × 23 step-by-step with clear derivation." \
  --max-tokens 256 \
  --temp 0.6
```

#### Option B: Verify 8-bit / MXFP8
```{code-cell}
!source .venv/bin/activate && python3 -m mlx_lm.generate \
  --model ~/models/Ling-3.0-tiny-mxfp8 \
  --prompt "Calculate 17 × 23 step-by-step with clear derivation." \
  --max-tokens 256 \
  --temp 0.6
```

#### Option C: Verify 4-bit Recommended
```{code-cell}
!source .venv/bin/activate && python3 -m mlx_lm.generate \
  --model ~/models/Ling-3.0-tiny-4bit \
  --prompt "Calculate 17 × 23 step-by-step with clear derivation." \
  --max-tokens 256 \
  --temp 0.6
```

Typical CLI output (Apple Silicon Mac steady-state empirical data):
```text
Loading model from /Users/sipan/models/Ling-3.0-tiny...
Prompt: Calculate 17 × 23 step-by-step with clear derivation.
------
To calculate 17 × 23, we can use the distributive property:
1. Decompose 23 into 20 + 3:
   17 × 23 = 17 × (20 + 3)
2. Expand using distributive multiplication:
   = 17 × 20 + 17 × 3
   = 340 + 51
3. Sum the partial products:
   = 391
Therefore, 17 × 23 = 391.
------
Prompt: 17 tokens, 255.22 tokens-per-sec
Generation: 256 tokens, 86.05 tokens-per-sec
Peak memory: 16.39 GB
```

+++

### Step 5: Launch MLX-LM HTTP Inference Server (OpenAI-Compatible)

Launch `mlx_lm.server` to expose an OpenAI-compatible HTTP REST endpoint locally. Key CLI arguments:
- `--model <path>`: Local model weight directory (e.g., `~/models/Ling-3.0-tiny`)
- `--host 127.0.0.1 --port 8080`: Binding host and port (`8080` standard)
- `--trust-remote-code`: Trust custom model code
- `--chat-template-args '{"enable_thinking": true}'`: Configure chat template behavior (preserve reasoning chain)

⚠️ Notice:

`mlx_lm.server` runs as a foreground service. If executed directly inside a Notebook cell, it will block kernel execution. **Run the server launch command in an independent terminal window**; once initialized, verify endpoints and execute inferences within the Notebook. Select the command for your preferred precision profile:

#### Option A: Launch BF16 Full-Precision Server
```bash
python3 -m mlx_lm.server \
  --model ~/models/Ling-3.0-tiny \
  --host 127.0.0.1 \
  --port 8080 \
  --trust-remote-code
```

#### Option B: Launch 8-bit / MXFP8 High-Fidelity Server
```bash
python3 -m mlx_lm.server \
  --model ~/models/Ling-3.0-tiny-mxfp8 \
  --host 127.0.0.1 \
  --port 8080 \
  --trust-remote-code
```

#### Option C: Launch 4-bit Lightweight Server
```bash
python3 -m mlx_lm.server \
  --model ~/models/Ling-3.0-tiny-4bit \
  --host 127.0.0.1 \
  --port 8080 \
  --trust-remote-code
```

#### Option D: Launch Single-User Peak Performance Server (4-bit + Prefix Caching)

For personal local workflows (terminal, IDE copilot, or single-user multi-turn Agent tasks), enabling **Prompt Prefix Caching** retains conversation history and system instructions in unified memory, reducing subsequent multi-turn Time to First Token (TTFT) to milliseconds:

```bash
python3 -m mlx_lm.server \
  --model ~/models/Ling-3.0-tiny-4bit \
  --host 127.0.0.1 \
  --port 8080 \
  --prompt-cache-size 10 \
  --prompt-cache-bytes 2000000000 \
  --prefill-step-size 2048 \
  --trust-remote-code
```

Typical server startup logs:
```text
Loading model from /Users/sipan/models/Ling-3.0-tiny...
Model loaded successfully.
INFO:     Started server process [65892]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
```

+++

### Step 6: Verify Deployment and Test Model Inference

Once the server is running, use standard OpenAI-compatible client libraries to perform end-to-end testing:

1. **Health Check** - Verify model availability and HTTP connectivity;
2. **Streaming Reasoning & Speed Benchmarking** - Validate `<think>` reasoning extraction, TTFT, and decode TPS;
3. **Function Calling Tool Use** - Test structured tool invocation;
4. **Empirical Benchmark & Memory Matrix Across Precisions**.

+++

#### Step 6.1: Connectivity and Health Check

Query `GET /v1/models` to verify active model status:

```{code-cell}
import urllib.request
import json

url = "http://127.0.0.1:8080/v1/models"
try:
    req = urllib.request.Request(url)
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
Health check status: 200
Models response: {"object": "list", "data": [{"id": "/Users/sipan/models/Ling-3.0-tiny-4bit", "object": "model", "created": 1788849582}]}
```

+++

#### Step 6.2: Streaming Inference, Reasoning Extraction, and Performance Metrics

Send a streaming completion request using the OpenAI Python SDK. Note: `mlx_lm.server` parses reasoning content into the `delta.reasoning` field:

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="EMPTY"
)

# Retrieve active model ID dynamically
model_id = client.models.list().data[0].id

def verify_streaming_and_thinking():
    prompt = "Calculate 17 × 23 step-by-step with clear derivation."
    print(f"Sending prompt: '{prompt}' to MLX model '{model_id}'...")

    start_time = time.time()
    first_token_time = None
    total_tokens = 0
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=2048,
            stream=True
        )

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            now = time.time()
            if first_token_time is None:
                first_token_time = now

            reasoning_chunk = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
            if reasoning_chunk:
                reasoning_text += reasoning_chunk
                total_tokens += 1
            elif delta.content:
                content_text += delta.content
                total_tokens += 1

        end_time = time.time()
        ttft = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
        decode_duration = end_time - first_token_time if first_token_time else 0.001
        tps = total_tokens / decode_duration

        print("\n=== Latency & Throughput Metrics ===")
        print(f"TTFT (Time to First Token): {ttft:.2f} ms")
        print(f"Decode TPS (Tokens/s): {tps:.2f} tok/s")
        print(f"Total Generated Tokens: {total_tokens}")
        print(f"Total Duration: {end_time - start_time:.2f} s")

        print("\n=== Extracted Reasoning Chain (<think>) ===")
        print(reasoning_text if reasoning_text else "[Note: Reasoning merged in main content or disabled]")

        print("\n=== Final Response Content ===")
        print(content_text)

    except Exception as e:
        print(f"Streaming verification failed: {e}")

if __name__ == "__main__":
    verify_streaming_and_thinking()
```

Typical test output (empirical Apple Silicon Mac benchmark):
```text
Sending prompt: 'Calculate 17 × 23 step-by-step with clear derivation.' to MLX model '/Users/sipan/models/Ling-3.0-tiny-4bit'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 172.94 ms
Decode TPS (Tokens/s): 105.32 tok/s
Total Generated Tokens: 449
Total Duration: 4.44 s

=== Extracted Reasoning Chain (<think>) ===
1. Analyze problem: Calculate 17 × 23 step-by-step.
2. Select method: Distributive property of multiplication.
   Step 1: Decompose 23 into 20 and 3: 17 × (20 + 3)
   Step 2: Multiply each term: 17 × 20 = 340, 17 × 3 = 51
   Step 3: Add results: 340 + 51 = 391

=== Final Response Content ===
To calculate 17 × 23, we can break it down using the distributive property:
1. Step 1: Decompose 23 into 20 + 3
2. Step 2: Multiply both parts by 17:
   - 17 × 20 = 340
   - 17 × 3 = 51
3. Step 3: Add the partial products together:
   340 + 51 = 391
Final answer: 17 × 23 = 391.
```

+++

#### Step 6.3: Function Calling / Tool Use Verification

Provide a structured tool definition and verify native Function Calling support in `mlx_lm.server`:

```{code-cell}
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="EMPTY"
)

model_id = client.models.list().data[0].id

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert currency amounts between different international currencies based on exchange rates",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {
                        "type": "string",
                        "description": "Source currency code, e.g., USD, EUR, CNY"
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "Target currency code, e.g., CNY, JPY, USD"
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount to convert"
                    }
                },
                "required": ["from_currency", "to_currency", "amount"]
            }
        }
    }
]

def verify_tool_calling():
    print(f"Testing Function Calling with MLX-LM model '{model_id}'...")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": "Please convert 100 USD to CNY."}
            ],
            tools=tools_schema,
            tool_choice="auto"
        )

        message = response.choices[0].message
        if message.tool_calls:
            print("\n=== Function Call Output Detected ===")
            for tool_call in message.tool_calls:
                print(f"Tool Call ID: {tool_call.id}")
                print(f"Function Name: {tool_call.function.name}")
                print(f"Arguments JSON: {tool_call.function.arguments}")
            if hasattr(message, "reasoning") and message.reasoning:
                print(f"\nTool Selection Reasoning: {message.reasoning}")
        else:
            print("\n=== Direct Response ===")
            print(message.content)

    except Exception as e:
        print(f"Tool calling verification failed: {e}")

if __name__ == "__main__":
    verify_tool_calling()
```

Typical test output:
```text
Testing Function Calling with MLX-LM model '/Users/sipan/models/Ling-3.0-tiny-4bit'...

=== Function Call Output Detected ===
Tool Call ID: 16f58426-f0cb-4d90-b34d-7d50f82678b9
Function Name: convert_currency
Arguments JSON: {"from_currency": "USD", "to_currency": "CNY", "amount": 100}

Tool Selection Reasoning: The user wants to convert 100 USD to CNY. I need to call the convert_currency tool with arguments: from_currency="USD", to_currency="CNY", amount=100.
```

+++

#### Step 6.4: Empirical Decode Throughput and Memory Benchmark across Precisions

The following benchmark presents end-to-end inference and resident memory measurements conducted on Apple Silicon Mac (M5 Pro, 48GB unified memory, macOS 15) for `Ling-3.0-tiny` across all precision tiers, utilizing warm-up (Metal shader JIT compilation) followed by 3 rounds of steady-state sampling:

| Profile / Precision | Weight Format / Type | Disk Size | Cold Start Prefill | Steady Prefill (Prompt TPS) | Steady Decode (Decode TPS) | Steady TTFT | Peak Resident RAM (Peak RAM) | Recommended Mac Hardware |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Ling-3.0-tiny` (BF16)** | Safetensors (Full Precision) | **14.72 GB** | 300.98 tok/s | **421.58 tok/s** | **88.33 tok/s** (88.1~88.7) | **87.93 ms** | **15.85 GB** | MacBook Pro 24GB / 36GB / 48GB+ |
| **`Ling-3.0-tiny-8bit`** | MLX 8-bit (High Fidelity) | **8.27 GB** | 169.35 tok/s | **538.28 tok/s** | **119.88 tok/s** (119.1~120.4) | **68.88 ms** | **8.98 GB** | MacBook Pro 16GB / 18GB / 24GB |
| **`Ling-3.0-tiny-mxfp8`** | MLX MXFP8 (Microscaling FP8) | **8.06 GB** | 163.05 tok/s | **505.23 tok/s** | **118.35 tok/s** (116.7~119.8) | **73.47 ms** | **8.75 GB** | MacBook Pro 16GB / 18GB / 24GB |
| **`Ling-3.0-tiny-4bit`** | MLX 4-bit (Recommended) | **4.83 GB** | 438.48 tok/s | **683.18 tok/s** | **150.51 tok/s** (149.4~151.5) | **54.16 ms** | **5.30 GB** | MacBook Air / Mac mini 8GB / 16GB |

> [!TIP]
> **MLX vs llama.cpp Comparison & Selection Guidelines**:
> - **Throughput Comparison**: In steady-state CLI generation, MLX 4-bit achieved a remarkable **150.51 tok/s**, outperforming llama.cpp's 123.69 tok/s on identical hardware;
> - **Developer Ecosystem**: MLX runs purely on Python + native Metal drivers, allowing native interoperability with Python tooling for LoRA fine-tuning, embedded data pipelines, and rapid experimentation;
> - **Framework Recommendation**: Choose **MLX-LM** if your workflow benefits from Python-native integration, custom operator exploration, or local fine-tuning. Choose **llama.cpp** if your primary goal is lightweight, cross-platform binary deployment with C++ bindings.

+++

#### Step 6.5: Single-User Throughput & Low-Latency Tuning Guide

For personal Mac workstations used in daily coding, terminal copilot interactions, or single-user multi-turn Agent loops, combine the following 4 tuning strategies for maximum responsiveness and efficiency:

1. **4-bit Weight Quantization (Primary Speed Priority)**:
   - **Mechanism**: LLM autoregressive token decoding is strictly memory-bandwidth bound. Migrating from BF16 to 4-bit slashes the volume of weight data moved across unified memory bus per token step by **~70%**;
   - **Impact**: Generative decode speed surges from **86.1 tok/s** to **147.9 tok/s** (an increase of over **+70%**), while resident RAM drops to only **5.2 GB**, leaving ample headroom for your active developer toolchain.

2. **Prompt Prefix Caching (Eliminate Multi-Turn Latency)**:
   - **Mechanism**: In multi-turn chat or workflows with static system prompts (e.g., extensive instructions, code schemas, or domain knowledge), the attention states for identical prefixes are reused. `mlx_lm.server`'s `--prompt-cache-size 10` caches these states via LRU in unified memory, bypassing redundant prefill computation;
   - **Impact**: Subsequent multi-turn Time to First Token (TTFT) plunges from hundreds of milliseconds down to **< 10 ms**.

3. **KV Cache Quantization in Extended Contexts (MLA Synergy)**:
   - **Mechanism**: `Ling-3.0-tiny` leverages **MLA (Multi-head Latent Attention)** to compress KV projections to `kv_lora_rank = 512`. When processing 32K ~ 128K documents, pass `--kv-bits 4` or `--kv-bits 8` during CLI generation to compress KV memory overhead by another 50% ~ 75%;
   - **Impact**: Prevents memory bandwidth saturation and avoid decoding throughput collapse during late-stage long-context generation.

4. **System-Level Unified Memory Locking (Wired Memory)**:
   - **Mechanism**: Prevents macOS virtual memory subsystems from paging active model weights out to disk swap during multitasking;
   - **Configuration**: Increase the process Metal allocation ceiling via `sysctl` (requires root privileges):
     ```bash
     sudo sysctl iogpu.wired_mem_limit=30000000000  # Adjust to ~30GB
     ```

+++

### Step 7: Troubleshooting & FAQ

1. **`Model type bailing_hybrid not supported` Error**:
   - Symptom: `mlx_lm.server` or `mlx_lm.generate` fails with `ValueError: Model type bailing_hybrid not supported`.
   - Cause: The installed `mlx-lm` is an older PyPI release (e.g., 0.31.3) before Ling-3.0 support was merged in PR #1711.
   - Solution: Install the latest upstream package from GitHub:
     ```bash
     uv pip install --upgrade "mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git"
     ```

2. **Missing Hugging Face Cache Directory Error (`CacheNotFound`)**:
   - Symptom: Accessing `GET /v1/models` triggers `huggingface_hub.errors.CacheNotFound: Cache directory not found: ~/.cache/huggingface/hub`.
   - Solution: Create the cache directory manually:
     ```bash
     mkdir -p ~/.cache/huggingface/hub
     ```

3. **Unified Memory Pressure and Swap Overhead**:
   - Symptom: Loading BF16 weights on 8GB or 16GB Macs causes heavy disk swapping and system latency.
   - Solution: Use `mlx_lm.convert` to quantize the model to 4-bit (~4.8 GB), lowering RAM consumption and boosting decoding throughput:
     ```bash
     python3 -m mlx_lm.convert \
       --hf-path ~/models/Ling-3.0-tiny \
       --mlx-path ~/models/Ling-3.0-tiny-4bit \
       --quantize \
       --q-bits 4
     ```

4. **Port Conflict (Port 8080 Occupied)**:
   - Symptom: Server launch fails with `Errno 48: Address already in use`.
   - Solution: Terminate the occupying process or select another port:
     ```bash
     lsof -i :8080
     kill -9 <PID>
     ```
     Or pass `--port 8081` in the launch command.
