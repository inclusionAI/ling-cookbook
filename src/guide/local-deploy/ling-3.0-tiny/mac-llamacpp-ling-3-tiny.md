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

# Ling-3.0-tiny on Apple Silicon Mac (llama.cpp Metal GGUF) Deployment Guide

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

`Ling-3.0-tiny` is a 7.9B lightweight Sparse MoE language model in the Ling model family with 1.3B active parameters per token, natively supporting a 128K context window. Optimized for Apple Silicon Mac's unified memory architecture, running official GGUF quantized models with `llama.cpp` native Metal hardware acceleration achieves ultra-low memory footprint, high-throughput on-device inference, and agent tool calling capabilities on consumer-grade Macs.

This guide demonstrates how to install `llama.cpp` on Apple Silicon Mac (M1 / M2 / M3 / M4 / M5 series) via Homebrew, launch the `Ling-3.0-tiny` GGUF inference service, and serve an OpenAI-compatible HTTP REST API.

---

### Hardware & Quantization Matrix

`llama.cpp` has merged native support for the Ling-3.0 architecture in the `master` branch (PR #26608). Select the profile that best matches your Mac device's unified memory capacity:

| Deployment Profile | Quantization Type | Model Disk Size | Recommended Memory (8K Context) | Supported Mac Hardware | Tested Decode TPS |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Q4_K_M (Recommended)** | 4-bit GGUF | **~4.80 GB** | **≥ 8 GB** | MacBook Air / Mac mini / MacBook Pro 8GB/16GB | **~123.7 tok/s (Measured)** |
| **Q8_0 (High Fidelity)** | 8-bit GGUF | **~7.80 GB** | **≥ 16 GB - 18 GB** | MacBook Pro 16GB / 18GB / 24GB | **~110.2 tok/s (Measured)** |
| **BF16 (Full Precision)** | 16-bit GGUF | **~15.80 GB** | **≥ 16 GB - 24 GB** | MacBook Pro 18GB / 24GB / 36GB+ | **~81.6 tok/s (Measured)** |

> [!TIP]
> **Environment & Dependency Recommendations**:
> - Recommended OS: **macOS 14+ (Sonoma / Sequoia)**;
> - We recommend using **uv** to create an isolated Python 3.11 / 3.12 virtual environment;
> - `llama.cpp` supports offloading all model layers to GPU (`-ngl all`), leveraging Apple Metal API to accelerate matrix computations.

+++

### Step 1: Prepare Environment and Install llama.cpp

#### Step 1.1: Install llama.cpp and uv via Homebrew

On Apple Silicon Mac, we recommend installing the precompiled binary with native Metal hardware acceleration directly via Homebrew:

```{code-cell}
# Install Homebrew llama.cpp and uv (skip if already installed)
!brew install llama.cpp uv
```

Typical installation output:
```text
==> Downloading bottle manifests
✔︎ Bottle Manifest llama.cpp (0.4.0)                 Downloaded   94.0KB/ 94.0KB
==> Would install 1 formula:
llama.cpp
==> Would install 2 dependencies for llama.cpp:
libomp
ggml
==> Installing dependencies for llama.cpp: libomp and ggml
==> Installing llama.cpp dependency: libomp
==> Pouring libomp--23.1.0.arm64_tahoe.bottle.tar.gz
🍺  /opt/homebrew/Cellar/libomp/23.1.0: 11 files, 1.8MB
==> Installing llama.cpp dependency: ggml
==> Pouring ggml--0.23.0.arm64_tahoe.bottle.tar.gz
🍺  /opt/homebrew/Cellar/ggml/0.23.0: 38 files, 4.8MB
==> Installing llama.cpp
==> Pouring llama.cpp--0.4.0.arm64_tahoe.bottle.tar.gz
🍺  /opt/homebrew/Cellar/llama.cpp/0.4.0: 70 files, 19MB
```

+++

#### Step 1.2: Prepare Python Virtual Environment and Client Dependencies

Use `uv` to create a virtual environment and install `openai` and `modelscope` client libraries:

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0' 'modelscope>=1.18.0'
```

Typical execution output:
```text
Resolved 27 packages in 506ms
Prepared 17 packages in 903ms
Installed 17 packages in 51ms
 + httpcore==1.0.9
 + httpx==0.28.1
 + modelscope==1.39.1
 + modelscope-hub==0.4.0
 + openai==1.109.1
 + requests==2.34.2
 + tqdm==4.70.0
```

+++

### Step 2: Download Official Ling-3.0-tiny GGUF Model Weights

Official pre-quantized GGUF weights are available on ModelScope and Hugging Face:
- [Ling-3.0-tiny-GGUF on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny-GGUF)
- [Ling-3.0-tiny-GGUF on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-tiny-GGUF)

We recommend downloading the desired precision to `~/models/Ling-3.0-tiny-GGUF` using the `modelscope` CLI:

#### Option A: BF16 Full Precision Version

Original full-precision weights without quantization loss, suitable for Mac devices with unified memory ≥ 24GB / 32GB (e.g., MacBook Pro M3/M4/M5 series):

> [!NOTE]
> This step downloads ~15.8 GB of full-precision model weights from ModelScope. It may take some time depending on your network connection.

```{code-cell}
!mkdir -p ~/models/Ling-3.0-tiny-GGUF
!source .venv/bin/activate && uv run modelscope download \
  --model inclusionAI/Ling-3.0-tiny-GGUF Ling-3.0-tiny-bf16.gguf \
  --local-dir ~/models/Ling-3.0-tiny-GGUF
```

Typical download output:
```text
Ling-3.0-tiny-bf16.gguf: 100%|██████████| 15.8G/15.8G [14:53<00:00, 18.5MB/s]
✓ Ling-3.0-tiny-bf16.gguf → /Users/sipan/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-bf16.gguf
```

#### Option B: Q8_0 Quantized Version

High-fidelity 8-bit quantization, suitable for devices with 18GB / 24GB memory:

```{code-cell}
!mkdir -p ~/models/Ling-3.0-tiny-GGUF
!source .venv/bin/activate && uv run modelscope download \
  --model inclusionAI/Ling-3.0-tiny-GGUF Ling-3.0-tiny-Q8_0.gguf \
  --local-dir ~/models/Ling-3.0-tiny-GGUF
```

Typical download output:
```text
✓ Ling-3.0-tiny-Q8_0.gguf → /Users/sipan/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q8_0.gguf
```

#### Option C: Q4_K_M Quantized Version

Compact 4-bit quantization, suitable for 8GB / 16GB consumer Macs (e.g., MacBook Air / Mac mini):

```{code-cell}
!mkdir -p ~/models/Ling-3.0-tiny-GGUF
!source .venv/bin/activate && uv run modelscope download \
  --model inclusionAI/Ling-3.0-tiny-GGUF Ling-3.0-tiny-Q4_K_M.gguf \
  --local-dir ~/models/Ling-3.0-tiny-GGUF
```

Typical download output:
```text
✓ Ling-3.0-tiny-Q4_K_M.gguf → /Users/sipan/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q4_K_M.gguf
```

+++

### Step 3: Launch llama-server HTTP Inference Service (Metal Accelerated)

Launch `llama-server` to load the GGUF weights and expose OpenAI-compatible HTTP REST endpoints. Key command-line options:
- `-m ~/models/.../Ling-3.0-tiny-*.gguf`: Path to the local GGUF model file
- `--alias <model-alias>`: Canonical model alias matching the precision for explicit client requests
- `-ngl all`: Offload all model layers to Apple Metal GPU unified memory
- `-fa on`: Enable FlashAttention kernel acceleration
- `-c 32768`: Set context window size to 32K (can be scaled from 16K to 128K based on available memory)
- `-np 2`: Configure 2 concurrent request processing slots
- `--host 127.0.0.1 --port 9102`: Set server listen address and port (canonical port `9102`)
- `--api-key sk-ling-cookbook-test`: Configure client access authentication token

⚠️ Important Note:

`llama-server` runs in the foreground by default. Executing it directly inside a Notebook cell will block the kernel. **We recommend running the server launch command in an independent terminal session**. Once the server is ready, proceed with the verification steps in this Notebook. Select the corresponding command based on the precision downloaded in Step 2:

#### Option A: Launch BF16 Full Precision Server

```bash
llama-server \
  -m ~/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-bf16.gguf \
  --alias Ling-3.0-tiny-bf16 \
  -ngl all \
  -fa on \
  -c 32768 \
  -np 2 \
  --host 127.0.0.1 \
  --port 9102 \
  --api-key sk-ling-cookbook-test
```

Typical server startup logs:
```text
0.00.117.743 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.121.571 I srv    load_model: loading model '/Users/sipan/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-bf16.gguf'
0.00.311.892 W load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
0.03.146.450 I cmn          init: llama threadpool init, n_threads = 5
0.03.310.140 I srv    load_model: initializing, n_slots = 2, n_ctx_slot = 16384, kv_unified = 'false'
0.03.319.396 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
0.03.319.402 I srv  llama_server: model loaded
0.03.319.405 I srv  llama_server: listening on http://127.0.0.1:9102
```

#### Option B: Launch Q8_0 High-Fidelity Server

```bash
llama-server \
  -m ~/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q8_0.gguf \
  --alias Ling-3.0-tiny-Q8_0 \
  -ngl all \
  -fa on \
  -c 32768 \
  -np 2 \
  --host 127.0.0.1 \
  --port 9102 \
  --api-key sk-ling-cookbook-test
```

Typical server startup logs:
```text
0.00.128.707 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.130.297 I srv    load_model: loading model '/Users/sipan/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q8_0.gguf'
0.00.321.842 W load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
0.01.720.776 I cmn          init: llama threadpool init, n_threads = 5
0.01.762.667 I srv    load_model: initializing, n_slots = 2, n_ctx_slot = 16384, kv_unified = 'false'
0.01.770.021 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
0.01.770.030 I srv  llama_server: model loaded
0.01.770.033 I srv  llama_server: listening on http://127.0.0.1:9102
```

#### Option C: Launch Q4_K_M Lightweight Server

```bash
llama-server \
  -m ~/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q4_K_M.gguf \
  --alias Ling-3.0-tiny-Q4_K_M \
  -ngl all \
  -fa on \
  -c 32768 \
  -np 2 \
  --host 127.0.0.1 \
  --port 9102 \
  --api-key sk-ling-cookbook-test
```

Typical server startup logs:
```text
0.00.079.933 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.081.423 I srv    load_model: loading model '/Users/sipan/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q4_K_M.gguf'
0.00.280.360 W load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
0.01.339.747 I cmn          init: llama threadpool init, n_threads = 5
0.01.379.077 I srv    load_model: initializing, n_slots = 2, n_ctx_slot = 16384, kv_unified = 'false'
0.01.387.025 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
0.01.387.035 I srv  llama_server: model loaded
0.01.387.037 I srv  llama_server: listening on http://127.0.0.1:9102
```

+++

### Step 4: Verify Deployment and Test Model Inference

Once the server is running, you can connect using any OpenAI-compatible client. Below are 3 end-to-end tests:

1. **Health Check** - Verify model load status and HTTP endpoint reachability;
2. **Streaming Inference & `<think>` Deep Thinking Test** - Validate reasoning chain generation and measure TTFT and Decode TPS throughput benchmarks;
3. **Function Calling Tool Test** - Verify structured parameter extraction and tool invocation.

+++

#### Step 4.1: Endpoint Reachability & Model Health Check

Request `GET /v1/models` to inspect running model metadata:

```{code-cell}
import urllib.request
import json

url = "http://127.0.0.1:9102/v1/models"
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
Health check status: 200
Models response: {"models":[{"name":"Ling-3.0-tiny","model":"Ling-3.0-tiny","modified_at":"","size":"","digest":"","type":"model","description":"","tags":[""],"capabilities":["completion"],"parameters":"","details":{"parent_model":"","format":"gguf","family":"","families":[""],"parameter_size":"","quantization_level":""}}],"object":"list","data":[{"id":"Ling-3.0-tiny","aliases":["Ling-3.0-tiny"],"tags":[],"object":"model","created":1788805461,"owned_by":"llamacpp","meta":{"vocab_type":true,"n_vocab":157184,"n_ctx":16384,"n_ctx_train":131072,"n_embd":1536,"n_params":7893392800,"size":15796958848,"ftype":"BF16"}}]}
```

+++

#### Step 4.2: Streaming Inference, Reasoning Chain Extraction, and Throughput Measurement

Use the OpenAI Python SDK with `enable_thinking: True` to extract `<think>` reasoning chain tokens and measure TTFT and Decode TPS:

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9102/v1",
    api_key="sk-ling-cookbook-test"
)

# Replace with the profile you launched ("Ling-3.0-tiny-bf16", "Ling-3.0-tiny-Q8_0", or "Ling-3.0-tiny-Q4_K_M")
MODEL_NAME = "Ling-3.0-tiny-bf16"

def verify_streaming_and_thinking():
    prompt = "Compute 17 × 23 and provide step-by-step reasoning logic."
    print(f"Sending prompt: '{prompt}' to model '{MODEL_NAME}'...")

    start_time = time.time()
    first_token_time = None
    total_tokens = 0
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            temperature=0.6,
            top_p=0.95,
            stream=True
        )

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            now = time.time()
            if first_token_time is None:
                first_token_time = now

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_text += delta.reasoning_content
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
        print(f"Decode TPS (Tokens/s): {tps:.2f} t/s")
        print(f"Total Generated Tokens: {total_tokens}")
        print(f"Total Duration: {end_time - start_time:.2f} s")

        print("\n=== Extracted Reasoning Chain (<think>) ===")
        print(reasoning_text if reasoning_text else "[Note: Reasoning merged in main content or parsed separately]")

        print("\n=== Final Response Content ===")
        print(content_text)

    except Exception as e:
        print(f"Streaming verification failed: {e}")

if __name__ == "__main__":
    verify_streaming_and_thinking()
```

Typical execution output (empirically measured on Apple Silicon Mac):
```text
Sending prompt: 'Compute 17 × 23 and provide step-by-step reasoning logic.' to model 'Ling-3.0-tiny-bf16'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 395.72 ms
Decode TPS (Tokens/s): 81.57 t/s
Total Generated Tokens: 2377
Total Duration: 29.54 s

=== Extracted Reasoning Chain (<think>) ===
1. Analyze the request:
   - Task: Compute 17 × 23.
   - Requirement: Provide step-by-step reasoning logic.

2. Select approach:
   - Method 1: Distributive property (factoring), demonstrates mathematical logic.
   - Method 2: Long multiplication (standard algorithm).

3. Draft step-by-step derivation:
   - Split 17 into 10 + 7
   - Apply distributive property: (10 + 7) × 23 = 10 × 23 + 7 × 23
   - Compute components: 10 × 23 = 230, 7 × 23 = 161
   - Sum: 230 + 161 = 391

=== Final Response Content ===
To compute 17 × 23, here are two step-by-step derivation methods:

### Method 1: Distributive Property (Mental Math)
1. Split 17 into 10 and 7: 17 = 10 + 7
2. Apply the distributive property: 17 × 23 = (10 + 7) × 23 = 10 × 23 + 7 × 23
3. Compute each component: 10 × 23 = 230, 7 × 23 = 161
4. Add the results: 230 + 161 = 391

### Method 2: Standard Long Multiplication
1. Multiply 17 by the units digit 3: 51
2. Multiply 17 by the tens digit 2: 340
3. Add the partial products: 51 + 340 = 391

### Final Result
17 × 23 = 391
```

+++

#### Step 4.3: Test Function Calling Tool Invocation

Provide a standard Function Calling schema (using currency conversion as an example) to verify the model's native tool calling capabilities:

```{code-cell}
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9102/v1",
    api_key="sk-ling-cookbook-test"
)

# Replace with the profile you launched ("Ling-3.0-tiny-bf16", "Ling-3.0-tiny-Q8_0", or "Ling-3.0-tiny-Q4_K_M")
MODEL_NAME = "Ling-3.0-tiny-bf16"

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert real-time exchange rates between specified currencies",
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
    print(f"Testing Function Calling with model '{MODEL_NAME}'...")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "Please convert 100 USD to CNY and calculate how much I will receive."}
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
        else:
            print("\n=== Direct Response (No Tool Call Triggered) ===")
            print(message.content)

    except Exception as e:
        print(f"Tool calling verification failed: {e}")

if __name__ == "__main__":
    verify_tool_calling()
```

Typical test output:
```text
Testing Function Calling with model 'Ling-3.0-tiny-bf16'...

=== Function Call Output Detected ===
Tool Call ID: call_cur_9823f0a1
Function Name: convert_currency
Arguments JSON: {"from_currency": "USD", "to_currency": "CNY", "amount": 100}
```

+++

#### Step 4.4: Decode Throughput and Memory Benchmark Across 3 Precisions

Below are comprehensive empirical benchmark results measured on an Apple Silicon Mac (M5 Pro, 48GB unified memory, macOS 15) running `Ling-3.0-tiny` across all three precision profiles (BF16, Q8_0, Q4_K_M) with end-to-end streaming generation and memory profiling:

| Model Profile / Alias | Quantization Type | Model Disk Size | TTFT (Time to First Token) | Decode Throughput (TPS) | Idle Memory (RSS) | Peak Memory (Peak RSS) | Dynamic KV Cache Delta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Ling-3.0-tiny-bf16`** | BF16 (Full Precision) | **15.8 GB** | 395.72 ms | **81.57 tok/s** | 11.88 GB | **12.07 GB** | 0.18 GB (~188 MB) |
| **`Ling-3.0-tiny-Q8_0`** | 8-bit GGUF (High Fidelity) | **7.8 GB** | 271.69 ms | **110.18 tok/s** | 8.13 GB | **8.17 GB** | 0.04 GB (~43 MB) |
| **`Ling-3.0-tiny-Q4_K_M`** | 4-bit GGUF (Recommended) | **4.8 GB** | 248.72 ms | **123.69 tok/s** | 4.17 GB | **4.21 GB** | 0.04 GB (~46 MB) |

> [!TIP]
> **Hardware Selection & Deployment Advice**:
> - **8GB / 16GB Memory Devices** (e.g., MacBook Air / Mac mini / MacBook Pro 8GB/16GB): Choose **`Q4_K_M`**. Peak physical memory is only **~4.2 GB**, decode throughput reaches **~123.7 tok/s**, providing instantaneous on-device response;
> - **18GB / 24GB Memory Devices** (e.g., MacBook Pro M3/M4 series): Choose **`Q8_0`**. High-fidelity 8-bit quantization retains virtually lossless mathematical reasoning, achieves **~110.2 tok/s**, and requires only **~8.2 GB** peak physical memory;
> - **32GB+ Memory Devices** (e.g., MacBook Pro M-Max series): Run **`BF16`** directly for zero quantization loss, delivering stable **~81.6 tok/s** decode throughput with **~12.1 GB** resident memory.

+++

### Step 5: Troubleshooting & FAQ

1. **Metal Unified Memory Allocation Limit**:
   - Issue: macOS restricts a single process from allocating more than ~75% of physical unified memory by default. When loading large context windows, you may encounter `ggml_metal_init: error allocating buffer`.
   - Solution: You can temporarily raise the allocation ceiling via `sysctl` (requires root privileges):
     ```bash
     sudo sysctl iogpu.wired_mem_limit=30000000000  # Adjust to ~30GB
     ```

2. **Context Window Out of Memory (OOM)**:
   - Issue: On 8GB or 16GB Macs, configuring excessive context like `-c 131072` alongside `-np 4` can cause memory exhaustion or crashes.
   - Solution: On entry-level machines, limit context size to `-c 32768` or `-c 16384`, and keep concurrency slots at `-np 1` or `-np 2`.

3. **Port Conflict (Port 9102 Occupied)**:
   - Issue: `llama-server` errors with `failed to bind port 9102` or `Address already in use`.
   - Solution: Identify and terminate the occupying process, or launch on an alternative port:
     ```bash
     lsof -i :9102
     kill -9 <PID>
     ```
     Or specify `--port 9103` in the launch command.
