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

MLX is Apple's open-source machine learning framework engineered specifically for Apple Silicon's unified memory architecture. It features minimal abstraction overhead and direct execution of native Metal kernels.

Support for the Ling-3.0 architecture (`bailing_hybrid` / `BailingMoeV3ForCausalLM`) has been officially merged into upstream `mlx-lm` (PR #1711).

This guide covers the workflow from installation, downloading weights, and verification to launching an OpenAI-compatible HTTP service. For memory-constrained devices (8GB / 16GB), it also provides instructions for converting model weights to 4-bit / MXFP8.

---

### Hardware & Precision Matrix

| Deployment Profile | Weight Format | Disk Size | Recommended RAM (8K Context) | Recommended Mac Hardware | Typical Decode TPS |
|:---|:---|:---:|:---:|:---|:---:|
| BF16 | Native Precision | ~14.72 GB | ≥ 24 GB - 32 GB | MacBook Pro 24GB / 36GB / 48GB+ | ~88.3 tok/s (Empirical) |
| MXFP8 Quantized | MLX Quantized Weights | ~8.06 GB | ≥ 16 GB - 18 GB | MacBook Pro 16GB / 18GB / 24GB | ~118.4 tok/s (Empirical) |
| 4-bit Quantized | MLX Quantized Weights | ~4.83 GB | ≥ 8 GB - 12 GB | MacBook Air / Mac mini 8GB/16GB | ~150.5 tok/s (Empirical) |

> [!TIP]
> **Environment Requirements**:
> - **Python 3.11 / 3.12** is recommended, managed with **uv**;
> - Ling-3.0 architecture (`bailing_hybrid`) requires `mlx-lm>=0.32.0`;
> - Default port convention: `mlx_lm.server` listens on port `8080` by default.

+++

### Step 1: Set Up Python Virtual Environment and Install MLX-LM

Use `uv` to create an isolated Python virtual environment and install `mlx-lm` with native Ling-3.0 support:

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!mkdir -p ~/.cache/huggingface/hub
!source .venv/bin/activate && uv pip install --upgrade mlx "mlx-lm>=0.32.0" 'openai>=1.52.0,<2.0.0' 'modelscope>=1.18.0'
```

+++

### Step 2: Download Official Ling-3.0-tiny BF16 Base Weights

Download the official `inclusionAI/Ling-3.0-tiny` Safetensors base weights (~14.7 GB) from ModelScope or Hugging Face:
- [Ling-3.0-tiny on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny)
- [Ling-3.0-tiny on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-tiny)

```{code-cell}
!mkdir -p ~/models/Ling-3.0-tiny
!source .venv/bin/activate && uv run modelscope download \
  --model inclusionAI/Ling-3.0-tiny \
  --local_dir ~/models/Ling-3.0-tiny
```

> [!IMPORTANT]
> Pre-quantized versions released on ModelScope are optimized specifically for vLLM / SGLang runtimes.
> MLX-LM lacks the corresponding operators; loading ModelScope quantized weights directly into `mlx_lm` will fail.
> Therefore, in the Apple MLX ecosystem, if a quantized model is needed, you must download the BF16 base weights and perform local quantization using MLX's built-in `mlx_lm.convert`.

+++

### Step 3: Deploy Model Using MLX-LM CLI (Using Base Weights as Example)

If device memory is limited and a quantized model is preferred, refer to Step 6 for local quantization using `mlx_lm.convert`.

`mlx_lm.generate` verifies model loading and inference speed directly from the command line, with zero format conversion required:

```{code-cell}
!source .venv/bin/activate && python3 -m mlx_lm.generate \
  --model ~/models/Ling-3.0-tiny \
  --prompt "Calculate 17 × 23 step-by-step with clear derivation." \
  --max-tokens 256 \
  --temp 0.6
```

Typical CLI output (Apple Silicon Mac steady-state empirical data):
```text
Loading model from ~/models/Ling-3.0-tiny...
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
Prompt: 37 tokens, 421.58 tokens-per-sec
Generation: 256 tokens, 88.33 tokens-per-sec
Peak memory: 15.85 GB
```

+++

### Step 4: Launch MLX-LM HTTP Inference Server (OpenAI-Compatible API)

Launch `mlx_lm.server` to expose an OpenAI-compatible HTTP REST endpoint locally.

For single-user scenarios, attaching Prompt Prefix Caching is recommended to reduce first token latency.

```bash
python3 -m mlx_lm.server \
  --model ~/models/Ling-3.0-tiny \
  --host 127.0.0.1 \
  --port 8080 \
  --prompt-cache-size 10 \
  --prompt-cache-bytes 2000000000 \
  --prefill-step-size 2048 \
  --trust-remote-code
```

Typical server startup logs:
```text
Loading model from ~/models/Ling-3.0-tiny...
Model loaded successfully.
INFO:     Started server process [65892]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
```

+++

### Step 5: Verify Deployment and Test API Calling

Once the server is running, use standard OpenAI-compatible client libraries to perform call verification:

#### Step 5.1: Connectivity and Health Check

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
Models response: {"object": "list", "data": [{"id": "~/models/Ling-3.0-tiny", "object": "model", "created": 1788849582}]}
```

+++

#### Step 5.2: Streaming Inference, Reasoning Extraction, and Performance Measurement

Send a streaming completion request using the OpenAI Python SDK. Note: `mlx_lm.server` parses reasoning content into the `delta.reasoning` field:

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="EMPTY"
)

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

+++

#### Step 5.3: Function Calling / Tool Use Verification

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

+++

### Step 6: MLX Quantization for Low-Memory Devices

For Mac devices with 8GB or 16GB unified memory (such as MacBook Air / Mac mini), running BF16 models may cause memory pressure.

You can use `mlx_lm.convert` to generate native MLX quantized weights on-device:

#### Option 1: Convert to 4-bit Quantization (Lower Memory Footprint)
```bash
python3 -m mlx_lm.convert \
  --hf-path ~/models/Ling-3.0-tiny \
  --mlx-path ~/models/Ling-3.0-tiny-4bit \
  --quantize \
  --q-bits 4 \
  --trust-remote-code
```
Memory consumption at runtime is approximately 5 GB after conversion.

#### Option 2: Convert to MXFP8 Quantization (Higher Precision)
```bash
python3 -m mlx_lm.convert \
  --hf-path ~/models/Ling-3.0-tiny \
  --mlx-path ~/models/Ling-3.0-tiny-mxfp8 \
  --quantize \
  --q-mode mxfp8 \
  --trust-remote-code
```
Memory consumption at runtime is approximately 8.7 GB after conversion.

> [!TIP]
> **Deployment Path Update**:
> After quantization, update the `--model` argument to point to the corresponding quantized model directory in subsequent verification or serving commands. For example, to start the inference service with the 4-bit model:
> ```bash
> python3 -m mlx_lm.server \
>   --model ~/models/Ling-3.0-tiny-4bit \
>   --host 127.0.0.1 \
>   --port 8080 \
>   --prompt-cache-size 10 \
>   --prompt-cache-bytes 2000000000 \
>   --prefill-step-size 2048 \
>   --trust-remote-code
> ```

#### Performance Benchmarks

The following data is empirically measured on Apple Silicon M5 Pro 48GB.

| Profile | Weight Format / Type | Disk Size | Prefill Throughput | Decode TPS | Time to First Token (TTFT) | Peak Resident RAM | Recommended Mac Hardware |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| Ling-3.0-tiny | BF16 | 14.72 GB | 421.58 tok/s | 88.33 tok/s (88.1~88.7) | 87.93 ms | 15.85 GB | MacBook Pro 24GB / 36GB / 48GB+ |
| Ling-3.0-tiny-mxfp8 | MXFP8 | 8.06 GB | 505.23 tok/s | 118.35 tok/s (116.7~119.8) | 73.47 ms | 8.75 GB | MacBook Pro 16GB / 18GB / 24GB |
| Ling-3.0-tiny-4bit | 4-bit | 4.83 GB | 683.18 tok/s | 150.51 tok/s (149.4~151.5) | 54.16 ms | 5.30 GB | MacBook Air / Mac mini 8GB / 16GB |

+++

### Step 7: Troubleshooting & FAQ

1. **`Model type bailing_hybrid not supported` Error**:
   - Symptom: Server launch or generation fails with an unknown architecture error.
   - Cause: The installed `mlx-lm` is an older PyPI release before PR #1711.
   - Solution: Update to the latest development version:
     ```bash
     uv pip install --upgrade "mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git"
     ```

2. **Error when Loading Official `int4` / `fp8` Weights**:
   - Symptom: Passing `inclusionAI/Ling-3.0-tiny-int4` throws `ValueError: Received ... parameters not in model`.
   - Solution: Official `int4` and `fp8` weights use vLLM-specific formats. In MLX, download the BF16 base weights and run `mlx_lm.convert --q-bits 4` for native MLX quantization.

3. **Port Conflict (Port 8080 Occupied)**:
   - Solution: Query occupying processes via `lsof -i :8080` and terminate, or launch on `--port 8081`.
