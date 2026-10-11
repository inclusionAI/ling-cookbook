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

# Ling-3.0-tiny on DGX Spark (SGLang BF16) Deployment Guide

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

Ling-3.0-tiny is a lightweight Sparse MoE language model with 7.9B total parameters and 1.3B active parameters per token. It adopts a hybrid attention architecture combining KDA linear attention and Gated MLA, equipped with 128 routed experts and native support for 128K long context.

This notebook demonstrates how to build `SGLang` from source on `NVIDIA DGX Spark` to deploy the full precision BF16 weights of Ling-3.0-tiny, providing high throughput and low latency API services.

> [!TIP]
> **Environment Recommendations**:
> - We recommend using a **Python 3.11 / 3.12** environment;
> - We recommend using **uv** to manage isolated virtual environments, ensuring clean dependency management and operator compatibility.

+++

### Step 1: Set Up Python Virtual Environment (uv) and Clone SGLang Repository

We recommend using `uv` to create an isolated Python virtual environment and install the OpenAI client, while cloning the official Ling-3.0 support branch (`inclusionAI/sglang:ling_v3_support_mxfp4`):

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
!git clone -b ling_v3_support_mxfp4 https://github.com/inclusionAI/sglang.git
```

Typical execution output:
```text
Using CPython 3.11 interpreter at: /usr/bin/python3.11
Creating virtualenv at: .venv
Cloning into 'sglang'...
remote: Enumerating objects: 38200, done.
Switched to a new branch 'ling_v3_support_mxfp4'
```

+++

### Step 2: Build SGLang from Source with Full Runtime Dependencies

Install the branch code in editable mode inside the virtual environment along with full dependencies (`[all]`). Set `MAX_JOBS=4` to avoid running out of memory during multi-core concurrent compilation:

```{code-cell}
!source .venv/bin/activate && MAX_JOBS=4 uv pip install -e "./sglang/python[all]" --quiet
```

Typical execution output:
```text
Successfully built sglang cuda-tile xatlas
Installing collected packages: ....
Successfully installed sglang
```

+++

### Step 3: Download Official Ling-3.0-tiny Full Precision Model Weights

Official BF16 Safetensors weights are hosted on:
- [Ling-3.0-tiny on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny)
- [Ling-3.0-tiny on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-tiny)

We recommend using the ModelScope CLI to download weights directly to `~/models/Ling-3.0-tiny` (4 shards totaling ~15.8 GB, typically completing in 1-2 minutes):

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope --quiet
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-tiny --local-dir ~/models/Ling-3.0-tiny
```

Typical execution output:
```text
Downloading [config.json, model.safetensors.index.json, ...]
Downloading shard 1/4: 100%|██████████| 4.95G/4.95G [00:10<00:00, 480MB/s]
Downloading shard 2/4: 100%|██████████| 4.98G/4.98G [00:10<00:00, 492MB/s]
Downloading shard 3/4: 100%|██████████| 4.92G/4.92G [00:10<00:00, 475MB/s]
Downloading shard 4/4: 100%|██████████| 1.02G/1.02G [00:02<00:00, 460MB/s]
Successfully downloaded Ling-3.0-tiny to ~/models/Ling-3.0-tiny
```

+++

### Step 4: Launch SGLang HTTP Inference Service

Launch SGLang Server to provide an OpenAI-compatible HTTP interface.

Key optimization parameters:

- `--cuda-graph-backend-decode full --cuda-graph-bs-decode 1 2`: Enables full CUDA Graph capture to reduce scheduling and communication overhead;
- `--moe-runner-backend triton`: Uses Triton MoE expert routing kernels;
- `--json-model-override-args '{"num_nextn_predict_layers":0}'`: Disables auxiliary MTP layer overhead;
- `--mem-fraction-static 0.35`: Pre-allocates 35% of memory (~42 GiB). Model weights take only ~16 GiB, leaving ample cache for 128K context and concurrency;
- `--tool-call-parser ling3 --reasoning-parser ling3`: Enables Ling-3.0 tool calling and reasoning chain parsers;

⚠️ Note:

`sglang.launch_server` runs in the foreground. If executed directly inside this notebook cell, Jupyter will block. We recommend launching the command in a separate terminal window, then running the remaining verification cells here.

```{code-cell}
!source .venv/bin/activate && \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
python3 -m sglang.launch_server \
  --model-path ~/models/Ling-3.0-tiny \
  --served-model-name Ling-3.0-tiny-bf16 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key sk-ling-cookbook-test \
  --mem-fraction-static 0.35 \
  --max-running-requests 16 \
  --max-mamba-cache-size 64 \
  --chunked-prefill-size 8192 \
  --page-size 64 \
  --context-length 131072 \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 2 \
  --cuda-graph-bs-decode 1 2 \
  --cuda-graph-backend-prefill disabled \
  --random-seed 308534008 \
  --reasoning-parser ling3 \
  --tool-call-parser ling3 \
  --attention-backend flashinfer \
  --disable-flashinfer-autotune \
  --moe-runner-backend triton \
  --enable-fp32-lm-head \
  --json-model-override-args '{"num_nextn_predict_layers":0}'
```

Typical startup logs:
```text
Multi-thread loading shards: 100% Completed | 32/32 [01:39<00:00,  3.12s/it]
[2026-08-17 16:52:10] Load weight end. elapsed=100.59 s, type=BailingMoeV3ForCausalLM, avail mem=97.27 GB, mem usage=15.47 GB.
[2026-08-17 16:52:12] Mamba Cache is allocated. max_mamba_cache_size: 614, conv_state size: 0.38GB, ssm_state size: 10.81GB
[2026-08-17 16:52:13] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 1933682, KV size: 12.45 GB
[2026-08-17 16:52:13] Linear attention kernel backend: decode=triton, prefill=triton, verify=triton
[2026-08-17 16:52:13] KDA kernel dispatcher: decode=TritonKDAKernel, verify=TritonKDAKernel, extend=TritonKDAKernel packed_decode=True
[2026-08-17 16:52:13] Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, bs=[1, 2], avail mem=73.10 GB
Capturing batches (bs=1 avail_mem=72.47 GB): 100%|██████████| 2/2 [00:02<00:00,  1.10s/it]
[2026-08-17 16:52:18] Capture target decode CUDA graph end. elapsed=2.58 s, mem usage=0.45 GB, avail mem=72.65 GB.
[2026-08-17 16:52:18] INFO:     Started server process [1963736]
[2026-08-17 16:52:18] INFO:     Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
[2026-08-17 16:52:41] INFO:     127.0.0.1:37664 - "POST /generate HTTP/1.1" 200 OK
[2026-08-17 16:52:41] The server is fired up and ready to roll!
```

+++

### Step 5: Verify Deployment and Test API Calls

Once the service is active, it can be accessed via standard LLM clients (run `pip install openai requests` if not yet installed). The following examples demonstrate:

1. Health check - Verify model loading and endpoint connectivity.
2. Streaming reasoning - Verify `<think>` reasoning chain generation and measure TTFT and Decode TPS.
3. Tool calling - Verify Function Calling with structured outputs.

+++

#### Step 5.1: Connectivity & Model Health Check

Query `GET /v1/models` to check the served model:

```{code-cell}
import urllib.request
import json

url = "http://localhost:30000/v1/models"
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
```text
Health check status: 200
Models response: {"object":"list","data":[{"id":"Ling-3.0-tiny-bf16","object":"model","created":1786898113,"owned_by":"sglang","root":"Ling-3.0-tiny-bf16","parent":null,"max_model_len":131072}]}
```

+++

#### Step 5.2: Streaming Inference, Reasoning, and Benchmark (TTFT / TPS)

Use the OpenAI Python SDK with `enable_thinking: True` to extract reasoning chain tokens and measure TTFT and Decode TPS in real-time:

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-ling-cookbook-test"
)

def verify_streaming_and_thinking():
    prompt = "Derive 25 × 48 and show the calculation steps in detail."
    print(f"Sending prompt: '{prompt}' to model 'Ling-3.0-tiny-bf16'...")

    start_time = time.perf_counter()
    first_token_time = None
    first_content_time = None
    reasoning_chunks = []
    content_chunks = []
    usage_info = None

    try:
        response = client.chat.completions.create(
            model="Ling-3.0-tiny-bf16",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=2048,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True}
            }
        )

        for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                usage_info = chunk.usage

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Extract reasoning chunk (compatible with vLLM/SGLang variations)
            reasoning_piece = (
                getattr(delta, 'reasoning', None)
                or getattr(delta, 'reasoning_content', None)
                or (delta.model_extra.get('reasoning') if hasattr(delta, 'model_extra') and delta.model_extra else None)
            )
            content_piece = delta.content

            if reasoning_piece:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                reasoning_chunks.append(reasoning_piece)

            if content_piece:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                if first_content_time is None:
                    first_content_time = time.perf_counter()
                content_chunks.append(content_piece)

        end_time = time.perf_counter()

        reasoning_text = ''.join(reasoning_chunks).strip()
        final_content = ''.join(content_chunks).strip()

        # Compute latency and throughput
        ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
        ttf_content_ms = (first_content_time - start_time) * 1000.0 if first_content_time else 0.0
        total_duration = end_time - start_time
        decode_duration = end_time - first_token_time if first_token_time else total_duration

        total_tokens = usage_info.completion_tokens if usage_info else (len(reasoning_chunks) + len(content_chunks))
        reasoning_tokens = getattr(getattr(usage_info, 'completion_tokens_details', None), 'reasoning_tokens', None) if usage_info else None
        content_tokens = (total_tokens - reasoning_tokens) if (total_tokens and reasoning_tokens is not None) else len(content_chunks)

        overall_tps = total_tokens / total_duration if total_duration > 0 else 0.0
        decode_tps = total_tokens / decode_duration if decode_duration > 0 else 0.0

        print("\n=== Latency & Throughput Metrics ===")
        print(f"TTFT (Time to First Token): {ttft_ms:.2f} ms")
        if first_content_time:
            print(f"Time to First Content Token: {ttf_content_ms:.2f} ms")
        print(f"Total Duration: {total_duration:.2f} s")
        print(f"Decode Duration: {decode_duration:.2f} s")
        if reasoning_tokens is not None:
            print(f"Total Output Tokens: {total_tokens} (Reasoning: {reasoning_tokens} tokens, Content: {content_tokens} tokens)")
        else:
            print(f"Total Output Tokens: {total_tokens}")
        print(f"Overall Throughput (TPS): {overall_tps:.2f} tokens/s")
        print(f"Decode Throughput (TPS): {decode_tps:.2f} tokens/s")

        print("\n=== Extracted Reasoning Chain ===")
        print(reasoning_text if reasoning_text else "[No separate reasoning chain]")

        print("\n=== Final Response Content ===")
        print(final_content)

    except Exception as e:
        print(f"Streaming verification failed: {e}")

if __name__ == "__main__":
    verify_streaming_and_thinking()
```

Typical output:
```text
Sending prompt: 'Derive 25 × 48 and show the calculation steps in detail.' to model 'Ling-3.0-tiny-bf16'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 42.80 ms
Time to First Content Token: 3120.50 ms
Total Duration: 5.25 s
Decode Duration: 5.21 s
Total Output Tokens: 468 (Reasoning: 280 tokens, Content: 188 tokens)
Overall Throughput (TPS): 79.95 tokens/s
Decode Throughput (TPS): 80.65 tokens/s

=== Extracted Reasoning Chain ===
...
```

+++

#### Step 5.3: Tool Calling and Structured Outputs Test

Verify Function Calling with a standard weather tool schema:

```{code-cell}
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-ling-cookbook-test"
)

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get real-time weather and temperature for a specified city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g., Hangzhou, Beijing, San Francisco"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

def verify_tool_calling():
    print(f"Testing Function Calling / Tool Use with model 'Ling-3.0-tiny-bf16'...")
    try:
        response = client.chat.completions.create(
            model="Ling-3.0-tiny-bf16",
            messages=[
                {"role": "user", "content": "What is the weather in Hangzhou today?"}
            ],
            tools=tools_schema,
            tool_choice="auto"
        )

        message = response.choices[0].message
        if message.tool_calls:
            print("")
            print("=== Function Call Output Detected ===")
            for tool_call in message.tool_calls:
                print(f"Tool Call ID: {tool_call.id}")
                print(f"Function Name: {tool_call.function.name}")
                print(f"Arguments JSON: {tool_call.function.arguments}")
        else:
            print("")
            print("=== Direct Response (No Tool Call Triggered) ===")
            print(message.content)

    except Exception as e:
        print(f"Tool calling verification failed: {e}")

if __name__ == "__main__":
    verify_tool_calling()
```

Typical output:
```text
Testing Function Calling / Tool Use with model 'Ling-3.0-tiny-bf16'...

=== Function Call Output Detected ===
Tool Call ID: call_f3861ca366004a94b162e8b1
Function Name: get_weather
Arguments JSON: {"city": "Hangzhou", "unit": "celsius"}
```

+++

### Step 6: Troubleshooting & FAQ

1. **MLA Assertion Error during `sglang.launch_server` (`AssertionError: K must be a multiple of 1024, got 1536`)**:
   - Cause: On Blackwell architecture, SGLang routes MLA Fused A GEMM to CuteDSL, which requires dimension K to be a multiple of 1024. Ling-3.0-tiny has K=1536, triggering the assertion.
   - Solution: In `sglang/python/sglang/kernels/ops/gemm/fused_a_gemm.py`, modify the `fused_a_gemm_weight_eligible` check by changing `layer.weight.shape[1] % 256 == 0` to `layer.weight.shape[1] % 1024 == 0`. This safely falls back to standard PyTorch/CUDA GEMM kernels (already included in the official branch).

2. **Out of Memory (OOM) during Source Compilation**:
   - Symptom: Process terminated while compiling C++/CUDA extensions.
   - Solution: Restrict concurrent build workers with the `MAX_JOBS=4` environment variable.

3. **Port Conflict (`Port 30000 occupied`)**:
   - Symptom: Server throws `Address already in use` upon startup.
   - Solution: Identify and terminate the occupying process using `lsof -i :30000`, or specify an alternate port with `--port`.
