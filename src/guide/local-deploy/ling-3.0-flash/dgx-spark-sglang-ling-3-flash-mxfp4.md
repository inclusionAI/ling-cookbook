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

# Ling-3.0-flash on DGX Spark (SGLang MXFP4) Deployment Guide

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

Ling-3.0-flash is an MoE large language model with 124B total parameters and 5.1B active parameters per token.

This notebook demonstrates how to build `SGLang` from source on `NVIDIA DGX Spark` (GB10 121GB unified memory) to deploy the MXFP4 quantized version of Ling-3.0-flash. The quantized model weights take only ~60.5 GB, enabling high-performance inference on a single DGX Spark.

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

Install the branch code in editable mode along with full runtime dependencies (`[all]`). Set `MAX_JOBS=4` to prevent multi-core concurrent compilation from exhausting memory:

```{code-cell}
!source .venv/bin/activate && MAX_JOBS=4 uv pip install -e "./sglang/python[all]"
```

Typical execution output:
```text
Requirement already satisfied: pip in ...
Installing collected packages: sglang
  Running setup.py develop for sglang
Successfully installed sglang
```

+++

### Step 3: Download Ling-3.0-flash FP4 (MXFP4) Model Weights

Pre-quantized FP4 (MXFP4) weights are hosted on:
- [Ling-3.0-flash-fp4 on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-fp4)
- [Ling-3.0-flash-fp4 on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-fp4)

Download weights to `~/models/Ling-3.0-flash-fp4` using the ModelScope CLI (or Hugging Face CLI):

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-fp4 --local-dir ~/models/Ling-3.0-flash-fp4
```

Typical execution output:
```text
Downloading [model-00024-of-00024.safetensors]: 100%|█| 2.80G/2.80G [01:15<00:00]
Processing 35 items: 100%|███████████████████| 35.0/35.0 [09:40<00:00, 16.5s/it]
Successfully downloaded Ling-3.0-flash-fp4 to ~/models/Ling-3.0-flash-fp4
```

+++

### Step 4: Launch SGLang HTTP Inference Service (Marlin Kernel Acceleration)

Launch SGLang Server to provide an OpenAI-compatible HTTP interface. For MXFP4 weights, the deployment uses the Marlin MoE kernel backend (`--moe-runner-backend marlin`), balancing high throughput with numerical stability.

Key parameter explanations:
- `--moe-runner-backend marlin`: Enables the Marlin hardware acceleration kernel optimized for MXFP4 expert matrices
- `--cuda-graph-backend-decode full --cuda-graph-max-bs-decode 1`: Enables full CUDA Graph capture for single-batch decoding to reduce scheduling latency
- `--attention-backend flashinfer --fp8-gemm-backend cutlass`: Enables FlashInfer attention and CUTLASS GEMM acceleration
- `--mem-fraction-static 0.75`: Pre-allocates 75% memory, reserving ample KV cache for long context
- `--tool-call-parser ling3 --reasoning-parser ling3`: Enables Ling-3.0 tool calling and reasoning chain parsers
- `--json-model-override-args`: Configures YaRN RoPE Scaling to natively support a 256K long context window

⚠️ Note:

`sglang.launch_server` runs in the foreground. If executed directly inside this notebook cell, Jupyter will block. We recommend launching the command in a separate terminal window, then running the remaining verification cells here.

```{code-cell}
!source .venv/bin/activate && \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_DSV4_FP4_DEQUANT=0 \
SGLANG_FP8_IGNORED_LAYERS="" \
python3 -m sglang.launch_server \
  --model-path ~/models/Ling-3.0-flash-fp4 \
  --served-model-name ling-v3-flash-fp4 \
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
  --page-size 64 \
  --context-length 262144 \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-bs-decode 1 \
  --cuda-graph-backend-prefill disabled \
  --random-seed 308534008 \
  --reasoning-parser ling3 \
  --tool-call-parser ling3 \
  --attention-backend flashinfer \
  --disable-flashinfer-autotune \
  --mem-fraction-static 0.75 \
  --fp8-gemm-backend cutlass \
  --moe-runner-backend marlin \
  --disable-shared-experts-fusion \
  --enable-fp32-lm-head \
  --json-model-override-args '{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}'
```

Typical startup logs:
```text
[2026-08-15 22:08:22] Load weight end. elapsed=346.90 s, type=BailingMoeV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=47.35 GB, mem usage=64.88 GB.
[2026-08-15 22:08:26] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 1912896, KV size: 14.37 GB
[2026-08-15 22:08:27] Uvicorn running on http://0.0.0.0:30000
[2026-08-15 22:09:00] The server is fired up and ready to roll!
```

+++

### Step 5: Verify Deployment and Test API Calls

Once the service is active, it can be called using standard LLM clients. The verification suite includes:

1. **Health Check** - Verify model status and endpoint availability.
2. **Streaming Reasoning** - Test `<think>` reasoning chain generation while measuring TTFT and Decode TPS.
3. **Tool Calling** - Test Function Calling with structured outputs.

+++

#### Step 5.1: Connectivity & Model Health Check

Query `GET /v1/models` to inspect the served model:

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
```json
{
  "object": "list",
  "data": [
    {
      "id": "ling-v3-flash-fp4",
      "object": "model",
      "created": 1786799000,
      "owned_by": "sglang"
    }
  ]
}
```

+++

#### Step 5.2: Streaming Inference, Reasoning, and Benchmark (TTFT / TPS)

Use the OpenAI Python SDK with `enable_thinking: True` to extract `<think>` reasoning chain tokens, and use `stream_options={"include_usage": True}` to obtain official token counts to calculate TTFT and Decode TPS:

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-ling-cookbook-test"
)

def verify_streaming_and_thinking():
    prompt = "Derive 17 × 23 and show the calculation steps in detail."
    print(f"Sending prompt: '{prompt}' to model 'ling-v3-flash-fp4'...")
    
    start_time = time.time()
    first_token_time = None
    chunk_count = 0
    exact_completion_tokens = None
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model="ling-v3-flash-fp4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            temperature=0.6,
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
            if first_token_time is None and (delta.content or (hasattr(delta, 'reasoning_content') and delta.reasoning_content)):
                first_token_time = now

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_text += delta.reasoning_content
                chunk_count += 1
            elif delta.content:
                content_text += delta.content
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
        print(reasoning_text if reasoning_text else "[Note: Reasoning merged in main content or parsed separately]")
        
        print("\n=== Final Response Content ===")
        print(content_text)

    except Exception as e:
        print(f"Streaming verification failed: {e}")

if __name__ == "__main__":
    verify_streaming_and_thinking()
```

Typical output:
```text
Sending prompt: 'Derive 17 × 23 and show the calculation steps in detail.' to model 'ling-v3-flash-fp4'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 135.16 ms
Decode TPS (Tokens/s): 40.55 t/s
Total Generated Tokens: 965 (Exact tokens from usage)
Total Duration: 23.93 s

=== Extracted Reasoning Chain (<think>) ===
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
    print(f"Testing Function Calling / Tool Use with model 'ling-v3-flash-fp4'...")
    try:
        response = client.chat.completions.create(
            model="ling-v3-flash-fp4",
            messages=[
                {"role": "user", "content": "What is the weather in Hangzhou today?"}
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

Typical output:
```text
Testing Function Calling / Tool Use with model 'ling-v3-flash-fp4'...

=== Function Call Output Detected ===
Tool Call ID: call_fc09b2cf0a3d4accb936277f
Function Name: get_weather
Arguments JSON: {"city": "Hangzhou", "unit": "celsius"}
```

+++

### Step 6: Troubleshooting & FAQ

1. **Out of Memory (OOM) during Source Compilation**:
   - Symptom: Process terminated by system while compiling C++/CUDA extensions with `MAX_JOBS=4 uv pip install -e "./sglang/python[all]"`.
   - Solution: Restrict concurrent build workers using `MAX_JOBS=4` or `MAX_JOBS=2` to reduce memory pressure.

2. **Port Conflict (`Port 30000 occupied`)**:
   - Symptom: Server throws `Address already in use` upon startup.
   - Solution: Terminate occupying processes via `lsof -i :30000`, or specify an alternate port with `--port`.

3. **Memory Allocation and CUDA Graph Capture**:
   - Symptom: CUDA Out of Memory error during the `Capture target decode CUDA graph` phase at startup.
   - Solution: Adjust `--mem-fraction-static` slightly (e.g., from 0.75 down to 0.70) to ensure sufficient headroom for CUDA Graph memory allocation.
