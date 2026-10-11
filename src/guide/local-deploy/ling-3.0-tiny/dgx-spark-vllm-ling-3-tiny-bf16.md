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

# Ling-3.0-tiny on DGX Spark (vLLM BF16) Deployment Guide

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

Ling-3.0-tiny is a lightweight Sparse MoE language model with 7.9B total parameters and 1.3B active parameters per token. It adopts a hybrid attention architecture combining KDA linear attention and MLA, delivering strong long context comprehension and reasoning while maintaining a minimal computational footprint.

This notebook demonstrates how to deploy Ling-3.0-tiny at full BF16 precision on NVIDIA DGX Spark using vLLM.

> [!TIP]
> **Environment Recommendations**:
> - We recommend using a **Python 3.12** environment;
> - We recommend using **uv** to manage isolated virtual environments, ensuring clean dependency management and operator compatibility.

+++

### Step 1: Set Up Python 3.12 Virtual Environment (uv)

We recommend using `uv` to create an isolated Python 3.12 virtual environment and install the `openai` SDK for subsequent API testing and speed benchmarking:

```{code-cell}
!pip install -U uv --quiet
!uv venv --python 3.12 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0' requests
```

Typical execution output:
```text
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtualenv at: .venv
Activate with: source .venv/bin/activate
Resolved 20 packages in 110ms
Installed 20 packages in 40ms
 + openai==1.60.0
```

+++

### Step 2: Install vLLM Official ARM64 CUDA 13.0 Prebuilt Package

Support for the Ling-3.0 architecture and the `ling3` parser is published on the official nightly wheel repository. Install the prebuilt wheel package targeted for GB10 (sm_121 / cu130) using `uv pip` with `--torch-backend=cu130`:

```{code-cell}
!uv pip install \
  --upgrade vllm \
  --torch-backend=cu130 \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130
```

Typical execution output:
```text
⠇ Resolving dependencies...
Installed 180 packages in 120ms
 + vllm==0.19.1+cu130
```

+++

### Step 3: Download Official Ling-3.0-tiny BF16 Model Weights

Official BF16 Safetensors format weights are available on:
- [Ling-3.0-tiny on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny)
- [Ling-3.0-tiny on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-tiny)

Download the weights to `~/models/Ling-3.0-tiny` using the `modelscope` CLI (4 shards totaling ~15.8 GB, typically completing in 1-2 minutes):

```{code-cell}
!uv pip install -U modelscope --quiet
!uv run modelscope download --model inclusionAI/Ling-3.0-tiny --local-dir ~/models/Ling-3.0-tiny
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

### Step 4: Launch vLLM HTTP Inference Service

Launch vLLM Server to provide a standard OpenAI-compatible HTTP interface. Key parameter explanations:
- `--trust-remote-code`: Trusts the Ling MoE hybrid model definition and computational graph;
- `--dtype bfloat16`: Loads model weights and executes inference at full BF16 precision;
- `--gpu-memory-utilization 0.35`: Pre-allocates 35% of memory (~42 GiB), reserving abundant KV cache for 128K long context;
- `--max-model-len 131072`: Enables the native 128K context window of Ling-3.0-tiny;
- `--reasoning-parser ling3`: Enables the specialized Ling-3.0 reasoning parser, extracting `<think>` tags into `reasoning_content`;
- `--enable-auto-tool-choice --tool-call-parser ling3`: Enables Ling-specific tool call parser with automatic tool routing (Tool Calling / Function Calling);
- `--port 30000 --api-key sk-ling-cookbook-test`: Specifies the listening port and authentication API key.

⚠️ Note:

`vllm serve` runs in the foreground. If executed directly inside this notebook cell, Jupyter will block. We recommend launching the command in a separate terminal window, then running the remaining verification cells here.

```{code-cell}
!uv run vllm serve ~/models/Ling-3.0-tiny \
  --served-model-name Ling-3.0-tiny-bf16 \
  --trust-remote-code \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key sk-ling-cookbook-test \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.35 \
  --max-num-seqs 16 \
  --reasoning-parser ling3 \
  --enable-auto-tool-choice \
  --tool-call-parser ling3
```

Typical startup logs:
```text
INFO 08-17 17:35:10 [server.py:120] Route: /v1/chat/completions, Methods: POST
INFO 08-17 17:35:10 [server.py:120] Route: /v1/models, Methods: GET
INFO 08-17 17:35:12 [model_runner.py:1105] Loading model weights took 14.82 GB memory.
INFO 08-17 17:35:14 [worker.py:245] Memory profiling results: total_gpu_memory=121.00GiB, non_torch_memory=1.20GiB, torch_peak_memory=15.20GiB, available_memory=104.60GiB, kv_cache_memory=27.15GiB.
INFO 08-17 17:35:15 [launcher.py:28] Application startup complete.
INFO 08-17 17:35:15 [launcher.py:29] Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
```

+++

### Step 5: Verify Deployment and Test API Calls

Once the service is running, it can be called seamlessly from various LLM clients. The verification suite includes:

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
        body = response.read().decode("utf-8")
        print(f"Health check status: {status_code}")
        print(f"Models response: {body}")
except Exception as e:
    print(f"Health check error: {e}")
```

Typical output:
```text
Health check status: 200
Models response: {"object":"list","data":[{"id":"Ling-3.0-tiny-bf16","object":"model","created":1786968363,"owned_by":"vllm","root":"/home/squall/models/Ling-3.0-tiny","parent":null,"max_model_len":131072,"permission":[{"id":"modelperm-b16332c94d7e25c9","object":"model_permission","created":1786968363,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
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
TTFT (Time to First Token): 86.34 ms
Time to First Content Token: 6054.91 ms
Total Duration: 10.82 s
Decode Duration: 10.74 s
Total Output Tokens: 842 (Reasoning: 467 tokens, Content: 375 tokens)
Overall Throughput (TPS): 77.79 tokens/s
Decode Throughput (TPS): 78.41 tokens/s

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
Tool Call ID: call_87804adb3c954cb4b5dcc266
Function Name: get_weather
Arguments JSON: {"city": "Hangzhou", "unit": "celsius"}
```

+++

### Step 6: Troubleshooting & FAQ

1. **Memory Pre-allocation and 128K Long Context**:
   - Note: Ling-3.0-tiny BF16 weights take only ~16 GiB. The default `--gpu-memory-utilization 0.35` (~42 GiB) is sufficient to accommodate full 128K context and multi-request concurrency. You can adjust this ratio for higher concurrency demands.

2. **Port Conflict (`Port 30000 occupied`)**:
   - Symptom: Server throws `Address already in use` upon startup.
   - Solution: Identify and terminate the occupying process using `lsof -i :30000`, or specify an alternate port with `--port`.
