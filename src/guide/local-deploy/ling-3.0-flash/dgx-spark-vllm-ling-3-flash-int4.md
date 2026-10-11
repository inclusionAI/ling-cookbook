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

# Deployment Guide: Ling-3.0-flash on DGX Spark (vLLM INT4)

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

Ling-3.0-flash is a Mixture-of-Experts (MoE) large language model with 124B total parameters and 5.1B activated parameters per token.

This notebook demonstrates how to deploy the INT4 quantized version of Ling-3.0-flash on NVIDIA DGX Spark using the official vLLM ARM64 CUDA 13.0 prebuilt wheel.

> [!TIP]
> **Environment Recommendations**:
> - **Python 3.12** is recommended.
> - Using **uv** to create an isolated Python virtual environment is recommended to ensure dependency isolation and operator compatibility.

+++

### Step 1: Set Up Python 3.12 Virtual Environment (uv)

Create an isolated Python 3.12 virtual environment using `uv` to ensure clean dependencies. Install the `openai` package for OpenAI-compatible API calls:

```{code-cell}
!pip install -U uv
!uv venv --python 3.12 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
```

Typical output:
```text
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtualenv at: .venv
Activate with: source .venv/bin/activate
Resolved 20 packages in 120ms
Installed 20 packages in 45ms
 + openai==1.60.0
```

+++

### Step 2: Install Official vLLM ARM64 CUDA 13.0 Prebuilt Wheel

Support for the Ling-3.0 architecture and the `ling3` parser has been merged into the official vLLM wheel distribution. Install the prebuilt package built for GB10 (sm_121 / cu130) by specifying `--torch-backend=cu130` via `uv pip`:

```{code-cell}
!uv pip install \
  --upgrade vllm \
  --torch-backend=cu130 \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130
```

Typical output:
```text
⠇ Resolving dependencies...
Installed 180 packages in 114ms
 + aiohappyeyeballs==2.7.1
 + aiohttp==3.14.3
```

+++

### Step 3: Download Ling-3.0-flash INT4 Model Weights

Download the `inclusionAI/Ling-3.0-flash-int4` model weights from ModelScope or Hugging Face to a local directory. Using the `modelscope` or `huggingface` CLI tool is recommended. Below is the `modelscope` example:

- [Ling-3.0-flash-int4 on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-int4)
- [Ling-3.0-flash-int4 on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-int4)

```{code-cell}
!uv pip install -U modelscope
!uv run modelscope download --model inclusionAI/Ling-3.0-flash-int4 --local-dir ~/models/Ling-3.0-flash-int4
```

Typical output:
```text
Downloading [model_files]... 100%|██████████| 62.1G/62.1G [02:10<00:00, 490MB/s]
Successfully downloaded to /home/ubuntu/models/Ling-3.0-flash-int4
```

+++

### Step 4: Launch vLLM Inference Service

Launch `vllm` to load the model and start the OpenAI-compatible HTTP API server.

⚠️ Note: `vllm` runs in foreground persistent mode by default. If run directly inside a Jupyter Notebook cell, it will occupy the execution kernel indefinitely, preventing interaction with subsequent cells. You should run the launch command in an independent terminal session.

```{code-cell}
!uv run vllm serve ~/models/Ling-3.0-flash-int4 \
  --served-model-name ling-3.0-flash-int4 \
  --port 30000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --reasoning-parser ling3 \
  --enable-auto-tool-choice \
  --tool-call-parser ling3 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --api-key sk-ling-cookbook-test \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Typical output when server is ready:
```text
INFO: Initializing an OpenAI-compatible API server on http://0.0.0.0:30000
INFO: Route: /v1/chat/completions, Methods: POST
INFO: Route: /health, Methods: GET
INFO: Model loaded: ling-3.0-flash-int4 (Quantization: compressed-tensors/marlin, TP: 1)
INFO: MTP Speculative Decoding enabled: num_speculative_tokens=3
INFO: Server ready to receive incoming requests.
```

+++

### Step 5: Verify Deployment and Test API Calls

Once the service is started, perform end-to-end verification via an OpenAI-compatible client:

1. Health Check - Verify model endpoint connectivity.
2. Streaming Reasoning Verification and Speed Measurement - Test `<think>` reasoning chain parsing, and measure TTFT and generation throughput.
3. Function Calling Test - Verify structured tool calling support.

```{code-cell}
import urllib.request

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
      "id": "ling-3.0-flash-int4",
      "object": "model",
      "owned_by": "vllm"
    }
  ]
}
```

+++

#### Step 5.2: Test Streaming Inference, Reasoning, and Throughput Metrics

Use the OpenAI Python SDK to initiate requests, enable the reasoning chain via `chat_template_kwargs={"enable_thinking": True}`, and measure TTFT and Decode TPS throughput metrics:

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-ling-cookbook-test"
)

def verify_streaming_and_thinking():
    prompt = "请推导公式 17 × 23 并详细给出计算步骤。"
    print(f"Sending prompt: '{prompt}' to model 'ling-3.0-flash-int4'...")

    start_time = time.perf_counter()
    first_token_time = None
    first_content_time = None
    reasoning_chunks = []
    content_chunks = []
    usage_info = None

    try:
        response = client.chat.completions.create(
            model="ling-3.0-flash-int4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=1024,
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

            # Extract reasoning snippet (vLLM ling3 parser returns delta.reasoning directly)
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

        # Calculate latency and generation throughput
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
            print(f"Time to First Content: {ttf_content_ms:.2f} ms")
        print(f"Total Duration: {total_duration:.2f} s")
        print(f"Decode Duration: {decode_duration:.2f} s")
        if reasoning_tokens is not None:
            print(f"Total Generated Tokens: {total_tokens} (Reasoning: {reasoning_tokens} tokens, Content: {content_tokens} tokens)")
        else:
            print(f"Total Generated Tokens: {total_tokens}")
        print(f"Overall TPS: {overall_tps:.2f} tokens/s")
        print(f"Decode TPS: {decode_tps:.2f} tokens/s")

        print("\n=== Extracted Reasoning Chain (<think>) ===")
        print(reasoning_text if reasoning_text else "[No separate reasoning or direct output]")

        print("\n=== Final Response Content ===")
        print(final_content)

    except Exception as e:
        print(f"Streaming verification failed: {e}")

if __name__ == "__main__":
    verify_streaming_and_thinking()
```

Typical output:
```text
Sending prompt: '请推导公式 17 × 23 并详细给出计算步骤。' to model 'ling-3.0-flash-int4'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 335.93 ms
Total Duration: 27.05 s
Decode Duration: 26.72 s
Total Generated Tokens: 1024 (Reasoning: 1024 tokens, Content: 0 tokens)
Overall TPS: 37.85 tokens/s
Decode TPS: 38.33 tokens/s

=== Extracted Reasoning Chain (<think>) ===
```

+++

#### Step 5.3: Test Function Calling and Structured Output

Pass a standard Function Calling schema to verify the model's structured tool calling capability:

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
            "description": "获取指定城市的实时天气预报与气温信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：杭州、北京、上海"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

def verify_tool_calling():
    print(f"Testing Function Calling with model 'ling-3.0-flash-int4'...")
    try:
        response = client.chat.completions.create(
            model="ling-3.0-flash-int4",
            messages=[
                {"role": "user", "content": "请帮我查一下杭州今天的天气如何？"}
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
            print("\n=== Direct Response ===")
            print(message.content)

    except Exception as e:
        print(f"Tool calling verification failed: {e}")

if __name__ == "__main__":
    verify_tool_calling()
```

Typical output:
```text
Testing Function Calling with model 'ling-3.0-flash-int4'...

=== Function Call Output Detected ===
Tool Call ID: chatcmpl-tool-8f0a2d48
Function Name: get_weather
Arguments JSON: {"city": "杭州", "unit": "celsius"}
```

+++

### Step 6: Troubleshooting & FAQ

1. **Quantization Format and Operator Support**:
   - Note: Ling-3.0-flash INT4 weights are loaded in vLLM via the `compressed-tensors/marlin` format. Prompts related to Marlin operators during startup are expected.

2. **GPU Memory Utilization Tuning**:
   - Symptom: CUDA Out of Memory occurs under high concurrency or growing context length.
   - Solution: Fine-tune `--gpu-memory-utilization` (e.g. reduce from 0.90 to 0.85) to leave adequate memory buffering for the system.

3. **Port Conflict (Port 30000 occupied)**:
   - Symptom: The server fails to start with `Address already in use`.
   - Solution: Identify and terminate the occupying process via `lsof -i :30000`, or change the service port using `--port` in startup arguments.
