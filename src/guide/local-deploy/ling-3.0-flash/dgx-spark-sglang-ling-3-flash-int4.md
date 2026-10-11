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

# Deployment Guide: Ling-3.0-flash on DGX Spark (SGLang INT4 W4A16)

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

This notebook demonstrates how to deploy Ling-3.0-flash INT4 (W4A16) quantized weights using `SGLang` built from source on a `Single NVIDIA DGX Spark` (GB10 with 121 GB unified memory), and enable `NextN` speculative decoding for acceleration. The quantized model weights occupy only ~72 GB. Combined with the NextN algorithm, this significantly boosts decode throughput and reduces time-to-first-token on a single DGX Spark.

> [!TIP]
> **Environment Recommendations**:
> - **Python 3.11 / 3.12** is recommended.
> - Using **uv** to create an isolated Python virtual environment is recommended to ensure dependency isolation and operator compatibility.

+++

### Step 1: Set Up Python Virtual Environment (uv) and Clone SGLang Repository

Create an isolated Python virtual environment using `uv` and install the OpenAI client library. Clone the official SGLang repository (Ling-3.0-flash INT4 Marlin MoE operators are natively supported on main):

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
!git clone https://github.com/sgl-project/sglang.git
```

Typical output:
```text
Using CPython 3.11 interpreter at: /usr/bin/python3.11
Creating virtualenv at: .venv
Cloning into 'sglang'...
remote: Enumerating objects: 38200, done.
Resolving deltas: 100% (28500/28500), done.
```

+++

### Step 2: Install SGLang and Runtime Dependencies from Source

Install the branch code and full dependencies (`[all]`) in editable mode within the virtual environment. Set `MAX_JOBS=4` to avoid running out of memory during multi-core concurrent compilation:

```{code-cell}
!source .venv/bin/activate && MAX_JOBS=4 uv pip install -e "./sglang/python[all]"
```

Typical output:
```text
Successfully built sglang cuda-tile xatlas
Installing collected packages: ....
```

+++

### Step 3: Download Ling-3.0-flash INT4 Model Weights

Pre-quantized INT4 weights are officially provided:
- [Ling-3.0-flash-int4 on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-int4)
- [Ling-3.0-flash-int4 on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-int4)

If you are located in China, using the [ModelScope CLI](https://github.com/modelscope/modelscope/blob/master/README.md) is recommended; alternatively, you can use the [Hugging Face CLI](https://huggingface.co/docs/hub/agents-cli) to download weights to a local directory:

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-int4 --local-dir ~/models/Ling-3.0-flash-int4
```

Typical output:
```text
Downloading [model-00024-of-00024.safetensors]: 100%|█| 3.10G/3.10G [01:27<00:00
Processing 35 items: 100%|███████████████████| 35.0/35.0 [11:18<00:00, 19.4s/it]
Successfully downloaded Ling-3.0-flash-int4 to ~/models/Ling-3.0-flash-int4
```

+++

### Step 4: Launch SGLang HTTP Inference Service (with NextN Speculative Decoding)

Start the SGLang Server providing an OpenAI-compatible HTTP interface. Key deployment parameters include:
- `SGLANG_ENABLE_SPEC_V2=1`: Enables the high-performance Speculative V2 scheduling core.
- `--speculative-algorithm NEXTN`: Enables the NextN speculative decoding algorithm to accelerate decoding.
- `--max-mamba-cache-size 32`: Configures the Mamba cache buffer size for speculative decoding.
- `--mem-fraction-static 0.75`: Pre-allocates 75% GPU memory, reserving sufficient headroom for KV cache and speculative branches.
- `--tool-call-parser ling3 --reasoning-parser ling3`: Enables Ling-3.0 dedicated parsers for tool calling and reasoning chain extraction.
- `--json-model-override-args`: Configures YaRN RoPE scaling for native 256K long context window support.

⚠️ Note:

`sglang.launch_server` runs as a persistent foreground process. If you run the cell below directly inside the notebook, Jupyter will block and cannot execute subsequent cells. It is recommended to run the startup command in a separate terminal session. Once the service is ready, proceed with the verification steps in Jupyter.

```{code-cell}
!source .venv/bin/activate && \
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
python3 -m sglang.launch_server \
  --model-path ~/models/Ling-3.0-flash-int4 \
  --served-model-name ling-v3-flash-int4 \
  --host 0.0.0.0 --port 30000 \
  --api-key sk-ling-cookbook-test \
  --mem-fraction-static 0.75 --max-running-requests 8 \
  --tp-size 1 --chunked-prefill-size 8192 --allow-auto-truncate \
  --tool-call-parser ling3 --reasoning-parser ling3 --context-length 262144 \
  --speculative-algorithm NEXTN --max-mamba-cache-size 32 --enable-fp32-lm-head \
  --disable-shared-experts-fusion \
  --json-model-override-args '{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}'
```

Typical output when server is ready:
```text
Triton kernel 'chunk_kda_fwd_kernel_inter_solve_fused' took 2.44 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
Prefill batch, #new-seq: 1, #new-token: 6, #cached-token: 0, full token usage: 0.00, mamba usage: 0.09, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 0.25
INFO:     127.0.0.1:58576 - "POST /generate HTTP/1.1" 200 OK
The server is fired up and ready to roll!
```

+++

### Step 5: Verify Deployment and Test API Calls

Once the service is running, you can connect to it using any LLM client (run `pip install openai` if not already installed). The following examples demonstrate:

1. Health Check - Verify model loading status and endpoint connectivity.
2. Streaming Reasoning Verification - Test streaming `<think>` reasoning chain generation and measure TTFT and TPS metrics under speculative decoding acceleration.
3. Tool Calling Verification - Test structured Function Calling support.

+++

#### Step 5.1: Check Service Health and Connectivity

Query `GET /v1/models` to verify the running model information:

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
```
Health check status: 200
Models response: {"object":"list","data":[{"id":"ling-v3-flash-int4","object":"model","created":1786799113,"owned_by":"sglang","root":"ling-v3-flash-int4","parent":null,"max_model_len":262144}]}
```

+++

#### Step 5.2: Test Streaming Inference, Reasoning, and Throughput Metrics

Use the OpenAI Python SDK to call the API, extract the `<think>` reasoning chain via `enable_thinking: True`, and measure TTFT and Decode TPS in real-time under speculative decoding acceleration (the very first request compiles JIT operators and may take tens of seconds; subsequent requests will return to normal latency):

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-ling-cookbook-test"
)

def verify_streaming_and_thinking():
    prompt = "请推导公式 17 × 23 并详细给出计算步骤。"
    print(f"Sending prompt: '{prompt}' to model 'ling-v3-flash-int4'...")
    
    start_time = time.time()
    first_token_time = None
    chunk_count = 0
    exact_completion_tokens = None
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model="ling-v3-flash-int4",
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
Sending prompt: '请推导公式 17 × 23 并详细给出计算步骤。' to model 'ling-v3-flash-int4'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 135.51 ms
Decode TPS (Tokens/s): 42.23 t/s
Total Generated Tokens: 726 (Exact tokens from usage)
Total Duration: 17.33 s

=== Extracted Reasoning Chain (<think>) ===
...
```

+++

#### Step 5.3: Test Function Calling and Structured Output

Pass a standard Function Calling schema (weather query example) to verify the model's structured tool calling capability:

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
    print(f"Testing Function Calling / Tool Use with model 'ling-v3-flash-int4'...")
    try:
        response = client.chat.completions.create(
            model="ling-v3-flash-int4",
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
            print("\n=== Direct Response (No Tool Call Triggered) ===")
            print(message.content)

    except Exception as e:
        print(f"Tool calling verification failed: {e}")

if __name__ == "__main__":
    verify_tool_calling()
```

Typical output:
```text
Testing Function Calling / Tool Use with model 'ling-v3-flash-int4'...

=== Function Call Output Detected ===
Tool Call ID: call_87804adb3c954cb4b5dcc266
Function Name: get_weather
Arguments JSON: {"city": "杭州", "unit": "celsius"}
```

+++

### Step 6: Troubleshooting & FAQ

1. **JIT Kernel Compilation Delay on First Request**:
   - Symptom: The initial inference request has a long TTFT (potentially tens of seconds), with logs showing `Triton kernel took X.XX s to compile`.
   - Explanation: This is expected behavior during Triton JIT compilation. Subsequent requests will reuse the cached compiled kernels and run at normal speed.

2. **Out-of-Memory (OOM) During Source Build**:
   - Symptom: The process is terminated by the OS when executing `MAX_JOBS=4 uv pip install -e "./sglang/python[all]"` while compiling C++/CUDA extensions.
   - Solution: Limit parallel compilation jobs using `MAX_JOBS=4` or `MAX_JOBS=2`.

3. **Port Conflict (Port 30000 occupied)**:
   - Symptom: The server fails to start with `Address already in use`.
   - Solution: Identify and terminate the occupying process via `lsof -i :30000`, or modify `--port` in the startup arguments.
