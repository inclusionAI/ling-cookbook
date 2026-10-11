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

# Ling-3.0-tiny on DGX Spark (llama.cpp BF16 GGUF) Deployment Guide

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

`Ling-3.0-tiny` is a lightweight Sparse MoE language model with 7.9B total parameters and 1.3B active parameters per token.

This notebook demonstrates how to deploy Ling-3.0-tiny at full BF16 GGUF precision on `NVIDIA DGX Spark` using `llama.cpp`. The weight file takes ~16 GB and runs entirely in GPU memory on a single DGX Spark, providing high-speed, low-latency inference services.

> [!TIP]
> **Environment Recommendations**:
> - We recommend using a **Python 3.11 / 3.12** environment;
> - We recommend using **uv** to manage isolated virtual environments, ensuring clean dependency management and toolchain compatibility.

+++

### Step 1: Set Up Python Virtual Environment (uv) and Clone llama.cpp Repository

We recommend using `uv` to create an isolated virtual environment and install the OpenAI client. `llama.cpp` upstream master branch natively supports the Ling-3.0 architecture:

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0' requests
!git clone https://github.com/ggerganov/llama.cpp.git
```

Typical execution output:
```text
Using CPython 3.11 interpreter at: /usr/bin/python3.11
Creating virtualenv at: .venv
Cloning into 'llama.cpp'...
remote: Enumerating objects: 45210, done.
remote: Counting objects: 100% (210/210), done.
```

+++

### Step 2: Build llama.cpp (with CUDA Acceleration)

Compile `llama.cpp` upstream master on DGX Spark with CUDA hardware acceleration enabled:

```{code-cell}
!cd llama.cpp && cmake -B build -DGGML_CUDA=ON . && cmake --build build --parallel 8
```

Typical execution output:
```text
-- The CUDA compiler identification is NVIDIA 13.0.88
-- Building with CUDA architecture: native (sm_121)
[100%] Built target llama-server
```

+++

### Step 3: Download Official Ling-3.0-tiny Model Weights (Fast Shell Download)

Download official Safetensors weights to `~/models/Ling-3.0-tiny` using the `modelscope` CLI (~15.8 GB total, typically completing in 1-2 minutes):

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
```

+++

### Step 4: Convert Model to Full Precision (BF16) GGUF Format

Convert the model using `convert_hf_to_gguf.py`. Due to the compact size of the Tiny model, **the converted BF16 GGUF file can be served directly for full precision inference without requiring secondary quantization with `llama-quantize`**:

```{code-cell}
!source .venv/bin/activate && uv pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt --quiet
!source .venv/bin/activate && python3 ./llama.cpp/convert_hf_to_gguf.py ~/models/Ling-3.0-tiny \
  --outfile ~/models/Ling-3.0-tiny-bf16.gguf \
  --outtype bf16 --model-name Ling-3.0-tiny
```

Typical execution output:
```text
INFO:hf-to-gguf:Loading model: Ling-3.0-tiny
INFO:hf-to-gguf:Set model parameters
INFO:hf-to-gguf:Writing tensors to /home/squall/models/Ling-3.0-tiny-bf16.gguf
INFO:hf-to-gguf:Done. Output file: /home/squall/models/Ling-3.0-tiny-bf16.gguf (15.82 GiB)
```

+++

### Step 5: Launch Model Inference Service with llama-server

Launch `llama-server` to provide an OpenAI-compatible HTTP interface. Parameter explanations:

- `-m ~/models/Ling-3.0-tiny-bf16.gguf`: Loads full precision BF16 GGUF model weights
- `--alias Ling-3.0-tiny-bf16`: Assigns model alias for downstream client requests
- `-ngl all`: Offloads all layers to GPU (memory footprint ~18 GB)
- `-fa on`: Enables FlashAttention acceleration
- `-c 131072`: Enables native 128K context window
- `-cb -np 4`: Enables continuous batching with 4 concurrent slots
- `--port 9102`: Uses port `9102` to avoid conflicts with default 8080 port
- `--api-key sk-ling-cookbook-test`: Configures authentication API key

> [!WARNING]
> **Important: Foreground Execution and Blocking**
> `llama-server` runs in the foreground. Executing this cell directly will block the notebook.
> We recommend running the launch command in a **separate terminal window**. Once the service starts, return here to run the remaining verification cells.

```{code-cell}
!./llama.cpp/build/bin/llama-server \
  -m ~/models/Ling-3.0-tiny-bf16.gguf \
  --alias Ling-3.0-tiny-bf16 \
  -ngl all -fa on -c 131072 -cb -np 4 \
  --host 0.0.0.0 --port 9102 --api-key sk-ling-cookbook-test
```

Typical server startup logs:
```text
llama_server: HTTP server listening on 0.0.0.0:9102
llama_server: model loaded successfully, n_ctx = 131072, offload = 100% (GPU)
```

+++

### Step 6: Verify Deployment and Test API Calls

Once the service is active, it can be called using standard LLM clients (run `pip install openai requests` if not yet installed).

The verification suite includes:
1. **Health Check** - Confirm model status and endpoint availability.
2. **Streaming Reasoning** - Test `<think>` reasoning chain generation and measure TTFT and Decode TPS.
3. **Tool Calling** - Test Function Calling with structured outputs.

+++

#### Step 6.1: Connectivity & Model Health Check

Query `GET /v1/models` to inspect the served model:

```{code-cell}
import urllib.request

url = "http://localhost:9102/v1/models"
try:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer sk-ling-cookbook-test"})
    with urllib.request.urlopen(req) as response:
        print(f"Health check status: {response.getcode()}")
        print(f"Models response: {response.read().decode('utf-8')}")
except Exception as e:
    print(f"Health check error: {e}")
```

Typical output:
```text
Health check status: 200
Models response: {"models":[{"name":"Ling-3.0-tiny-bf16","model":"Ling-3.0-tiny-bf16",...}]}
```

+++

#### Step 6.2: Streaming Inference, Reasoning, and Benchmark (TTFT / TPS)

Use the OpenAI Python SDK with `enable_thinking: True` to extract `<think>` reasoning chain tokens and measure TTFT and Decode TPS:

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9102/v1",
    api_key="sk-ling-cookbook-test"
)

prompt = "Derive 25 × 48 and show the calculation steps in detail."
print(f"Sending prompt: '{prompt}' to model 'Ling-3.0-tiny-bf16'...")

start_time = time.time()
first_token_time = None
total_tokens = 0
reasoning_text = ""
content_text = ""

try:
    response = client.chat.completions.create(
        model="Ling-3.0-tiny-bf16",
        messages=[{"role": "user", "content": prompt}],
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
            print(delta.content, end="", flush=True)
            total_tokens += 1

    end_time = time.time()
    ttft = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
    decode_duration = end_time - first_token_time if first_token_time else 0.001
    tps = total_tokens / decode_duration

    print("")
    print("=== Latency & Throughput Metrics ===")
    print(f"TTFT (Time to First Token): {ttft:.2f} ms")
    print(f"Decode TPS (Tokens/s): {tps:.2f} t/s")
    print(f"Total Generated Tokens: {total_tokens}")
    print(f"Total Duration: {end_time - start_time:.2f} s")

    if reasoning_text:
        print("")
        print("=== Extracted Reasoning Chain (<think>) ===")
        print(reasoning_text)

except Exception as e:
    print(f"Streaming verification failed: {e}")
```

Typical output:
```text
=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 58.39 ms
Decode TPS (Tokens/s): 85.21 t/s
Total Generated Tokens: 1315
Total Duration: 15.49 s

=== Extracted Reasoning Chain (<think>) ===
```

+++

#### Step 6.3: Tool Calling and Structured Outputs Test

Verify Function Calling with a standard weather tool schema:

```{code-cell}
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

print("Testing Function Calling / Tool Use with model 'Ling-3.0-tiny-bf16'...")
try:
    response = client.chat.completions.create(
        model="Ling-3.0-tiny-bf16",
        messages=[{"role": "user", "content": "What is the weather in Hangzhou today?"}],
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
```

Typical output:
```text
Testing Function Calling / Tool Use with model 'Ling-3.0-tiny-bf16'...

=== Function Call Output Detected ===
Tool Call ID: pa6MtfWs1LLBkrFJpXaFnjlXKxU7oTi2
Function Name: get_weather
Arguments JSON: {"city":"Hangzhou"}
```

+++

### Step 7: Troubleshooting & FAQ

1. **Missing GGUF Conversion Dependencies**:
   - Symptom: `convert_hf_to_gguf.py` reports missing Python modules.
   - Solution: Install required packages via `uv pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt`.

2. **Port Conflict (`Port 9102 occupied`)**:
   - Symptom: `llama-server` fails to bind address.
   - Solution: Terminate the occupying process using `lsof -i :9102`, or select an alternative port with `--port`.
