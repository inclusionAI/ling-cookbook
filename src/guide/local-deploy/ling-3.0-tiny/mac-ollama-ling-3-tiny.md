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

# Ling-3.0-tiny on Apple Silicon Mac (Ollama) Deployment Guide

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

`Ling-3.0-tiny` is a 7.9B lightweight Sparse MoE language model in the Ling model family with 1.3B active parameters per token. It natively supports a 128K long context and delivers strong on-device agent capabilities, tool calling, and deep reasoning under low compute and memory requirements.

This guide demonstrates how to deploy `Ling-3.0-tiny` on Apple Silicon Mac using Ollama.

---

### Hardware & Quantization Matrix

Select the profile that best matches your Mac device's unified memory capacity:

| Deployment Profile | Quantization Type | Model Size | Recommended Memory (8K Context) | Supported Mac Hardware |
| :--- | :--- | :---: | :---: | :--- |
| BF16 (Full Precision) | 16-bit GGUF | ~15.80 GB | ≥ 24 GB - 32 GB | MacBook Pro 36GB / 48GB+ |
| Q8_0 (High Precision) | 8-bit GGUF | ~8.30 GB | ≥ 16 GB - 24 GB | MacBook Pro 18GB / 24GB |
| Q4_K_M (Quantized) | 4-bit GGUF | ~4.30 GB | ≥ 8 GB | MacBook Air / Pro 16GB |

+++

### Step 1: Prepare Environment and Install Ollama

#### Step 1.1: Install Ollama and uv via Homebrew

We recommend installing Ollama and the Python package manager `uv` via Homebrew:

```{code-cell}
# Install Ollama and uv (skip if already installed)
!brew install ollama uv
```

Typical installation output:
```text
==> Would install 1 formula:
ollama
==> Would install 2 dependencies for ollama:
mlx
mlx-c
==> Do you want to proceed with the installation? [y/n]
==> Fetching downloads for: ollama

==> Installing ollama
==> Pouring ollama--0.33.3.arm64_tahoe.bottle.tar.gz
🍺  /opt/homebrew/Cellar/ollama/0.33.3: 16 files, 52.9MB
==> Caveats
==> ollama
To start ollama now and restart at login:
  brew services start ollama
Or, if you don't want/need a background service you can just run:
  OLLAMA_FLASH_ATTENTION="1" OLLAMA_KV_CACHE_TYPE="q8_0" /opt/homebrew/opt/ollama/bin/ollama serve
```

#### Step 1.2: Create Python Test Virtual Environment and Install Dependencies

This virtual environment is used for API verification and speed benchmarking in subsequent steps:

```{code-cell}
# Create an isolated virtual environment with uv and install OpenAI SDK
!uv venv .venv --python 3.12
!uv pip install --upgrade 'openai>=1.52.0'
```

Typical execution output:
```text
❯ uv venv .venv --python 3.12
Using CPython 3.12.14
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate

❯ source .venv/bin/activate

❯ uv pip install --upgrade 'openai>=1.52.0'
Resolved 14 packages in 559ms
Prepared 14 packages in 358ms
Installed 14 packages in 23ms
 + annotated-types==0.8.0
 + anyio==4.15.1
 + h11==0.16.0
 + httpcore2==2.12.0
 + httpx2==2.12.0
 + idna==3.19
 + jiter==0.16.0
 + openai==3.8.0
 + pydantic==2.13.5
 + pydantic-core==2.46.5
 + sniffio==1.3.1
 + truststore==0.10.4
 + typing-extensions==4.16.0
 + typing-inspection==0.4.4
```

#### Step 1.3: Launch Ollama Service and Verify Health Status

Ollama needs to run as a persistent service to listen for API requests (default port `11434`).

Open a separate terminal window and run the following command to start the service (enabling Flash Attention and KV cache quantization is recommended):

```bash
OLLAMA_FLASH_ATTENTION="1" OLLAMA_KV_CACHE_TYPE="q8_0" ollama serve
```

Once started, the terminal displays logs similar to:
```text
time=2026-09-07T14:41:36.482+08:00 level=INFO source=routes.go:2012 msg="Listening on 127.0.0.1:11434 (version 0.33.3)"
```

After the service is ready, verify connectivity and version in your working terminal:

```{code-cell}
# Check whether local Ollama service is ready
!curl -s http://127.0.0.1:11434/api/version
```

Typical output:
```json
{"version":"0.33.3"}
```

+++

### Step 2: Deploy Model

Ollama natively supports pulling and running GGUF repositories from Hugging Face. Select the profile matching your hardware:

#### Option A: Full Precision (BF16)

Original precision weights, recommended for devices with >24GB memory:

> [!NOTE]
> This step downloads ~15 GB of model weights from Hugging Face. This operation may take several minutes. Please be patient.

```{code-cell}
# Pull and run the official BF16 GGUF model
!ollama run hf.co/inclusionAI/Ling-3.0-tiny-GGUF:bf16 "Please introduce yourself in one sentence."
```

Typical pull and run output:
```text
pulling manifest 
pulling b11d4a45d3ad: 100% ▕███████████████████████████████████████████████████████████████████████▏  15 GB                         
verifying sha256 digest 
writing manifest 
success 
<think>
The user asks for a one-sentence self-introduction. I should summarize my identity and core role concisely.
</think>I am Ling, an AI assistant developed by Ant Group designed to provide information processing and intelligent assistance through natural language interaction.
```

#### Option B: High Precision Quantization (Q8_0)

Recommended for devices with 18GB / 24GB memory:

```{code-cell}
# Pull and run the Q8_0 profile
!ollama run hf.co/inclusionAI/Ling-3.0-tiny-GGUF:Q8_0 "Please introduce yourself in one sentence."
```

Typical pull and run output:
```text
pulling manifest 
pulling 9299a9e5cbc5: 100% ▕███████████████████████████████████████████████████████████████████████▏ 8.4 GB                         
pulling 62edd696268b: 100% ▕███████████████████████████████████████████████████████████████████████▏  223 B                         
pulling a254ca5329e8: 100% ▕███████████████████████████████████████████████████████████████████████▏   65 B                         
verifying sha256 digest 
writing manifest 
success 
<think>
Summarize identity and core capabilities cleanly in one sentence.
</think>I am Ling, a general-purpose language model developed by Ant Group adept at multiple language tasks and deep reasoning.
```

#### Option C: Quantization (Q4_K_M)

Recommended for devices with 8GB / 16GB memory:

```{code-cell}
# Pull and run the Q4_K_M profile
!ollama run hf.co/inclusionAI/Ling-3.0-tiny-GGUF:Q4_K_M "Please introduce yourself in one sentence."
```

Typical pull and run output:
```text
pulling manifest 
pulling 246d67d45f5b: 100% ▕███████████████████████████████████████████████████████████████████████▏ 4.8 GB                         
pulling 62edd696268b: 100% ▕███████████████████████████████████████████████████████████████████████▏  223 B                         
pulling a254ca5329e8: 100% ▕███████████████████████████████████████████████████████████████████████▏   65 B                         
verifying sha256 digest 
writing manifest 
success 
<think>
Introduce myself concisely as Ling.
</think>I am Ling, a general-purpose language model developed by Ant Group aimed at supporting a wide range of tasks.
```

+++

### Step 3: Verify Deployment and Test API Calls

Ollama natively conforms to the OpenAI API specification. The following scripts verify basic chat and tool calling capabilities:

#### Step 3.1: Basic Chat and Reasoning Control (Hello World)

`Ling-3.0-tiny` features native reasoning chain capabilities. Prompts and system messages can steer whether reasoning is expanded:

```{code-cell}
from openai import OpenAI

# Initialize client connecting to local Ollama service on port 11434
client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

# Replace with the profile you pulled (:bf16, :Q8_0, or :Q4_K_M)
MODEL_NAME = "hf.co/inclusionAI/Ling-3.0-tiny-GGUF:bf16"

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "You are a helpful and concise on-device AI assistant."},
        {"role": "user", "content": "Compute 17 × 23 and briefly explain the steps."}
    ],
    temperature=0.1
)

print("=== Model Output ===")
print(response.choices[0].message.content)
```

Typical output:
```text
=== Model Output ===
To compute 17 × 23:
17 × 20 = 340
17 × 3 = 51
340 + 51 = 391
Therefore, 17 × 23 = 391.
```

#### Step 3.2: Verify On-Device Tool Calling

`Ling-3.0-tiny` is well suited for agent workloads. We define a weather inquiry tool to verify Function Calling:

```{code-cell}
import json

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
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

messages = [
    {"role": "user", "content": "What is the current weather in Hangzhou?"}
]

tool_response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    tools=tools,
    temperature=0.0
)

message = tool_response.choices[0].message
if message.tool_calls:
    print("✅ Tool call check passed! Model initiated a tool call:")
    for call in message.tool_calls:
        print(f"  - Function name: {call.function.name}")
        print(f"  - Arguments: {call.function.arguments}")
else:
    print("❌ Tool call not triggered. Raw output:", message.content)
```

Typical output:
```text
✅ Tool call check passed! Model initiated a tool call:
  - Function name: get_current_weather
  - Arguments: {"city":"Hangzhou"}
```

+++

### Step 4: Benchmark Inference Speed

Call Ollama's native `/api/generate` endpoint to measure generation throughput (Decode TPS) and end-to-end latency:

```{code-cell}
import urllib.request
import json
import time

prompt_data = {
    "model": MODEL_NAME,
    "prompt": "Explain the advantages and computational complexity differences of hybrid attention mechanisms (such as linear attention combined with MLA) in long context LLM inference in about 150 words.",
    "stream": False,
    "options": {
        "temperature": 0.0,
        "num_predict": 256
    }
}

req = urllib.request.Request(
    "http://127.0.0.1:11434/api/generate",
    data=json.dumps(prompt_data).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

start_time = time.time()
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode("utf-8"))
elapsed_total = time.time() - start_time

# Extract native Ollama metrics
eval_count = result.get("eval_count", 0)                  # Generated tokens count
eval_duration = result.get("eval_duration", 1) / 1e9       # Generation time in seconds

decode_tps = eval_count / eval_duration if eval_duration > 0 else 0

print("========================================")
print(f" Ling-3.0-tiny on Mac (Ollama) Benchmark")
print("========================================")
print(f"  - Model profile: {MODEL_NAME}")
print(f"  - Decode throughput: {decode_tps:.2f} tokens/s ({eval_count} tokens / {eval_duration:.3f}s)")
print(f"  - End-to-end latency: {elapsed_total:.3f}s")
print("========================================")
```

Typical execution output (measured on Apple Silicon Mac):
```text
========================================
 Ling-3.0-tiny on Mac (Ollama) Benchmark
========================================
  - Model profile: hf.co/inclusionAI/Ling-3.0-tiny-GGUF:bf16
  - Decode throughput: 72.63 tokens/s (358 tokens / 4.93s)
  - End-to-end latency: 4.93s
========================================
```

Verified benchmark on Apple Silicon Mac:

| Hardware | Profile | Context Length | Decode Throughput (TPS) | Peak Memory |
| :--- | :--- | :-: | :-: | :-: |
| M5 Pro (48GB) | BF16 (Full Precision) | 32K | ~ 72.6 tokens/s | ~ 14.9 GiB |
| M5 Pro (48GB) | Q8_0 (High Precision) | 32K | ~ 105.4 tokens/s | ~ 6.0 GiB |
| M5 Pro (48GB) | Q4_K_M | 32K | ~ 127.3 tokens/s | ~ 4.9 GiB |

+++

### Step 5: Troubleshooting & FAQ

1. Connection refused: `Failed to connect to 127.0.0.1 port 11434`
   - Cause: The Ollama service is not running in the background.
   - Solution: In an independent terminal window, start the service with `OLLAMA_FLASH_ATTENTION="1" OLLAMA_KV_CACHE_TYPE="q8_0" ollama serve`.

2. Specifying and controlling context length (`num_ctx`)
   `Ling-3.0-tiny` natively supports a 128K context. Ollama's default context setting is conservative (often 2048 or 4096). You can adjust it via:
   - Method A (Custom Modelfile, recommended):
     Create a `Modelfile` specifying the target context length (e.g., 32K):
     ```dockerfile
     FROM hf.co/inclusionAI/Ling-3.0-tiny-GGUF:bf16
     PARAMETER num_ctx 32768
     ```
     Build and register the model: `ollama create ling-3-tiny-32k -f Modelfile`.
   - Method B (OpenAI SDK runtime option):
     Pass `extra_body` in API requests:
     ```python
     response = client.chat.completions.create(
         model=MODEL_NAME,
         messages=[...],
         extra_body={"options": {"num_ctx": 32768}}
     )
     ```
   - Method C (Interactive CLI):
     In the `ollama run` prompt, execute `/set parameter num_ctx 32768`.

3. Performance slowdown due to memory swapping
   - Cause: Context window set excessively large (e.g., 64K / 128K) or multiple heavy applications competing for unified memory.
   - Solution: For 8GB / 16GB devices, use the `Q4_K_M` profile and keep context within 8192 or 16384 tokens to preserve system responsiveness.
