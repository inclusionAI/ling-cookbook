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

# Deployment Guide: Ling-3.0-flash on DGX Spark (llama.cpp Q4_K_M GGUF)

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

This notebook demonstrates how to deploy the 4-bit quantized (Q4_K_M) version of Ling-3.0-flash on an `NVIDIA DGX Spark` using `llama.cpp`. After quantization, the model weights occupy ~60.5 GB, providing fast and stable concurrent serving on a single DGX Spark.

> [!TIP]
> **Environment Recommendations**:
> - **Python 3.11 / 3.12** is recommended.
> - Using **uv** to create an isolated Python virtual environment is recommended to ensure dependency isolation and toolchain compatibility.

+++

### Step 1: Set Up Python Virtual Environment (uv) and Clone llama.cpp Repository

Create an isolated virtual environment using `uv` and install the OpenAI Python library for testing. Upstream `llama.cpp` has merged support for the Ling-3.0 architecture, so cloning the master branch directly is sufficient:

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
!git clone https://github.com/ggerganov/llama.cpp.git
```

Typical output:

```text
Using CPython 3.11 interpreter at: /usr/bin/python3.11
Creating virtualenv at: .venv
Cloning into 'llama.cpp'...
remote: Enumerating objects: 45210, done.
remote: Counting objects: 100% (210/210), done.
```

+++

### Step 2: Build llama.cpp

Build `llama.cpp` with CUDA acceleration enabled on DGX Spark:

```{code-cell}
!cd llama.cpp && cmake -B build -DGGML_CUDA=ON . && cmake --build build --parallel 8
```

Typical output:
```text
-- The CXX compiler identification is GNU 11.4.0
-- The CUDA compiler identification is NVIDIA 12.8.55
-- Detecting CUDA compiler ABI info - done
-- Building with CUDA architecture: native
[ 50%] Building CXX object CMakeFiles/llama.dir/src/llama.cpp.o
[100%] Built target llama-server
[100%] Built target llama-quantize
```

+++

### Step 3: Download Ling-3.0-flash Model Weights

First, obtain the model's original Safetensors weights. If you are located in China, using the [ModelScope CLI](https://github.com/modelscope/modelscope/blob/master/README.md) is recommended; alternatively, you can use the [Hugging Face CLI](https://huggingface.co/docs/hub/agents-cli) to download weights.

The model repository homepages on the two platforms are [Ling-3.0-flash on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash) and [Ling-3.0-flash on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash).

Below is an example using ModelScope:

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash --local-dir ~/models/Ling-3.0-flash
```

Typical output:
```text
Downloading [config.json, model.safetensors.index.json, ...]
Downloading shard 1/24: 100%|██████████| 4.98G/4.98G [00:15<00:00, 332MB/s]
...
Downloading shard 24/24: 100%|██████████| 3.12G/3.12G [00:09<00:00, 346MB/s]
Successfully downloaded Ling-3.0-flash to ~/models/Ling-3.0-flash
```

+++

### Step 4: Convert Model to Full Precision GGUF Format

`llama.cpp` requires models in GGUF format. Install conversion dependencies and use `convert_hf_to_gguf.py` to convert Safetensors weights to full-precision BF16 GGUF format:

```{code-cell}
!source .venv/bin/activate && uv pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
!source .venv/bin/activate && python3 llama.cpp/convert_hf_to_gguf.py ~/models/Ling-3.0-flash \
  --outfile ~/models/Ling-3.0-flash-bf16.gguf \
  --outtype bf16 --model-name Ling-3.0-flash
```

Typical output:
```text
INFO:hf-to-gguf:Loading model: Ling-3.0-flash
INFO:hf-to-gguf:Set model parameters
INFO:hf-to-gguf:Writing tensors to /home/squall/models/Ling-3.0-flash-bf16.gguf
INFO:hf-to-gguf:Done writing 124B model tensors (bf16).
```

+++

### Step 5: Quantize Model to Q4_K_M GGUF Format

The full-precision model cannot run on a single DGX Spark. Use the compiled `llama-quantize` tool to quantize the BF16 GGUF to 4-bit (Q4_K_M) format, reducing model size from ~248 GB to ~60.5 GB to fit on a single DGX Spark:

```{code-cell}
!./llama.cpp/build/bin/llama-quantize ~/models/Ling-3.0-flash-bf16.gguf ~/models/Ling-3.0-flash-Q4_K_M.gguf Q4_K_M
```

Typical output:
```text
main: build = 26608
main: quantizing '/home/squall/models/Ling-3.0-flash-bf16.gguf' to '/home/squall/models/Ling-3.0-flash-Q4_K_M.gguf' as Q4_K_M
[   1/ 780]              blk.0.attn_q.weight - [ 2560,  2560,     1], type =  f16, size =    12.50 MB -> Q4_K, size =     3.44 MB
...
[ 780/ 780]                     output.weight - [ 2560, 157184,     1], type =  f16, size =   767.50 MB -> Q6_K, size =   300.00 MB
main: model size  = 238410.20 MB -> 60512.44 MB
main: quantize time = 184512.20 ms
```

+++

### Step 6: Launch Model Inference Service with llama-server

Start `llama-server` to provide an OpenAI-compatible HTTP interface. Key deployment parameters include:

- `-ngl all`: Offload all model layers to GPU.
- `-fa on`: Enable FlashAttention acceleration.
- `-c 262144`: Enable 256K context window.
- `-np 4`: Support 4 concurrent slots.

By default, llama-server listens on port 8080. Here we use port `9102`.

⚠️ Note:

`llama-server` runs in foreground persistent mode. If run directly inside a Jupyter Notebook cell, Jupyter will block and cannot execute subsequent cells. It is recommended to run the launch command in an independent terminal session. Once the service is ready, proceed with the verification steps in Jupyter.

```{code-cell}
!./llama.cpp/build/bin/llama-server \
  -m ~/models/Ling-3.0-flash-Q4_K_M.gguf \
  --alias Ling-3.0-flash-Q4_K_M \
  --ngl all -fa on -c 262144 -cb -np 4 \
  --host 0.0.0.0 --port 9102 --api-key sk-ling-cookbook-test
```

Typical output when server is ready:
```text
llama_server: HTTP server listening on 0.0.0.0:9102
llama_server: model loaded successfully, n_ctx = 262144, offload = 100% (GPU)
```

+++

### Step 7: Verify Deployment and Test API Calls

Once the service is started, connect to it using any LLM client. The following examples demonstrate:

1. Health Check - Verify model loading status and endpoint connectivity.
2. Streaming Reasoning Verification - Test streaming `<think>` generation and measure TTFT and TPS metrics.
3. Tool Calling Verification - Test structured Function Calling support.

+++

#### Step 7.1: Check Service Health and Connectivity

Query `GET /v1/models` to verify the running model information:

```{code-cell}
import urllib.request

url = "http://localhost:9102/v1/models"
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

```
Health check status: 200
Models response: {"models":[{"name":"Ling-3.0-flash-Q4_K_M","model":"Ling-3.0-flash-Q4_K_M","modified_at":"","size":"","digest":"","type":"model","description":"","tags":[""],"capabilities":["completion"],"parameters":"","details":{"parent_model":"","format":"gguf","family":"","families":[""],"parameter_size":"","quantization_level":""}}],"object":"list","data":[{"id":"Ling-3.0-flash-Q4_K_M","aliases":["Ling-3.0-flash-Q4_K_M"],"tags":[],"object":"model","created":1786790315,"owned_by":"llamacpp","meta":{"vocab_type":2,"n_vocab":157184,"n_ctx":65536,"n_ctx_train":262144,"n_embd":2560,"n_params":127486405600,"size":77003601792,"ftype":"Q4_K - Medium"}}]}
```

+++

#### Step 7.2: Test Streaming Inference, Reasoning, and Throughput Metrics

Use the OpenAI Python SDK to call the API, extract the `<think>` reasoning chain via `enable_thinking: True`, and record TTFT and TPS metrics:

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9102/v1",
    api_key="sk-ling-cookbook-test"
)

def verify_streaming_and_thinking():
    prompt = "请推导公式 17 × 23 并详细给出计算步骤。"
    print(f"Sending prompt: '{prompt}' to model 'Ling-3.0-flash-Q4_K_M'...")

    start_time = time.time()
    first_token_time = None
    total_tokens = 0
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model="Ling-3.0-flash-Q4_K_M",
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

Typical output:

```
Sending prompt: '请推导公式 17 × 23 并详细给出计算步骤。' to model 'Ling-3.0-flash-Q4_K_M'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 667.49 ms
Decode TPS (Tokens/s): 46.19 t/s
Total Generated Tokens: 614
Total Duration: 13.96 s
...
```

+++

#### Step 7.3: Test Function Calling and Structured Output

Pass a standard Function Calling schema (weather query example) to verify structured tool calling capability:

```{code-cell}
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9102/v1",
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
    print(f"Testing Function Calling / Tool Use with model 'Ling-3.0-flash-Q4_K_M'...")
    try:
        response = client.chat.completions.create(
            model="Ling-3.0-flash-Q4_K_M",
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

```
Testing Function Calling / Tool Use with model 'Ling-3.0-flash-Q4_K_M'...

=== Function Call Output Detected ===
Tool Call ID: gpGiPyVIVoeg7Y4ZWyqKgjfFacOVCAeC
Function Name: get_weather
Arguments JSON: {"city":"杭州"}
```

+++

### Step 8: Troubleshooting & FAQ

1. **GGUF Conversion Missing Dependency Errors**:
   - Symptom: Running `convert_hf_to_gguf.py` raises `ModuleNotFoundError`.
   - Solution: Ensure that all conversion dependencies are installed in the virtual environment via `uv pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt` (such as `torch`, `sentencepiece`, etc.).

2. **Context Memory Allocation Under High Concurrency**:
   - Symptom: Specifying an ultra-large context window (e.g. `-c 262144`) with multiple parallel slots (e.g. `-np 4`) leads to GPU memory exhaustion.
   - Solution: Reduce the concurrency slot count to `-np 2` or lower the context length to `-c 131072`.

3. **Port Conflict (Port 9102 occupied)**:
   - Symptom: `llama-server` fails to start because the port is already in use.
   - Solution: Identify and terminate the occupying process via `lsof -i :9102`, or modify `--port` in the launch command.
