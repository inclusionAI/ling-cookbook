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

# Ling-3.0-flash-VL on DGX Spark (llama.cpp Q4_K_M GGUF) Deployment Guide

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

Ling-3.0-flash-VL is a multimodal MoE model supporting image and video understanding, featuring 124B total parameters with ~5.5B activated parameters per token and a 5:1 alternating ratio of KDA to MLA.

This guide demonstrates deploying a 4-bit quantized (Q4_K_M) checkpoint of Ling-3.0-flash-VL and its paired multimodal vision projector (`mmproj`) on a single `NVIDIA DGX Spark` (single GB10 with 121 GiB unified memory) using `llama.cpp`. Static weights and projector occupy ~80 GB, reserving ~40 GiB for KV Cache and the host OS when configured with a context length of 32,768 tokens, offering a lightweight C++ runtime with an OpenAI-compatible multimodal HTTP interface.

> [!TIP]
> Environment Requirements:
> - Operating System: Ubuntu 24.04 LTS (aarch64 / ARM64);
> - Python: Python 3.12 virtual environment managed by `uv`;
> - Dependencies: CUDA 13.0 driver environment, CMake 3.28+, C++17 compiler, latest llama.cpp master (>= b11156).

+++

### Step 1: Prepare Python 3.12 Virtual Environment (uv)

Create an isolated Python 3.12 virtual environment using `uv` and install testing dependencies:

```{code-cell}
!pip install -U uv
!uv venv --python 3.12 .venv
!source .venv/bin/activate && uv pip install openai pillow requests modelscope
```

Typical output:
```text
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtualenv at: .venv
Activate with: source .venv/bin/activate
Installed 4 packages in 12.3s
```

+++

### Step 2: Clone and Build llama.cpp (CUDA Enabled)

Support for the Ling-3.0 MoE architecture (`bailingmoe3`) and multimodal vision projection (`mmproj`) is merged upstream in `llama.cpp` (PR #26608 and PR #29151). Clone the repository and compile with CUDA acceleration:

```{code-cell}
!git clone https://github.com/ggml-org/llama.cpp.git
!cd llama.cpp && cmake -B build -DGGML_CUDA=ON . && cmake --build build --parallel 8
```

Typical output:
```text
Cloning into 'llama.cpp'...
-- The CXX compiler identification is GNU 13.3.0
-- The CUDA compiler identification is NVIDIA 13.0.88
-- Detecting CUDA compiler ABI info - done
-- Building with CUDA architecture: native
[100%] Built target llama-server
[100%] Built target llama-quantize
```

+++

### Step 3: Download Ling-3.0-flash-VL GGUF and mmproj Weights

Unquantized BF16 weights require ~248 GB of memory, which exceeds physical memory on a single 121 GiB device. Use the ModelScope CLI to download pre-quantized GGUF files and the multimodal projector (`mmproj`):
- [AtomicChat/Ling-3.0-flash-VL-GGUF on ModelScope](https://modelscope.cn/models/AtomicChat/Ling-3.0-flash-VL-GGUF)
- [bartowski/Ling-3.0-flash-VL-GGUF on Hugging Face](https://huggingface.co/bartowski/Ling-3.0-flash-VL-GGUF)

Use the `modelscope` CLI to download the `Q4_K_M` shards and `mmproj-Ling-3.0-flash-VL-F32.gguf`:

```{code-cell}
!source .venv/bin/activate && modelscope download --model AtomicChat/Ling-3.0-flash-VL-GGUF \
  --include "Ling-3.0-flash-VL-AD-Q4_K_M*" "mmproj-Ling-3.0-flash-VL-F32.gguf" \
  --local-dir ~/models/Ling-3.0-flash-VL-GGUF
```

Typical output:
```text
Downloading snapshot of AtomicChat/Ling-3.0-flash-VL-GGUF (model)…
Downloading 7 files from AtomicChat/Ling-3.0-flash-VL-GGUF@master
Ling-3.0-flash-VL-AD-Q4_K_M-00001-of-00006.gguf: 100%|██████████| 15.6G/15.6G [08:12<00:00, 31.7MB/s]
...
mmproj-Ling-3.0-flash-VL-F32.gguf: 100%|██████████| 1.74G/1.74G [00:58<00:00, 29.8MB/s]
Successfully downloaded to ~/models/Ling-3.0-flash-VL-GGUF
```

+++

### Step 4: Launch llama-server Inference Service

Launch `llama-server` providing an OpenAI-compatible HTTP interface:
- `-m`: Specify the language model weight path (shard root or single file);
- `--mmproj`: Mount the multimodal vision projector for image encoding;
- `--host 0.0.0.0 --port 9102`: Listen on port 9102;
- `-ngl 99`: Offload all 42 Transformer and MoE layers to GPU;
- `-c 32768`: Set context length to 32,768 tokens to reserve KV Cache memory;
- `--jinja`: Enable Jinja template rendering for `<think>` reasoning chains.

> [!WARNING]
> Foreground Execution Notice:
> `llama-server` runs in the foreground. Execute the command below in an independent terminal, wait for `HTTP server listening`, and return to this notebook.

```{code-cell}
!./llama.cpp/build/bin/llama-server \
  -m ~/models/Ling-3.0-flash-VL-GGUF/Ling-3.0-flash-VL-AD-Q4_K_M-00001-of-00006.gguf \
  --mmproj ~/models/Ling-3.0-flash-VL-GGUF/mmproj-Ling-3.0-flash-VL-F32.gguf \
  --host 0.0.0.0 \
  --port 9102 \
  --api-key sk-ling-cookbook-test \
  -ngl 99 \
  -c 32768 \
  --jinja
```

Typical startup log:
```text
llama_model_loader: loaded meta data with 38 key-value pairs and 685 tensors
llama_model_load: general.architecture       = bailingmoe3
llama_model_load: bailingmoe3.expert_count    = 512
llama_model_load: bailingmoe3.expert_used_count = 8
clip_model_load: loaded multimodal projector
clip_model_load: vision encoder initialized
llama_backend_init: GGML_CUDA enabled
main: HTTP server listening on http://0.0.0.0:9102
```

+++

### Step 5: Verify Deployment and Run Benchmarks

Once active, test functionality and throughput using an OpenAI-compatible client:
1. Health check: Confirm model endpoint connectivity;
2. Streaming inference and benchmarking: Compare text and image queries, recording TTFT, TPOT, and decoding TPS;
3. Multimodal tool calling: Verify parameter extraction from visual input.

+++

#### Step 5.1: Connectivity and Model Health Check

Query the `GET /v1/models` endpoint:

```{code-cell}
import urllib.request
import json

url = "http://127.0.0.1:9102/v1/models"
headers = {"Authorization": "Bearer sk-ling-cookbook-test"}
try:
    req = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req) as response:
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
Models response: {"object":"list","data":[{"id":"Ling-3.0-flash-VL-Q4_K_M","object":"model","created":1791788200,"owned_by":"llamacpp"}]}
```

+++

#### Step 5.2: Streaming Inference and Benchmarking

Compare text and image streaming requests, parse `<think>` reasoning chains, and record latency and throughput:

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9102/v1",
    api_key="sk-ling-cookbook-test"
)

def benchmark_stream(name, messages):
    print(f"\n============================================================")
    print(f"Sending [{name}] request to llama-server...")
    print("-" * 60)
    
    start_time = time.time()
    first_token_time = None
    chunk_count = 0
    exact_completion_tokens = None
    exact_prompt_tokens = None
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model="Ling-3.0-flash-VL-Q4_K_M",
            messages=messages,
            temperature=0.7,
            top_p=0.95,
            stream=True,
            stream_options={"include_usage": True}
        )

        for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                exact_completion_tokens = chunk.usage.completion_tokens
                exact_prompt_tokens = chunk.usage.prompt_tokens

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            r_text = getattr(delta, "reasoning_content", None)
            if not r_text and delta.model_extra:
                r_text = delta.model_extra.get("reasoning") or delta.model_extra.get("reasoning_content")

            if r_text:
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                reasoning_text += r_text
                chunk_count += 1

            if delta.content:
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                content_text += delta.content
                chunk_count += 1

        total_time = time.time() - start_time
        tokens = exact_completion_tokens if exact_completion_tokens else chunk_count
        decode_time = total_time - (first_token_time if first_token_time else 0)
        tps = (tokens - 1) / decode_time if decode_time > 0 and tokens > 1 else 0
        tpot = (decode_time / (tokens - 1) * 1000) if tokens > 1 else 0

        if reasoning_text:
            print(f"[Thinking Process]:\n{reasoning_text.strip()}\n")
            print("-" * 60)
        print(f"[Model Answer]:\n{content_text.strip()}\n")
        print("=" * 60)
        print(f"Metrics:")
        print(f"  Input Prompt Tokens: {exact_prompt_tokens}")
        print(f"  TTFT: {first_token_time * 1000:.2f} ms ({first_token_time:.3f} s)")
        print(f"  Total Duration: {total_time:.3f} s")
        print(f"  Generated Tokens: {tokens}")
        print(f"  Decoding Throughput (TPS): {tps:.2f} tokens/s")
        print(f"  Time Per Output Token (TPOT): {tpot:.2f} ms/tok")
        print("=" * 60)
        return {"ttft": first_token_time, "tps": tps, "tokens": tokens, "prompt_tokens": exact_prompt_tokens}

    except Exception as e:
        print(f"Request failed: {e}")
        return None

# 1. Pure text benchmark test
text_msg = [
    {"role": "user", "content": "Please write a concise paragraph reviewing the future direction of human-AI collaboration."}
]
text_res = benchmark_stream("Pure Text Benchmark", text_msg)

# 2. Image multimodal visual reasoning test
image_url = "https://raw.githubusercontent.com/inclusionAI/ling-cookbook/main/resources/assets/ling-flash-vl-benchmark.jpg"
mm_msg = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "Please observe and analyze the visual elements, compositional hierarchy, and primary color scheme in this image."}
        ]
    }
]
mm_res = benchmark_stream("Image Multimodal", mm_msg)
```

Typical output:
```text
============================================================
Sending [Pure Text Benchmark] request to llama-server...
------------------------------------------------------------
[Model Answer]:
Human-AI collaboration is evolving toward cognitive symbiosis. Machine learning models contribute vast data processing, pattern recognition, and automated synthesis, while humans guide normative choices, context-sensitive judgment, and intentional framing...
============================================================
Metrics:
  Input Prompt Tokens: 32
  TTFT: 245.30 ms (0.245 s)
  Total Duration: 6.820 s
  Generated Tokens: 135
  Decoding Throughput (TPS): 20.38 tokens/s
  Time Per Output Token (TPOT): 49.07 ms/tok
============================================================

============================================================
Sending [Image Multimodal] request to llama-server...
------------------------------------------------------------
[Model Answer]:
## 1. Visual Element Identification
The image features three labeled geometric shapes:
1. Red Rectangle (RED RECT): Located at the upper left.
2. Blue Circle (BLUE CIRCLE): Located at the upper right.
3. Green Triangle (GREEN TRI): Centered at the bottom.

## 2. Compositional Hierarchy
The arrangement forms an inverted triangle, where the base is anchored by the wide green triangle below, balanced by the two upper shapes.

## 3. Color Palette
The triad uses red, blue, and green on a neutral light blue-gray background.
============================================================
Metrics:
  Input Prompt Tokens: 296
  TTFT: 1780.20 ms (1.780 s)
  Total Duration: 24.150 s
  Generated Tokens: 450
  Decoding Throughput (TPS): 20.12 tokens/s
  Time Per Output Token (TPOT): 49.70 ms/tok
============================================================
```

#### Measured Performance Benchmark Comparison

| Metric | Pure Text Streaming | Image Multimodal Streaming | Delta |
| :--- | :--- | :--- | :--- |
| Input Prompt Tokens | 32 tokens | 296 tokens | +264 tokens |
| First Token Latency (TTFT) | 245.30 ms (0.25 s) | 1780.20 ms (1.78 s) | +1534.90 ms (~1.53 s) |
| Time Per Token (TPOT) | 49.07 ms/tok | 49.70 ms/tok | +0.63 ms/tok |
| Decoding Throughput (TPS) | 20.38 tokens/s | 20.12 tokens/s | -0.26 tokens/s (-1.3%) |
| Total Output Tokens | 135 tokens | 450 tokens | +315 tokens |
| End-to-End Latency | 6.82 s | 24.15 s | - |

> [!TIP]
> Benchmark Notes:
> 1. First-token latency (TTFT): Image input records a TTFT of 1780.20 ms, adding ~1.53 seconds over the text baseline;
> 2. Continuous decoding throughput: Throughput remains steady across modalities at 20.38 tokens/s for text and 20.12 tokens/s for vision;
> 3. Multimodal projector requirement: The `--mmproj` flag is mandatory to process image or video queries.

+++

#### Step 5.3: Test Multimodal Tool Calling

Verify structured tool calling from visual input:

```{code-cell}
def test_multimodal_tool_calling():
    print("Testing multimodal Tool Calling...")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_design_palette",
                "description": "Extract primary color, theme mode, and element count from images",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "primary_color": {"type": "string", "description": "Hexadecimal primary color code"},
                        "theme_mode": {"type": "string", "enum": ["dark", "light"], "description": "Color theme"},
                        "element_count": {"type": "integer", "description": "Detected geometric element count"}
                    },
                    "required": ["primary_color", "theme_mode", "element_count"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="Ling-3.0-flash-VL-Q4_K_M",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/inclusionAI/ling-cookbook/main/resources/assets/ling-flash-vl-benchmark.jpg"}},
                    {"type": "text", "text": "Analyze the color scheme of this image and call the tool to extract design metadata."}
                ]
            }
        ],
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        print(f"Tool call triggered: {tool_call.function.name}")
        print(f"Arguments JSON: {tool_call.function.arguments}")
    else:
        print("Direct response (no tool call):", message.content)

test_multimodal_tool_calling()
```

Typical output:
```text
Testing multimodal Tool Calling...
Tool call triggered: generate_design_palette
Arguments JSON: {"primary_color": "#E53935", "theme_mode": "light", "element_count": 3}
```

+++

### Step 6: Troubleshooting and Tuning

- Memory Allocation & Context Length:
  - Symptom: Startup fails during KV Cache allocation or triggers `CUDA out of memory`;
  - Root cause: `llama-server` defaults to large context windows. On 121 GiB unified memory systems, loading 80 GB of weights leaves insufficient RAM if context length is unconstrained;
  - Solution: Explicitly pass `-c 32768` (or `16384`) to reserve ~40 GiB for the host OS and buffers.
- Multimodal Projector (`mmproj`) Requirement:
  - Symptom: Server returns `image input is not supported` or ignores image tokens;
  - Root cause: Missing `--mmproj` flag at server launch;
  - Solution: Ensure `--mmproj ~/models/Ling-3.0-flash-VL-Q4_K_M/mmproj-Ling-3.0-flash-VL-f16.gguf` is provided.
- CUDA Build Architecture:
  - Symptom: Lower than expected performance on NVIDIA Blackwell architectures;
  - Solution: Build CMake with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native` to enable native architecture optimizations.
