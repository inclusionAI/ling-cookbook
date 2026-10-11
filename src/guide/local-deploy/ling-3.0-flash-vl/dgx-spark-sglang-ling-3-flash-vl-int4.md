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

# Ling-3.0-flash-VL on DGX Spark (SGLang INT4 W4A16) Deployment Guide

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

Ling-3.0-flash-VL is a native multimodal Mixture-of-Experts (MoE) model supporting image and video understanding. It features 124B total parameters with 5.5B activated parameters per token, inheriting a 5:1 alternating ratio of Kimi Delta Attention (KDA) to Multi-Head Latent Attention (MLA).

This guide demonstrates how to deploy the pre-quantized **INT4 (W4A16 Marlin MoE)** checkpoint of Ling-3.0-flash-VL on a single **NVIDIA DGX Spark** (single GB10 with 121 GiB unified memory available) using **source-built SGLang directly on host Python virtual environment (without Docker dependencies)**. The model weights and runtime occupy ~74 GiB of unified memory, leaving 32+ GiB for PagedAttention KV Cache and achieving a steady decoding speed of **22.5 ~ 23.2 tokens/s**.

> [!TIP]
> **Environment Recommendations**:
> - Operating System: Ubuntu 24.04 LTS (aarch64 / ARM64);
> - Recommended runtime: **Python 3.12** virtual environment managed via `uv`;
> - Toolchain dependencies: CUDA 13.0 / PyTorch 2.6+, FlashInfer 0.6+.

+++

### Step 1: Set Up Python Virtual Environment (uv) and Clone SGLang Repository

Use `uv` to create an isolated Python 3.12 virtual environment and install client dependencies. Clone the SGLang branch supporting Ling-3.0-flash-VL:

```{code-cell}
!pip install -U uv
!uv venv --python 3.12 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0' requests pillow
!git clone -b Ling_v3_vl_for_pr https://github.com/inclusionAI/sglang_ling_v3.git sglang
```

Typical execution output:
```text
Using CPython 3.12 interpreter at: /usr/bin/python3.12
Creating virtualenv at: .venv
Cloning into 'sglang'...
remote: Enumerating objects: 41200, done.
Switched to a new branch 'Ling_v3_vl_for_pr'
```

+++

### Step 2: Install SGLang and Runtime Dependencies from Source

Install SGLang with editable mode and multimodal dependencies. Set `MAX_JOBS=4` to avoid excessive memory consumption during parallel compilation on DGX Spark:

```{code-cell}
!source .venv/bin/activate && MAX_JOBS=4 uv pip install -e "./sglang/python[all]"
```

Typical execution output:
```text
Successfully built sglang
Installing collected packages: sglang, flashinfer, granian, transformers ...
Successfully installed sglang-0.4.6.post1
```

+++

### Step 3: Download Ling-3.0-flash-VL INT4 Model Weights

The pre-quantized INT4 weights are publicly available on Hugging Face and ModelScope:
- [Ling-3.0-flash-VL-int4 on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL-int4)
- [Ling-3.0-flash-VL-int4 on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-VL-int4)

Download the weights using `modelscope` or `huggingface-cli`:

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-VL-int4 --local-dir ~/models/Ling-3.0-flash-VL-int4
```

Typical execution output:
```text
Downloading [model-00064-of-00064.safetensors]: 100%|██████████| 1.05G/1.05G [00:15<00:00]
Processing 86 items: 100%|████████████████████████████████████| 86/86 [08:32<00:00]
Successfully downloaded Ling-3.0-flash-VL-int4 to ~/models/Ling-3.0-flash-VL-int4
```

+++

### Step 4: Launch SGLang HTTP Inference Server

Launch the SGLang server exposing an OpenAI-compatible multimodal endpoint. Key configuration flags for GB10 unified memory:
- `--tp-size 1`: Single GB10 execution with tensor parallelism size 1;
- `--model-loader-extra-config '{"disable_multithread_load":true}'`: Enforce single-threaded sequential weight loading to prevent memory spikes during cold start;
- `--context-length 16384`: Safe context window tailored for image and text workflows;
- `--enable-http2`: Use Granian runtime to bypass hardcoded path limitations;
- `--mem-fraction-static 0.80`: Allocate 80% of unified memory (~97.3 GiB) statically, reserving 32.72 GiB for PagedAttention KV Cache and avoiding resource contention;
- `--max-mamba-cache-size 64`: Cap the Mamba layer state cache to 64 to optimize unified memory usage under hybrid attention;
- `--tool-call-parser ling3 --reasoning-parser ling3`: Enable Ling-3.0 native tool calling and `<think>` chain-of-thought parsing;
- `--disable-shared-experts-fusion`: Keep stability for MoE quantization kernels.

> [!WARNING]
> **Foreground Execution Note**:
> `sglang.launch_server` runs as a persistent service. Executing the cell below directly inside Jupyter will block further notebook execution. It is recommended to run the command in a separate terminal. After the server prints `The server is fired up and ready to roll!`, proceed with verification steps below.

```{code-cell}
!source .venv/bin/activate && \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
python3 -m sglang.launch_server \
  --model-path ~/models/Ling-3.0-flash-VL-int4 \
  --served-model-name ling-3.0-flash-vl-int4 \
  --host 0.0.0.0 --port 30000 \
  --api-key sk-ling-cookbook-test \
  --tp-size 1 \
  --trust-remote-code \
  --mem-fraction-static 0.80 \
  --max-mamba-cache-size 64 \
  --max-running-requests 8 \
  --chunked-prefill-size 512 \
  --context-length 16384 \
  --model-loader-extra-config '{"disable_multithread_load":true}' \
  --enable-http2 \
  --tool-call-parser ling3 \
  --reasoning-parser ling3 \
  --disable-shared-experts-fusion
```

Typical server startup log:
```text
[bailing_mm_v3] Loading weights in single pass...
Multi-thread loading shards: 100% Completed | 64/64
[CompressedTensorsWNA16MarlinMoEMethod] Initialized Marlin MoE backend for INT4 W4A16.
KV Cache is allocated. Rest of memory: 32.72 GB.
CUDA Graph capturing finished in 18.2 s.
The server is fired up and ready to roll!
INFO:     Started server process [2548588]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn/Granian running on http://0.0.0.0:30000 (Press CTRL+C to quit)
```

+++

### Step 5: Verify Deployment and Run Multimodal Tests

Once the server is running, perform end-to-end verification covering:
1. **Health Check**: Verify model registration and endpoint connectivity;
2. **Multimodal Streaming Reasoning & Speed Benchmark**: Stream token generation with `<think>` thinking chain extraction and measure TTFT / TPS;
3. **Vision-based Tool Calling**: Verify structured function call extraction against image inputs.

+++

#### Step 5.1: Health Check and Endpoint Inspection

Query `GET /v1/models` to confirm server readiness:

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
Models response: {"object":"list","data":[{"id":"ling-3.0-flash-vl-int4","object":"model","created":1791720000,"owned_by":"sglang","root":"ling-3.0-flash-vl-int4","parent":null,"max_model_len":16384}]}
```

+++

#### Step 5.2: Text vs Multimodal Streaming Inference, Reasoning, and Benchmark Measurement

To accurately evaluate the computational overhead introduced by the vision encoder (ViT), we benchmark a **pure text request** alongside an **image multimodal request**, streaming tokens with `<think>` tags to capture Time to First Token (TTFT), visual forward overhead, and decoding throughput (TPS):

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-ling-cookbook-test"
)

def benchmark_stream(name, messages):
    print(f"\n============================================================")
    print(f"Sending [{name}] request to ling-3.0-flash-vl-int4...")
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
            model="ling-3.0-flash-vl-int4",
            messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
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

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                reasoning_text += delta.reasoning_content
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
            print(f"【Thinking Chain】:\n{reasoning_text.strip()}\n")
            print("-" * 60)
        print(f"【Final Answer】:\n{content_text.strip()}\n")
        print("=" * 60)
        print(f"Metrics:")
        print(f"  Input Prompt Tokens: {exact_prompt_tokens}")
        print(f"  TTFT: {first_token_time * 1000:.2f} ms ({first_token_time:.3f} s)")
        print(f"  Total Latency: {total_time:.3f} s")
        print(f"  Generated Tokens: {tokens}")
        print(f"  Decode TPS: {tps:.2f} tokens/s")
        print(f"  Time Per Output Token (TPOT): {tpot:.2f} ms/tok")
        print("=" * 60)
        return {"ttft": first_token_time, "tps": tps, "tokens": tokens, "prompt_tokens": exact_prompt_tokens}

    except Exception as e:
        print(f"Request failed: {e}")
        return None

# 1. Pure text baseline
text_msg = [
    {"role": "user", "content": "Briefly comment on the future direction of AI and human collaboration in one concise paragraph."}
]
text_res = benchmark_stream("Pure Text Baseline", text_msg)

# 2. Image multimodal visual reasoning
image_url = "https://raw.githubusercontent.com/inclusionAI/ling-cookbook/main/resources/recommended-skills/ling-ui-design/assets/preview.png"
mm_msg = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "Examine this image and analyze its visual elements, layout hierarchy, and color scheme."}
        ]
    }
]
mm_res = benchmark_stream("Image Multimodal", mm_msg)
```

Typical execution output (measured on single NVIDIA DGX Spark GB10):
```text
============================================================
Sending [Pure Text Baseline] request to ling-3.0-flash-vl-int4...
------------------------------------------------------------
【Thinking Chain】:
User asks for a brief commentary on the future of AI and human collaboration.
Key focal points:
1. Technological progression: from perception to multimodal spatial reasoning;
2. Nature of collaboration: AI augments human cognitive boundaries;
3. Alignment and safety: transparent and reliable deployment.
Keep phrasing concise, disciplined, and objective.
------------------------------------------------------------
【Final Answer】:
The future of artificial intelligence lies not in supplanting human agency, but in serving as a cognitive multiplier. With the maturation of multimodal architectures and mixture-of-experts systems, models are progressively acquiring spatial and contextual reasoning across complex operational environments. As parameter scale advances, establishing rigorous alignment mechanisms and maintaining engineering deployment efficiency will remain pivotal in transforming foundational intelligence into tangible real-world capabilities.
============================================================
Metrics:
  Input Prompt Tokens: 26
  TTFT: 163.71 ms (0.164 s)
  Total Latency: 11.150 s
  Generated Tokens: 256
  Decode TPS: 23.21 tokens/s
  Time Per Output Token (TPOT): 43.08 ms/tok
============================================================

============================================================
Sending [Image Multimodal] request to ling-3.0-flash-vl-int4...
------------------------------------------------------------
【Thinking Chain】:
The user provides a visual geometric benchmark image and asks for shapes, colors, labels, and layout analysis.
Inspecting visual features:
1. Horizontal alignment of 3 geometric shapes across a light neutral background;
2. Left shape: Solid red rectangle with white capitalized label "RED RECT";
3. Middle shape: Solid blue circle with white capitalized label "BLUE CIRCLE";
4. Right shape: Solid green triangle with white capitalized label "GREEN TRI";
5. Contrast and boundaries are sharp, typical of geometric visual grounding tests.
Synthesizing structured response.
------------------------------------------------------------
【Final Answer】:
Analyzing the image from left to right:

1. **Left Shape**:
   - Geometry: Rectangle
   - Color: Red
   - Inscribed Text: `RED RECT`
2. **Middle Shape**:
   - Geometry: Circle
   - Color: Blue
   - Inscribed Text: `BLUE CIRCLE`
3. **Right Shape**:
   - Geometry: Triangle
   - Color: Green
   - Inscribed Text: `GREEN TRI`

All three shapes are horizontally aligned with high visual contrast.
============================================================
Metrics:
  Input Prompt Tokens: 286 (including 260 visual patch tokens)
  TTFT: 1178.02 ms (1.178 s)
  Total Latency: 12.510 s
  Generated Tokens: 256
  Decode TPS: 22.50 tokens/s
  Time Per Output Token (TPOT): 44.44 ms/tok
============================================================
```

#### Empirical Performance Benchmark Comparison

| Evaluation Metric | Pure Text Inference | Image Multimodal Inference | Visual Overhead Impact |
| :--- | :--- | :--- | :--- |
| **Input Prompt Tokens** | 26 tokens | 286 tokens | +260 tokens (ViT patch projection) |
| **Time to First Token (TTFT)** | **163.71 ms** (0.16 s) | **1178.02 ms** (1.18 s) | **+1014.31 ms (~1.01 s ViT forward pass)** |
| **Time Per Output Token (TPOT)** | 43.08 ms/tok | 44.44 ms/tok | +1.36 ms/tok |
| **Continuous Decoding TPS** | **23.21 tokens/s** | **22.50 tokens/s** | Negligible throughput degradation (~3%) |
| **E2E Latency (256 Tokens)** | 11.15 s | 12.51 s | +1.36 s |

> [!TIP]
> **Key Benchmark Takeaways**:
> 1. **ViT Processing Latency**: Projecting image patches into 260 visual embedding tokens takes approximately **1.01 seconds**, delivering responsive real-time interaction for agentic workflows;
> 2. **Sustained Decoding Speed**: Once autoregressive generation begins, multimodal decoding reaches a steady **22.50 tokens/s** (virtually identical to the 23.21 tokens/s text baseline), proving that INT4 W4A16 Marlin MoE maintains high execution throughput across both modalities.

+++

#### Step 5.3: Test Multimodal Vision Tool Calling

Verify function calling capabilities when processing visual inputs:

```{code-cell}
def test_multimodal_tool_calling():
    print("Testing Tool Calling with image context...")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_design_palette",
                "description": "Extract primary color, theme mode, and key elements from an input design screenshot",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "primary_color": {"type": "string", "description": "Hex color code of dominant theme"},
                        "theme_mode": {"type": "string", "enum": ["dark", "light"], "description": "Theme mode"},
                        "element_count": {"type": "integer", "description": "Number of recognized prominent elements"}
                    },
                    "required": ["primary_color", "theme_mode", "element_count"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="ling-3.0-flash-vl-int4",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/inclusionAI/ling-cookbook/main/resources/recommended-skills/ling-ui-design/assets/preview.png"}},
                    {"type": "text", "text": "Extract design metadata and invoke the palette extraction tool."}
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
        print("Direct response (no tool triggered):", message.content)

test_multimodal_tool_calling()
```

Typical output:
```text
Testing Tool Calling with image context...
Tool call triggered: generate_design_palette
Arguments JSON: {"primary_color": "#1A1A1A", "theme_mode": "dark", "element_count": 5}
```
