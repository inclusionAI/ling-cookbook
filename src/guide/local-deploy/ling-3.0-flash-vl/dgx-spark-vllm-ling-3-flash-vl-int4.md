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

# Ling-3.0-flash-VL on DGX Spark (vLLM INT4) Deployment Guide

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

This guide demonstrates deploying the official INT4 (`compressed-tensors` / Marlin MoE) quantized checkpoint of Ling-3.0-flash-VL on a single `NVIDIA DGX Spark` (single GB10 with 121 GiB unified memory) using `vLLM`. Configuring `--max-model-len 32768` and `--gpu-memory-utilization 0.80` reserves memory for KV Cache and the host OS. Measured visual encoding and prefill latency is ~1.5 seconds (TTFT 1596 ms), with end-to-end processing of ~1.5 to 2.0 seconds per image, suitable for batch image tagging, classification, and structured visual metadata extraction.

> [!TIP]
> Environment Requirements:
> - Operating System: Ubuntu 24.04 LTS (aarch64 / ARM64);
> - Python: Python 3.12 virtual environment managed by `uv`;
> - Dependencies: CUDA 13.0 driver environment, official vLLM ARM64 nightly cu130 wheel.

+++

### Step 1: Prepare Python 3.12 Virtual Environment (uv)

Create an isolated Python 3.12 virtual environment using `uv` to satisfy PEP 701 syntax requirements:

```{code-cell}
!pip install -U uv
!uv venv --python 3.12 .venv
```

Typical output:
```text
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtualenv at: .venv
Activate with: source .venv/bin/activate
```

+++

### Step 2: Install Official vLLM ARM64 CUDA 13.0 Wheel and Dependencies

The Ling-3.0 architecture (`BailingMoeV3VLForConditionalGeneration`) and the `ling3` reasoning chain and tool-call parsers are merged upstream in vLLM. Install the precompiled Nightly Wheel targeting GB10 (sm_121 / cu130) via `uv pip`, and install testing libraries (`requests`, `pillow`):

```{code-cell}
!source .venv/bin/activate && uv pip install \
  --upgrade vllm \
  --torch-backend=cu130 \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130
!source .venv/bin/activate && uv pip install requests pillow
```

Typical output:
```text
Resolved 198 packages in 52.80s
Installed 198 packages in 45.2s
 + vllm==0.31.1rc1.dev282+gbd5bbce6e
 + openai==3.28.0
 + pillow==12.3.0
 + requests==2.34.2
```

+++

### Step 3: Download Ling-3.0-flash-VL INT4 Checkpoint

Pre-quantized INT4 weights are publicly available on Hugging Face and ModelScope:
- [Ling-3.0-flash-VL-int4 on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL-int4)
- [Ling-3.0-flash-VL-int4 on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-VL-int4)

Download using the `modelscope` CLI:

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-VL-int4 --local-dir ~/models/Ling-3.0-flash-VL-int4
```

Typical output:
```text
Downloading [model-00064-of-00064.safetensors]: 100%|██████████| 1.05G/1.05G [00:15<00:00]
Processing 86 items: 100%|████████████████████████████████████| 86/86 [08:32<00:00]
Successfully downloaded Ling-3.0-flash-VL-int4 to ~/models/Ling-3.0-flash-VL-int4
```

+++

### Step 4: Launch vLLM Inference Service

Launch `vllm serve` providing an OpenAI-compatible HTTP interface:
- `--tensor-parallel-size 1`: Single GPU tensor parallelism;
- `--trust-remote-code`: Allow loading Ling-3.0-flash-VL model architecture code;
- `--max-model-len 32768`: Set context length to 32,768 tokens (default is 131,072) to reserve memory for KV Cache;
- `--gpu-memory-utilization 0.80`: Cap static memory allocation at 80% (~96.8 GiB), reserving 24+ GiB for the host OS and buffers;
- `--reasoning-parser ling3`: Enable the `ling3` parser to output `reasoning_content`;
- `--enable-auto-tool-choice --tool-call-parser ling3`: Enable the `ling3` tool call parser.

> [!WARNING]
> Foreground Execution Notice:
> `vllm serve` runs in the foreground. Execute the command below in an independent terminal, wait for `Application startup complete.`, and return to this notebook.

```{code-cell}
!source .venv/bin/activate && \
vllm serve ~/models/Ling-3.0-flash-VL-int4 \
  --served-model-name ling-3.0-flash-vl-int4 \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key sk-ling-cookbook-test \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.80 \
  --reasoning-parser ling3 \
  --enable-auto-tool-choice \
  --tool-call-parser ling3
```

Typical startup log:
```text
(APIServer pid=2666912) INFO 10-12 13:26:45 [api_utils.py:347]        █     █     █▄   ▄█
(APIServer pid=2666912) INFO 10-12 13:26:45 [api_utils.py:347]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.31.1rc1.dev282+gbd5bbce6e
(APIServer pid=2666912) INFO 10-12 13:26:45 [api_utils.py:347]   █▄█▀ █     █     █     █  model   /home/squall/models/Ling-3.0-flash-VL-int4
(APIServer pid=2666912) INFO 10-12 13:26:45 [api_utils.py:347]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(EngineCore pid=2667050) INFO 10-12 13:26:54 [int_wna16.py:356] Using 'MARLIN' WNA16 MoE backend.
(EngineCore pid=2667050) INFO 10-12 13:26:55 [selector.py:131] Using FLASH_ATTN MLA prefill backend.
(EngineCore pid=2667050) INFO 10-12 13:26:55 [deep_gemm.py:225] DeepGEMM PDL enabled on vllm.third_party.deep_gemm.
(EngineCore pid=2667050) INFO 10-12 13:26:55 [deep_gemm.py:137] DeepGEMM E8M0 enabled on current platform.
Loading safetensors checkpoint shards: 100% Completed | 64/64
(EngineCore pid=2667050) INFO 10-12 13:37:08 [monitor.py:62] [backbone] torch.compile took 1.04 s in total
Capturing CUDA graphs (PIECEWISE): 100% Completed | 51/51
INFO:     Started server process [2666912]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
```

+++

### Step 5: Verify Deployment and Run Benchmarks

Once the service is active, verify functionality and performance using an OpenAI-compatible client:
1. Health check: Verify model endpoint connectivity;
2. Streaming inference and benchmarking: Compare pure text and vision queries, checking first-token latency (TTFT) and decoding throughput (TPS);
3. Multimodal tool calling: Verify parameter extraction from images.

+++

#### Step 5.1: Connectivity and Model Health Check

Query the `GET /v1/models` endpoint:

```{code-cell}
import urllib.request
import json

url = "http://127.0.0.1:30000/v1/models"
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
Models response: {"object":"list","data":[{"id":"ling-3.0-flash-vl-int4","object":"model","created":1791786357,"owned_by":"vllm","root":"/home/squall/models/Ling-3.0-flash-VL-int4","parent":null,"max_model_len":32768,"permission":[{"id":"modelperm-8ee3a120cb6f8323","object":"model_permission","created":1791786357,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```

+++

#### Step 5.2: Streaming Inference and Benchmarking

Compare pure text and multimodal streaming requests, parse `<think>` reasoning chains, and record TTFT, TPOT, and decoding TPS:

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
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

            # Extract reasoning tokens: compatible with vLLM ling3 parser reasoning and reasoning_content fields
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

# 2. Image multimodal visual reasoning test (Geometric visual analysis)
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

Typical benchmark results (measured on a single NVIDIA DGX Spark GB10):
```text
============================================================
Sending [Pure Text Benchmark] request to ling-3.0-flash-vl-int4...
------------------------------------------------------------
[Thinking Process]:
User asks for a concise paragraph reviewing the future of human-AI collaboration. Key themes: symbiosis, mutual enhancement, human-centric alignment...
------------------------------------------------------------
[Model Answer]:
Human-AI collaboration is evolving toward a paradigm of deep symbiosis and complementary strengths. AI will continue empowering humans in data processing, pattern recognition, and automated execution, while human judgment, creative synthesis, and ethical discernment remain irreplaceable...
============================================================
Metrics:
  Input Prompt Tokens: 34
  TTFT: 218.05 ms (0.218 s)
  Thinking Duration: 1.623 s (35 tokens)
  Answer Duration: 5.648 s (125 tokens)
  Total Duration: 7.489 s
  Generated Tokens: 162 (Reasoning: 35, Content: 125)
  Decoding Throughput (TPS): 22.14 tokens/s
  Time Per Output Token (TPOT): 45.16 ms/tok
============================================================

============================================================
Sending [Image Multimodal] request to ling-3.0-flash-vl-int4...
------------------------------------------------------------
[Thinking Process]:
The user wants me to observe and analyze the visual elements, compositional hierarchy, and main colors of this image.
1. Top left: a red rectangle (labeled "RED RECT")...
2. Top right: a blue circle (labeled "BLUE CIRCLE")...
3. Bottom center: a green triangle (labeled "GREEN TRI")...
------------------------------------------------------------
[Model Answer]:
## 1. Visual Element Identification
The image consists of three distinct geometric shapes, each containing white sans-serif text labels:
1. Red Rectangle (RED RECT) - Located in the upper left, approximately square, left-aligned label.
2. Blue Circle (BLUE CIRCLE) - Located in the upper right, centered label.
3. Green Triangle (GREEN TRI) - Positioned centrally at the bottom, horizontal base, upward apex.

## 2. Compositional Hierarchy
- The overall layout forms an inverted triangular structure: two shapes on top, one centered below, establishing a balanced foundation.
- Primary visual anchor: The green triangle with the widest horizontal base provides a stable visual grounding.
- Secondary elements: The blue circle and red rectangle create lateral equilibrium.
- Negative space: Generous margins keep elements distinct without visual occlusion.

## 3. Color Palette
The image employs a high-contrast triad (red, blue, green) set against a pale blue-gray neutral background.
============================================================
Metrics:
  Input Prompt Tokens: 296 (including 262 vision feature tokens)
  TTFT: 1596.51 ms (1.597 s)
  Thinking Duration: 90.259 s (1814 tokens)
  Answer Duration: 26.360 s (577 tokens)
  Total Duration: 118.215 s
  Generated Tokens: 2555 (Reasoning: 1814, Content: 577)
  Decoding Throughput (TPS): 21.90 tokens/s
  Time Per Output Token (TPOT): 45.66 ms/tok
============================================================
```

#### Measured Performance Benchmark Comparison

| Metric | Pure Text Streaming | Image Multimodal Streaming | Delta |
| :--- | :--- | :--- | :--- |
| Input Prompt Tokens | 34 tokens | 296 tokens | +262 tokens |
| First Token Latency (TTFT) | 218.05 ms (0.22 s) | 1596.51 ms (1.60 s) | +1378.46 ms (~1.38 s) |
| Thinking Phase Duration | 1.62 s (35 tokens) | 90.26 s (1814 tokens) | +88.64 s (+1779 tokens) |
| Answer Generation Duration | 5.65 s (125 tokens) | 26.36 s (577 tokens) | +20.71 s (+452 tokens) |
| Time Per Token (TPOT) | 45.16 ms/tok | 45.66 ms/tok | +0.50 ms/tok |
| Decoding Throughput (TPS) | 22.14 tokens/s | 21.90 tokens/s | -0.24 tokens/s (-1.1%) |
| Total Output Tokens | 162 tokens | 2555 tokens | +2393 tokens |
| End-to-End Latency | 7.49 s | 118.22 s | - |

> [!TIP]
> Benchmark Notes:
> 1. First-token latency (TTFT): Multimodal image input (including 262 vision tokens) records a TTFT of 1596.51 ms (~1.60 s), an increase of ~1.38 seconds over text baseline;
> 2. Continuous decoding throughput: Throughput remains steady across modalities at 22.14 tokens/s for text and 21.90 tokens/s for vision;
> 3. Execution phases: In streaming mode, the `ling3` parser emits `reasoning_content` first. First token arrival (TTFT ~1.6 s) and transition to final answer (~91.8 s) represent distinct phases;
> 4. Batch processing: Visual feature extraction and prefill require ~1.5 seconds. For tasks with concise outputs or disabled reasoning (such as image tagging or categorization), single-card end-to-end latency is ~1.5 to 2.0 seconds per image.

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
                "description": "Extract primary color, theme mode, and visual element hierarchy from design images",
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
        model="ling-3.0-flash-vl-int4",
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

Typical test output:
```text
Testing multimodal Tool Calling...
Tool call triggered: generate_design_palette
Arguments JSON: {"primary_color": "#E53935", "theme_mode": "light", "element_count": 3}
```

+++

### Step 6: Troubleshooting and Tuning

- Out of Memory & Memory Allocation:
  - Symptom: Startup fails with `torch.cuda.OutOfMemoryError` or logs `checkpoint size exceeds 90% of available RAM` and disables prefetch;
  - Root cause: vLLM defaults to preallocating KV Cache for 131,072 context length, exceeding physical memory on 121 GiB unified memory systems. Setting `--gpu-memory-utilization` too high (e.g. 0.85+) limits memory available to the host OS and shard loading buffers;
  - Solution: Set `--max-model-len 32768` (or `16384` if needed) and `--gpu-memory-utilization 0.80`, reserving 24+ GiB of memory for the host OS and buffers.
- Blackwell Architecture CUDA Graph Capture:
  - Symptom: Graph capture errors or stuttering under dynamic image resolutions;
  - Solution: Pass `--enforce-eager` to run in Eager mode.
- Python 3.12 Requirements:
  - Symptom: Startup fails with `SyntaxError: unterminated string literal` or `ImportError: cannot import name 'NamespaceTool' from 'openai.types.responses'`;
  - Root cause: vLLM requires Python 3.12+ (PEP 701 syntax) and `openai>=2.0.0` (`openai 3.28.0`);
  - Solution: Create an environment with `uv venv --python 3.12 .venv` and install dependencies with `uv pip install vllm`.
