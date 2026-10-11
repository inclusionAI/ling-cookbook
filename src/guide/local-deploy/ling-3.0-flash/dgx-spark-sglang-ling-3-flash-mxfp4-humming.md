---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
##### Copyright 2026 Ant Group and NVIDIA Corporation.

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

# Ling-3.0-flash on DGX Spark (SGLang MXFP4 Humming Optimization) Deployment Guide

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

Special thanks to the NVIDIA team [@ly01325](https://github.com/ly01325) for optimizing and contributing this deployment guide.

+++

`Ling-3.0-flash` is a hybrid attention MoE large language model with 124B total parameters and 5.1B active parameters per token.

This notebook demonstrates how to deploy the MXFP4 quantized version of Ling-3.0-flash on a single **NVIDIA DGX Spark** (Grace Blackwell GB10 / SM121, 121GB unified memory) with three inference optimizations enabled:

1. **Humming MoE Backend**: Uses the Humming operator backend optimized for the Blackwell architecture;
2. **Online FP8 LM Head**: Dynamically quantizes the BF16 LM Head to FP8 online (`SGLANG_ENABLE_FP8_LM_HEAD=1`), halving memory bandwidth pressure during decoding;
3. **MTP Speculative Decoding**: Employs the model's native Multi-Token Prediction architecture with 3-step speculative decoding (NEXTN).

Measured decode generation speed increases from the 37.8 tok/s baseline to ~53.8 tok/s, with verified lossless accuracy across the full GSM8K benchmark test set.

> [!TIP]
> **Environment and Hardware Recommendations**:
> - We recommend using **Python 3.11**, CUDA 13, Ubuntu / DGX OS;
> - We recommend using **uv** to manage isolated virtual environments;
> - Set `--mem-fraction-static` to **0.68** (setting it to 0.75 under heavy unified memory loads risks GPU OOM and driver lockups).

+++

### Step 1: Set Up Python Virtual Environment (uv) and Clone Tuned Branch

Clone the SGLang repository branch (`inclusionAI/sglang` branch `ling_v3_support_mxfp4_humming`) containing the Humming adapter, online FP8 LM Head, and MTP draft sharing patches:

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
# Note: This scheme relies on online FP8 LM head conversion and Eagle/MTP module-level parameter sharing, which are not yet merged upstream
!git clone -b ling_v3_support_mxfp4_humming https://github.com/inclusionAI/sglang.git
```

Typical execution output:
```text
Using CPython 3.11.15
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
Resolved 16 packages in 709ms
Prepared 16 packages in 469ms
Installed 16 packages in 21ms
 + annotated-types==0.8.0
 + anyio==4.15.1
 + certifi==2026.7.22
 + distro==1.9.0
 + h11==0.16.0
 + httpcore==1.0.9
 + httpx==0.28.1
 + idna==3.19
 + jiter==0.16.0
 + openai==1.109.1
 + pydantic==2.13.5
 + pydantic-core==2.46.5
 + sniffio==1.3.1
 + tqdm==4.70.0
 + typing-extensions==4.16.0
 + typing-inspection==0.4.4
Cloning into 'sglang'...
remote: Enumerating objects: 225775, done.
remote: Total 225775 (delta 0), reused 0 (delta 0), pack-reused 225775 (from 1)
Receiving objects: 100% (225775/225775), 189.01 MiB | 19.22 MiB/s, done.
Resolving deltas: 100% (172158/172158), done.
```

+++

### Step 2: Build SGLang from Source

Local inference does not require gRPC, multimodal, or Rust Server extensions.

Setting `SGLANG_BUILD_RUST_EXTS=none` skips the crates.io dependency tree, saving over 30 minutes of installation time.

Set compilation parallelism dynamically based on available memory (capped at 8 cores) to prevent memory exhaustion:

```{code-cell}
!source .venv/bin/activate && \
export SGLANG_BUILD_RUST_EXTS=none && \
J=$(( $(nproc) < 8 ? $(nproc) : 8 )) && \
export MAX_JOBS="$J" NINJA_NUM_JOBS="$J" CMAKE_BUILD_PARALLEL_LEVEL="$J" && \
echo "Compiling with MAX_JOBS=$J, skipping Rust extensions..." && \
uv pip install -e "./sglang/python[all]"
```

Typical execution output:

```text
Compiling with MAX_JOBS=8, skipping Rust extensions...
Resolved 240 packages in 6.88s
      Built sglang @ file:///home/squall/sipan/sglang/python
      Built cuda-tile==1.6.0rc5
      Built xatlas==0.0.11
Prepared 112 packages in 39.15s
Uninstalled 1 package in 5ms
Installed 225 packages in 174ms
```

+++

### Step 3: Download Ling-3.0-flash FP4 (MXFP4) Model Weights

Pre-quantized MXFP4 weights (~60GB) are hosted on:

- [ModelScope Ling-3.0-flash-fp4](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-fp4)
- [Hugging Face Ling-3.0-flash-fp4](https://huggingface.co/inclusionAI/Ling-3.0-flash-fp4)

Download model weights to `~/models/Ling-3.0-flash-fp4` using ModelScope:

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

### Step 4: Precompile FlashInfer CUTLASS MXFP4 Kernels

This step precompiles FlashInfer CUTLASS MoE kernels into a shared library (`.so`).

On a single DGX Spark (121GB unified memory), triggering inline JIT compilation while simultaneously loading ~60GB of model weights can exhaust physical memory and terminate the process.

The Python script below resolves this issue:

1. Triggers the SGLang loading process to generate `build.ninja` build rules;
2. Once rules are generated, terminates the SGLang instance and runs `ninja` to compile kernels with exclusive memory access.

Save the code below as an independent script (e.g., `compile-kernel.py`) and run it in the terminal:

```bash
uv run compile-kernel.py
```

Wait for compilation to finish (typically 6-8 minutes).

Subsequent SGLang service runs will directly load the compiled `.so` file without recompilation.

```{code-cell}
import os
import sys
import shutil
import subprocess
import time
import re
from pathlib import Path

ROOTS = [
    Path.home() / ".cache" / "flashinfer",
    Path.home() / ".cache" / "sglang" / ".cache" / "flashinfer",
]
PATTERN = "*/121a/cached_ops/fused_moe_120"

def newest(name):
    hits = [p for r in ROOTS for p in r.glob(f"{PATTERN}/{name}")]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None

so = newest("fused_moe_120.so")
if so:
    print(f"✅ FlashInfer CUTLASS kernel already compiled, reusing: {so}")
else:
    print("⏳ Precompiled kernel not detected. Starting probe to generate build rules (takes ~6-8 minutes)...")
    state = {"size": -1, "stable": 0}
    
    # Check ninja build rules completion: file size remains unchanged for 3 consecutive checks
    def ninja_ready():
        bn = newest("build.ninja")
        if bn is None:
            return False
        size = bn.stat().st_size
        state["stable"] = state["stable"] + 1 if size == state["size"] else 0
        state["size"] = size
        return state["stable"] >= 3

    # Spawn probe server to trigger rule generation (identical parameters to match shapes)
    env = {
        **os.environ,
        "FLASHINFER_JIT_MAX_JOBS": "1",
        "MAX_JOBS": "1",
        "NINJA_NUM_JOBS": "1",
        "CMAKE_BUILD_PARALLEL_LEVEL": "1",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
        "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
        "SGLANG_DSV4_FP4_DEQUANT": "0",
        "SGLANG_FP8_IGNORED_LAYERS": "",
        "SGLANG_ENABLE_FP8_LM_HEAD": "0"
    }
    probe_cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", str(Path.home() / "models" / "Ling-3.0-flash-fp4"),
        "--served-model-name", "ling-v3-flash-fp4",
        "--trust-remote-code", "--dtype", "bfloat16",
        "--tp-size", "1", "--ep-size", "1",
        "--host", "0.0.0.0", "--port", "30001",
        "--api-key", "sk-ling-cookbook-test",
        "--max-running-requests", "1", "--max-mamba-cache-size", "64",
        "--chunked-prefill-size", "8192", "--max-prefill-tokens", "16384",
        "--page-size", "64", "--context-length", "262144",
        "--cuda-graph-backend-decode", "full",
        "--cuda-graph-max-bs-decode", "1", "--cuda-graph-bs-decode", "1",
        "--cuda-graph-backend-prefill", "disabled",
        "--random-seed", "308534008",
        "--reasoning-parser", "deepseek-r1", "--tool-call-parser", "qwen25",
        "--attention-backend", "flashinfer", "--disable-flashinfer-autotune",
        "--mem-fraction-static", "0.68",
        "--fp8-gemm-backend", "cutlass",
        "--moe-runner-backend", "flashinfer_mxfp4",
        "--flashinfer-mxfp4-moe-precision", "default",
        "--disable-shared-experts-fusion",
        "--enable-fp32-lm-head",
        "--json-model-override-args",
        '{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,'
        '"rope_theta":6000000,"partial_rotary_factor":0.5,'
        '"original_max_position_embeddings":131072}}',
    ]
    
    proc = subprocess.Popen(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    try:
        while not ninja_ready():
            time.sleep(5)
            if proc.poll() is not None:
                raise RuntimeError("Probe server exited unexpectedly, check environment logs")
    finally:
        # Terminate probe server to release 66GB memory
        proc.terminate()
        proc.wait()
        subprocess.run(["pkill", "-f", "sglang.launch_server"], capture_output=True)
        time.sleep(3)
        print("💡 Build rules generated, memory released. Starting exclusive parallel build...")

    bn = newest("build.ninja")
    build_dir = bn.parent
    ninja_bin = shutil.which("ninja") or "ninja"

    # Reserve 10GB per build worker
    avail_gb = next(int(l.split()[1]) / 1048576 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    jobs = max(1, min(os.cpu_count() or 1, int(avail_gb // 10), 8))
    print(f"Building 97 targets with -j{jobs}...")

    ret = subprocess.run([ninja_bin, f"-j{jobs}", "-C", str(build_dir)])
    if ret.returncode != 0:
        raise RuntimeError("FlashInfer ninja operator build failed")
    print("✅ FlashInfer CUTLASS kernels compiled successfully!")
```

Typical execution output:
```text
[96/96] c++ /home/squall/.cache/flashinfer/0.6.17/121a/cached_ops/fused_moe_120/...e/squall/.cache/flashinfer/0.6.17/121a/cached_ops/fused_moe_120/fused_moe_120.so
✅ FlashInfer CUTLASS kernels compiled successfully!
```

+++

### Step 5: Launch SGLang HTTP Inference Service

Launch SGLang HTTP server to provide OpenAI-compatible API endpoints with all optimizations enabled:

- `--moe-runner-backend humming`: Enables Blackwell-optimized Humming MoE kernel;
- `SGLANG_ENABLE_FP8_LM_HEAD=1`: Online dynamic FP8 quantization for LM Head, halving memory bandwidth pressure;
- `--speculative-algorithm NEXTN --speculative-num-steps 3`: Enables 3-step MTP speculative decoding;
- `--mem-fraction-static 0.68`: Sets memory pre-allocation baseline for long context stability.

> [!NOTE]
> `sglang.launch_server` runs in the foreground. If executed directly inside this notebook cell, Jupyter will block. We recommend launching the command in a separate terminal window.

```{code-cell}
!source .venv/bin/activate && \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_DSV4_FP4_DEQUANT=0 \
SGLANG_FP8_IGNORED_LAYERS="" \
SGLANG_ENABLE_FP8_LM_HEAD=1 \
HUMMING_COMPILER=nvrtc \
HUMMING_CACHE_DIR=~/.humming/cache \
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
  --max-prefill-tokens 16384 \
  --page-size 64 \
  --context-length 262144 \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-bs-decode 1 \
  --cuda-graph-backend-prefill disabled \
  --random-seed 308534008 \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser qwen25 \
  --attention-backend flashinfer \
  --disable-flashinfer-autotune \
  --mem-fraction-static 0.68 \
  --fp8-gemm-backend cutlass \
  --moe-runner-backend humming \
  --flashinfer-mxfp4-moe-precision default \
  --disable-shared-experts-fusion \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path ~/models/Ling-3.0-flash-fp4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --json-model-override-args '{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}'
```

Typical startup logs:
```text
[2026-09-07 14:15:30] Online FP8 quantization enabled for lm_head.
[2026-09-07 14:16:10] Load weight end. elapsed=342.11 s, type=BailingMoeV3ForCausalLM
[2026-09-07 14:17:02] Capture draft decode CUDA graph begin...
[2026-09-07 14:18:20] The server is fired up and ready to roll!
```

+++

### Step 6: Verify Deployment and Test API Calls

#### Step 6.1: Connectivity & Model Health Check

Query `GET /v1/models` to inspect the served model:

```{code-cell}
import urllib.request
import json

url = "http://127.0.0.1:30000/v1/models"
headers = {"Authorization": "Bearer sk-ling-cookbook-test"}

try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=5) as response:
        body = response.read().decode('utf-8')
        print("✅ Health check passed!")
        print(f"Models response: {body}")
except Exception as e:
    print(f"❌ Connectivity check failed: {e}")
```

Typical output:
```text
✅ Health check passed!
Models response: {"object":"list","data":[{"id":"ling-v3-flash-fp4","object":"model","created":1788786583,"owned_by":"sglang","root":"ling-v3-flash-fp4","parent":null,"max_model_len":262144}]}
```

+++

#### Step 6.2: Streaming Request, Reasoning, and Benchmark (TTFT / TPS)

Use OpenAI Python SDK with streaming to extract `<think>` reasoning chain tokens and measure TTFT and Decode TPS:

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="sk-ling-cookbook-test"
)

prompt = "Derive 17 × 23 and show the calculation steps in detail."
print(f"Sending prompt: '{prompt}'...")

start_time = time.time()
first_token_time = None
chunk_count = 0
exact_completion_tokens = None
reasoning_text = ""
content_text = ""

try:
    response = client.chat.completions.create(
        model="ling-v3-flash-fp4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
        extra_body={
            "top_k": 29,
            "chat_template_kwargs": {"enable_thinking": True}
        },
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
        if first_token_time is None and (delta.content or getattr(delta, 'reasoning_content', None)):
            first_token_time = now

        if getattr(delta, "reasoning_content", None):
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

    print("\n================ Latency & Throughput Metrics ================")
    print(f"  - TTFT (Time to First Token): {ttft:.2f} ms")
    print(f"  - Decode Throughput (TPS): {tps:.2f} tokens/s")
    print(f"  - Total Generated Tokens: {total_tokens}")
    print(f"  - Total Duration: {end_time - start_time:.2f} s")
    print("===============================================================")
    
    print("\n=== Extracted Reasoning Chain (<think>) ===")
    print(reasoning_text if reasoning_text else "[Reasoning chain emitted in main content]")
    
    print("\n=== Final Response Content ===")
    print(content_text)

except Exception as e:
    print(f"❌ Inference benchmark error: {e}")
```

Typical output (measured on a single DGX Spark):
```text
================ Latency & Throughput Metrics ================
  - TTFT (Time to First Token): 172.00 ms
  - Decode Throughput (TPS): 34.87 tokens/s
  - Total Generated Tokens: 2048
  - Total Duration: 58.91 s
===============================================================
```

+++

#### Step 6.3: Tool Calling Test

Verify Function Calling with a weather inquiry tool schema:

```{code-cell}
tools_schema = [
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
    {"role": "user", "content": "What is the weather in Hangzhou right now?"}
]

try:
    tool_resp = client.chat.completions.create(
        model="ling-v3-flash-fp4",
        messages=messages,
        tools=tools_schema,
        temperature=0.0
    )

    message = tool_resp.choices[0].message
    if message.tool_calls:
        print("✅ Tool calling test passed! Model initiated a tool call:")
        for call in message.tool_calls:
            print(f"  - Function name: {call.function.name}")
            print(f"  - Arguments: {call.function.arguments}")
    else:
        print("Direct content:", message.content)

except Exception as e:
    print(f"❌ Tool calling verification failed: {e}")
```

Typical output:
```text
✅ Tool calling test passed! Model initiated a tool call:
  - Function name: get_current_weather
  - Arguments: {"city":"Hangzhou"}
```

+++

### Step 7: Troubleshooting & FAQ

1. **Unified Memory Exhaustion**:
   - Symptom: Server process freezes upon startup; driver throws `rmapiLockAcquire` and ceases responding.
   - Cause: JIT kernel compilation and model weight loading running concurrently, exceeding physical unified memory limits; or `--mem-fraction-static` set excessively high.
   - Solution: Precompile FlashInfer kernels as described in Step 4; maintain `--mem-fraction-static` around the recommended value of `0.68`.

2. **MTP Accept Length Dropping (Near 1.00) with Degraded Throughput**:
   - Symptom: Logs display `accept len: 1.00` and generation speed drops to ~24 tok/s.
   - Cause: When online FP8 LM Head is enabled, the draft model fails to share the target's quantized module, decoding weights as BF16 erroneously.
   - Solution: Verify that the specified branch `ling_v3_support_mxfp4_humming` is used (which includes the `set_lm_head_from_target` patch).

3. **Port Conflict (`Port 30000 occupied`)**:
   - Symptom: `bind: address already in use`.
   - Solution: Run `lsof -i :30000` to find and terminate existing processes, or change the port using `--port`.
