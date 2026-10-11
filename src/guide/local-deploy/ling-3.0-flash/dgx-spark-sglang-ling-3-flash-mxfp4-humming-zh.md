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

# Ling-3.0-flash on DGX Spark (SGLang MXFP4 Humming 优化) 部署指南

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

感谢 Nvidia 团队 [@ly01325](https://github.com/ly01325) 进行优化和提供本部署指南。

+++

`Ling-3.0-flash` 是总参数量 124B、单 Token 激活 5.1B 的混合注意力 (Hybrid Attention) MoE 大语言模型。

本 Notebook 演示如何在单台 **NVIDIA DGX Spark**（Grace Blackwell GB10 / SM121，121GB 统一内存）上，部署 Ling-3.0-flash 的 MXFP4 量化版本，并启用三项推理优化：

1. **Humming MoE 后端**：使用专为 Blackwell 架构优化的 Humming 算子后端；
2. **在线 FP8 LM Head**：将 BF16 的 LM Head 在线动态量化为 FP8（`SGLANG_ENABLE_FP8_LM_HEAD=1`），降低解码阶段的访存开销；
3. **MTP 投机解码**：利用模型原生自带的 Multi-Token Prediction 结构，配置 3 步投机解码（NEXTN）。

实测生成速率从基线的 37.8 tok/s 提升至约 53.8 tok/s。我们在 GSM8K 完整测试集上验证精度无损。

> [!TIP]
> **环境与硬件准备建议**：
> - 推荐使用 **Python 3.11**、CUDA 13、Ubuntu / DGX OS；
> - 推荐使用 **uv** 创建独立的 Python 虚拟环境，以确保依赖隔离；
> - 显存预分配参数 `--mem-fraction-static` 推荐设为 **0.68**（0.75 在统一内存高负载下存在触发 GPU OOM 导致驱动死锁风险）。

+++

### 步骤 1: 准备 Python 虚拟环境并克隆调优分支

由于包含 Humming 适配、在线 FP8 LM Head 以及 MTP draft 共享模块补丁，克隆已整合该优化的 SGLang 分支仓库（`inclusionAI/sglang` 的 `ling_v3_support_mxfp4_humming` 分支）：

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
!git clone -b ling_v3_support_mxfp4_humming https://github.com/inclusionAI/sglang.git
```

典型运行输出：
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
正克隆到 'sglang'...
remote: Enumerating objects: 225775, done.
remote: Total 225775 (delta 0), reused 0 (delta 0), pack-reused 225775 (from 1)
接收对象中: 100% (225775/225775), 189.01 MiB | 19.22 MiB/s, 完成.
处理 delta 中: 100% (172158/172158), 完成.
```

+++

### 步骤 2: 从源码安装 SGLang

本地推理不需要编译 gRPC、多模态及 Rust Server 扩展。

通过设置 `SGLANG_BUILD_RUST_EXTS=none` 跳过 crates.io 依赖树构建，可节约 30 分钟以上的安装耗时。

同时依据当前机器可用内存动态设置编译并行度（上限 8 核），避免内存耗尽：

```{code-cell}
!source .venv/bin/activate && \
export SGLANG_BUILD_RUST_EXTS=none && \
J=$(( $(nproc) < 8 ? $(nproc) : 8 )) && \
export MAX_JOBS="$J" NINJA_NUM_JOBS="$J" CMAKE_BUILD_PARALLEL_LEVEL="$J" && \
echo "使用并行度 MAX_JOBS=$J 编译，跳过 Rust 扩展..." && \
uv pip install -e "./sglang/python[all]"
```

典型运行输出：

```text
使用并行度 MAX_JOBS=8 编译，跳过 Rust 扩展...
Resolved 240 packages in 6.88s
      Built sglang @ file:///home/squall/sipan/sglang/python
      Built cuda-tile==1.6.0rc5
      Built xatlas==0.0.11
Prepared 112 packages in 39.15s
Uninstalled 1 package in 5ms
Installed 225 packages in 174ms
```

+++

### 步骤 3: 下载 Ling-3.0-flash FP4 (MXFP4) 模型权重

百灵已经提供预量化的 MXFP4 格式模型权重，约 60GB：

- [ModelScope Ling-3.0-flash-fp4](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-fp4)
- [Hugging Face Ling-3.0-flash-fp4](https://huggingface.co/inclusionAI/Ling-3.0-flash-fp4)

以下命令以 ModelScope 为例，将模型权重下载到 `~/models/Ling-3.0-flash-fp4`：

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-fp4 --local-dir ~/models/Ling-3.0-flash-fp4
```

典型运行输出：
```text
Downloading [model-00024-of-00024.safetensors]: 100%|█| 2.80G/2.80G [01:15<00:00]
Processing 35 items: 100%|███████████████████| 35.0/35.0 [09:40<00:00, 16.5s/it]
Successfully downloaded Ling-3.0-flash-fp4 to ~/models/Ling-3.0-flash-fp4
```

+++

### 步骤 4: 预编译 FlashInfer CUTLASS MXFP4 算子

本步骤用于将 FlashInfer CUTLASS MoE 算子提前编译为动态链接库（`.so`）。

在单台 DGX Spark（121GB 统一内存）上，如果在加载约 60GB 模型权重的过程中同时触发内联编译，编译进程与模型权重叠加，容易导致物理内存不足，进而触发进程异常终止。

下方的 Python 脚本将使用如下方案解决上述问题：

1. 用 SGLang 加载过程，生成 `build.ninja` 构建规则；
2. 规则生成后，不继续加载 SGLang 实例，而调用 `ninja` 完成算子编译。

建议将下方代码保存为独立脚本（如 `compile-kernel.py`），并在终端中执行：

```bash
uv run compile-kernel.py
```

等待编译完成（通常需要 6~8 分钟）。

这样，后续启动 SGLang 服务时将直接加载编译好的 `.so` 文件，无需重复编译，可避免内存占用过高导致异常退出。

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
    print(f"✅ FlashInfer CUTLASS 算子已完成编译，直接复用: {so}")
else:
    print("⏳ 未检测到预编译算子，启动临时探测以生成构建规则（约需 6~8 分钟）...")
    state = {"size": -1, "stable": 0}
    
    # 规则生成监听函数：文件大小连续三次无变化即判定生成完成
    def ninja_ready():
        bn = newest("build.ninja")
        if bn is None:
            return False
        size = bn.stat().st_size
        state["stable"] = state["stable"] + 1 if size == state["size"] else 0
        state["size"] = size
        return state["stable"] >= 3

    # 后台拉起临时服务触发构建规则生成（参数与正式运行完全对齐，确保 shape 规则一致）
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
                raise RuntimeError("临时探测服务异常退出，请检查环境输出")
    finally:
        # 规则一旦生成，立刻杀掉临时服务，完全释放 66GB 显存
        proc.terminate()
        proc.wait()
        subprocess.run(["pkill", "-f", "sglang.launch_server"], capture_output=True)
        time.sleep(3)
        print("💡 构建规则已生成，已释放显存占用，开始独占内存并行编译...")

    bn = newest("build.ninja")
    build_dir = bn.parent
    ninja_bin = shutil.which("ninja") or "ninja"

    # 读取可用内存并留足余量（每编译线程预留 10GB 内存）
    avail_gb = next(int(l.split()[1]) / 1048576 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    jobs = max(1, min(os.cpu_count() or 1, int(avail_gb // 10), 8))
    print(f"使用 -j{jobs} 并行编译 97 个算子目标...")

    ret = subprocess.run([ninja_bin, f"-j{jobs}", "-C", str(build_dir)])
    if ret.returncode != 0:
        raise RuntimeError("FlashInfer ninja 算子编译失败")
    print("✅ FlashInfer CUTLASS 算子独占编译完成！")
```

典型运行输出：
```text
[96/96] c++ /home/squall/.cache/flashinfer/0.6.17/121a/cached_ops/fused_moe_120/...e/squall/.cache/flashinfer/0.6.17/121a/cached_ops/fused_moe_120/fused_moe_120.so
✅ FlashInfer CUTLASS 算子独占编译完成！
```

+++

### 步骤 5: 启动 SGLang 推理服务

启动 SGLang HTTP 服务，提供 OpenAI 兼容的 API 端点。开启全部优化配置：

- `--moe-runner-backend humming`：启用 Blackwell 优化的 Humming MoE 算子；
- `SGLANG_ENABLE_FP8_LM_HEAD=1`：LM Head 在线动态量化为 FP8，减半访存带宽压力；
- `--speculative-algorithm NEXTN --speculative-num-steps 3`：启用 MTP=3 步投机解码；
- `--mem-fraction-static 0.68`：设置显存预分配基线，保障超长上下文稳定性。

> [!NOTE]
> `sglang.launch_server` 以前台常驻服务运行。如果直接执行下方单元，该单元将保持阻塞状态。建议你在独立终端会话中运行该命令。

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

服务端就绪时的典型日志：
```text
[2026-09-07 14:15:30] Online FP8 quantization enabled for lm_head.
[2026-09-07 14:16:10] Load weight end. elapsed=342.11 s, type=BailingMoeV3ForCausalLM
[2026-09-07 14:17:02] Capture draft decode CUDA graph begin...
[2026-09-07 14:18:20] The server is fired up and ready to roll!
```

+++

### 步骤 6: 验证服务、防静默降级检测与投机测速

服务启动后，可按下列方法进行验证：

+++

#### 步骤 6.1: 连通性与健康检查

请求 `GET /v1/models` 查看当前运行的模型信息：

```{code-cell}
import urllib.request
import json

url = "http://127.0.0.1:30000/v1/models"
headers = {"Authorization": "Bearer sk-ling-cookbook-test"}

try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=5) as response:
        body = response.read().decode('utf-8')
        print("✅ 服务健康检查通过！")
        print(f"Models response: {body}")
except Exception as e:
    print(f"❌ 连通性检查失败: {e}")
```

典型运行输出：
```text
✅ 服务健康检查通过！
Models response: {"object":"list","data":[{"id":"ling-v3-flash-fp4","object":"model","created":1788786583,"owned_by":"sglang","root":"ling-v3-flash-fp4","parent":null,"max_model_len":262144}]}
```

+++

#### 步骤 6.2: 流式请求、启用 Reasoning，并进行测速

使用 OpenAI Python SDK 进行流式请求，提取 `<think>` 思考链内容，并实时统计首字延迟 (TTFT) 与生成速率 (Decode TPS)：

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="sk-ling-cookbook-test"
)

prompt = "请推导公式 17 × 23 并详细给出计算步骤。"
print(f"发送测试请求: '{prompt}'...")

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

    print("\n================ 性能与吞吐指标 ================")
    print(f"  - 首字延迟 (TTFT): {ttft:.2f} ms")
    print(f"  - 生成速率 (Decode TPS): {tps:.2f} tokens/s")
    print(f"  - 生成总 Token 数量: {total_tokens}")
    print(f"  - 端到端完整耗时: {end_time - start_time:.2f} s")
    print("================================================")
    
    print("\n=== 提取的思考链 (<think>) ===")
    print(reasoning_text if reasoning_text else "[思考链已在正文中输出]")
    
    print("\n=== 最终回复内容 ===")
    print(content_text)

except Exception as e:
    print(f"❌ 推理测速异常: {e}")
```

典型测试输出（基于单台 DGX Spark 实测）：
```text
================ 性能与吞吐指标 ================
  - 首字延迟 (TTFT): 172.00 ms
  - 生成速率 (Decode TPS): 34.87 tokens/s
  - 生成总 Token 数量: 2048
  - 端到端完整耗时: 58.91 s
================================================
```

+++

#### 步骤 6.3: 测试工具调用

传入简单的天气查询工具 Schema，验证模型发起结构化调用的能力：

```{code-cell}
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "查询指定城市当天的实时天气与气温信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：杭州、北京、上海"
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

messages = [
    {"role": "user", "content": "请帮我查一下杭州现在的天气如何？"}
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
        print("✅ 工具调用检查通过！模型成功发起 Tool Call：")
        for call in message.tool_calls:
            print(f"  - 函数名称: {call.function.name}")
            print(f"  - 传入参数: {call.function.arguments}")
    else:
        print("直接返回内容:", message.content)

except Exception as e:
    print(f"❌ 工具调用验证失败: {e}")
```

典型输出：
```text
✅ 工具调用检查通过！模型成功发起 Tool Call：
  - 函数名称: get_current_weather
  - 传入参数: {"city":"杭州"}
```

+++

### 步骤 7: 常见问题

1. 统一内存耗尽
   - 现象：服务端启动时系统卡死，驱动抛出 `rmapiLockAcquire` 且无法响应任何命令。
   - 原因：FlashInfer 算子编译与模型加载并发执行，超过统一内存限制；或显存预分配 `--mem-fraction-static` 设定过高。
   - 解决：确保已完整执「预编译算子」；并将 `--mem-fraction-static` 设定为推荐的 `0.68` 左右。

2. MTP 接受长度（Accept Length）过低（接近 1.00）且吞吐下降
   - 现象：服务端日志中 `accept len: 1.00`，生成速度反而暴跌至 ~24 tok/s。
   - 原因：在线 FP8 LM Head 开启时，Draft 模型未成功共享 Target 的量化模块，导致权重按 BF16 错误解码。
   - 解决：确认使用的是指定的 `ling_v3_support_mxfp4_humming` 分支（包含 `set_lm_head_from_target` 补丁以修复上述问题）。

3. 端口冲突 (Port 30000 occupied)
   - 现象：提示 `bind: address already in use`。
   - 解决：执行 `lsof -i :30000` 查看并停掉旧的推理进程，释放端口后再启动。
