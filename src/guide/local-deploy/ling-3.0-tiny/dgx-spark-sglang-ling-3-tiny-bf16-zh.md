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

# Ling-3.0-tiny on DGX Spark (SGLang BF16) 部署指南

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

Ling-3.0-tiny 是总参数量 7.9B、单 Token 激活仅 1.3B 的轻量型 Sparse MoE 大语言模型。模型采用 KDA 线性注意力与 Gated MLA 混合架构，配备 128 个路由专家，原生支持 128K 上下文。

本 Notebook 将演示如何在 `NVIDIA DGX Spark` 上通过源码构建 `SGLang` 部署 Ling-3.0-tiny 的全精度 BF16 权重版本，提供高吞吐与低延迟的 API 服务。

> [!TIP]
> **环境准备建议**：
> - 推荐使用 **Python 3.11 / 3.12** 环境；
> - 推荐使用 **uv** 创建独立的 Python 虚拟环境，以确保依赖隔离与算子兼容性。

+++

### 步骤 1: 准备 Python 虚拟环境 (uv) 与克隆 SGLang 仓库

推荐使用 `uv` 创建独立的 Python 虚拟环境并安装 OpenAI 客户端。同时克隆 SGLang 官方主仓库（Ling-3.0-tiny 模型架构已由官方主干原生支持）：

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
!git clone https://github.com/sgl-project/sglang.git
```

典型运行输出：
```text
Using CPython 3.11 interpreter at: /usr/bin/python3.11
Creating virtualenv at: .venv
Cloning into 'sglang'...
remote: Enumerating objects: 38200, done.
Resolving deltas: 100% (28500/28500), done.
```

+++

### 步骤 2: 从源码安装 SGLang 与运行时全量依赖

在虚拟环境中以可编辑模式安装分支代码及全量依赖库（`[all]`）。设置 `MAX_JOBS=4` 防止多核并发编译导致内存耗尽：

```{code-cell}
!source .venv/bin/activate && MAX_JOBS=4 uv pip install -e "./sglang/python[all]" --quiet
```

典型运行输出：
```text
Successfully built sglang cuda-tile xatlas
Installing collected packages: ....
Successfully installed sglang
```

+++

### 步骤 3: 下载 Ling-3.0-tiny 官方全精度模型权重

官方提供了原生 BF16 Safetensors 格式权重：
- [Ling-3.0-tiny on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny)
- [Ling-3.0-tiny on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-tiny)

推荐使用 ModelScope CLI 一键下载至本地目录 `~/models/Ling-3.0-tiny`（共 4 个分卷约 15.8 GB，通常 1~2 分钟内完成）：

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope --quiet
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-tiny --local-dir ~/models/Ling-3.0-tiny
```

典型运行输出：
```text
Downloading [config.json, model.safetensors.index.json, ...]
Downloading shard 1/4: 100%|██████████| 4.95G/4.95G [00:10<00:00, 480MB/s]
Downloading shard 2/4: 100%|██████████| 4.98G/4.98G [00:10<00:00, 492MB/s]
Downloading shard 3/4: 100%|██████████| 4.92G/4.92G [00:10<00:00, 475MB/s]
Downloading shard 4/4: 100%|██████████| 1.02G/1.02G [00:02<00:00, 460MB/s]
Successfully downloaded Ling-3.0-tiny to ~/models/Ling-3.0-tiny
```

+++

### 步骤 4: 启动 SGLang HTTP 推理服务

启动 SGLang Server，提供 OpenAI 兼容的 HTTP 接口。

部分优化参数说明：

- `--cuda-graph-backend-decode full --cuda-graph-bs-decode 1 2`：启用 CUDA Graph 全图捕获，降低调度与通信开销；
- `--moe-runner-backend triton`：使用 Triton MoE 专家路由算子；
- `--json-model-override-args '{"num_nextn_predict_layers":0}'`：关闭多余的 MTP 辅助层开销；
- `--mem-fraction-static 0.35`：显存预分配 35%（约 42 GiB），模型权重仅占 ~16 GiB，为 128K 上下文与并发留出缓存；
- `--tool-call-parser ling3 --reasoning-parser ling3`：启用 Ling-3.0 工具调用与思考链解析；

⚠️ 特别注意：

`sglang.launch_server` 以前台常驻模式运行。如果你直接在 Notebook 中运行下方单元格，Jupyter 将阻塞而无法执行后续单元格的代码。建议你在单独的终端会话中执行启动命令。启动成功后即可使用 Jupyter 进行后续验证。

```{code-cell}
!source .venv/bin/activate && \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
python3 -m sglang.launch_server \
  --model-path ~/models/Ling-3.0-tiny \
  --served-model-name Ling-3.0-tiny-bf16 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key sk-ling-cookbook-test \
  --mem-fraction-static 0.35 \
  --max-running-requests 16 \
  --max-mamba-cache-size 64 \
  --chunked-prefill-size 8192 \
  --page-size 64 \
  --context-length 131072 \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 2 \
  --cuda-graph-bs-decode 1 2 \
  --cuda-graph-backend-prefill disabled \
  --random-seed 308534008 \
  --reasoning-parser ling3 \
  --tool-call-parser ling3 \
  --attention-backend flashinfer \
  --disable-flashinfer-autotune \
  --moe-runner-backend triton \
  --enable-fp32-lm-head \
  --json-model-override-args '{"num_nextn_predict_layers":0}'
```

服务端就绪时的典型日志：
```text
Multi-thread loading shards: 100% Completed | 32/32 [01:39<00:00,  3.12s/it]
[2026-08-17 16:52:10] Load weight end. elapsed=100.59 s, type=BailingMoeV3ForCausalLM, avail mem=97.27 GB, mem usage=15.47 GB.
[2026-08-17 16:52:12] Mamba Cache is allocated. max_mamba_cache_size: 614, conv_state size: 0.38GB, ssm_state size: 10.81GB
[2026-08-17 16:52:13] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 1933682, KV size: 12.45 GB
[2026-08-17 16:52:13] Linear attention kernel backend: decode=triton, prefill=triton, verify=triton
[2026-08-17 16:52:13] KDA kernel dispatcher: decode=TritonKDAKernel, verify=TritonKDAKernel, extend=TritonKDAKernel packed_decode=True
[2026-08-17 16:52:13] Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, bs=[1, 2], avail mem=73.10 GB
Capturing batches (bs=1 avail_mem=72.47 GB): 100%|██████████| 2/2 [00:02<00:00,  1.10s/it]
[2026-08-17 16:52:18] Capture target decode CUDA graph end. elapsed=2.58 s, mem usage=0.45 GB, avail mem=72.65 GB.
[2026-08-17 16:52:18] INFO:     Started server process [1963736]
[2026-08-17 16:52:18] INFO:     Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
[2026-08-17 16:52:41] INFO:     127.0.0.1:37664 - "POST /generate HTTP/1.1" 200 OK
[2026-08-17 16:52:41] The server is fired up and ready to roll!
```

+++

### 步骤 5: 验证部署成功并使用模型服务

服务启动后，即可在各种 LLM 客户端中使用（如尚未安装 OpenAI 客户端库，可先执行 `pip install openai requests`）。下面提供的例子包含：

1. 健康检查 - 确认模型加载状态与端点连通性。
2. 流式 Reasoning 验证 - 测试 `<think>` 思考链正常生成，同时测算 TTFT 与 TPS 性能指标。
3. Tool Calling 验证 - 测试 Function Calling 结构化工具调用是否可正常使用。

+++

#### 步骤 5.1: 连通性与模型健康检查

请求 `GET /v1/models` 查看当前运行的模型信息：

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

典型输出：
```text
Health check status: 200
Models response: {"object":"list","data":[{"id":"Ling-3.0-tiny-bf16","object":"model","created":1786898113,"owned_by":"sglang","root":"Ling-3.0-tiny-bf16","parent":null,"max_model_len":131072}]}
```

+++

#### 步骤 5.2: 流式推理、Reasoning 和速度测量

使用 OpenAI Python SDK 调用接口，通过 `enable_thinking: True` 提取 `<think>` 思考链内容，并实时统计 TTFT 与 Decode TPS：

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-ling-cookbook-test"
)

def verify_streaming_and_thinking():
    prompt = "请推导公式 25 × 48 并详细给出计算步骤。"
    print(f"Sending prompt: '{prompt}' to model 'Ling-3.0-tiny-bf16'...")

    start_time = time.perf_counter()
    first_token_time = None
    first_content_time = None
    reasoning_chunks = []
    content_chunks = []
    usage_info = None

    try:
        response = client.chat.completions.create(
            model="Ling-3.0-tiny-bf16",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=2048,
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

            # 获取 reasoning 思考链片段 (兼容 vLLM/SGLang 的多种返回格式)
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

        # 测算延迟与生成速率
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
        print(f"TTFT (首个 Token 延迟): {ttft_ms:.2f} ms")
        if first_content_time:
            print(f"Time to First Content (首个正文 Token 延迟): {ttf_content_ms:.2f} ms")
        print(f"端到端总耗时 (Total Duration): {total_duration:.2f} s")
        print(f"解码总耗时 (Decode Duration): {decode_duration:.2f} s")
        if reasoning_tokens is not None:
            print(f"生成 Token 总数: {total_tokens} (思考链: {reasoning_tokens} tokens, 正文: {content_tokens} tokens)")
        else:
            print(f"生成 Token 总数: {total_tokens}")
        print(f"端到端平均吞吐 (Overall TPS): {overall_tps:.2f} tokens/s")
        print(f"解码生成速率 (Decode TPS): {decode_tps:.2f} tokens/s")

        print("\n=== Extracted Reasoning Chain (思考链) ===")
        print(reasoning_text if reasoning_text else "[无独立思考过程或已直接输出]")

        print("\n=== Final Response Content (最终正文) ===")
        print(final_content)

    except Exception as e:
        print(f"Streaming verification failed: {e}")

if __name__ == "__main__":
    verify_streaming_and_thinking()
```

典型测试结果：
```text
Sending prompt: '请推导公式 25 × 48 并详细给出计算步骤。' to model 'Ling-3.0-tiny-bf16'...

=== Latency & Throughput Metrics ===
TTFT (首个 Token 延迟): 42.80 ms
Time to First Content (首个正文 Token 延迟): 3120.50 ms
端到端总耗时 (Total Duration): 5.25 s
解码总耗时 (Decode Duration): 5.21 s
生成 Token 总数: 468 (思考链: 280 tokens, 正文: 188 tokens)
端到端平均吞吐 (Overall TPS): 79.95 tokens/s
解码生成速率 (Decode TPS): 80.65 tokens/s

=== Extracted Reasoning Chain (思考链) ===
...
```

+++

#### 步骤 5.3: 测试 Function Calling 和结构化输出

传入标准 Function Calling Schema（以天气查询为例），验证模型对结构化工具调用的支持：

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
    print(f"Testing Function Calling / Tool Use with model 'Ling-3.0-tiny-bf16'...")
    try:
        response = client.chat.completions.create(
            model="Ling-3.0-tiny-bf16",
            messages=[
                {"role": "user", "content": "请帮我查一下杭州今天的天气如何？"}
            ],
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

if __name__ == "__main__":
    verify_tool_calling()
```

典型测试结果：
```text
Testing Function Calling / Tool Use with model 'Ling-3.0-tiny-bf16'...

=== Function Call Output Detected ===
Tool Call ID: call_f3861ca366004a94b162e8b1
Function Name: get_weather
Arguments JSON: {"city": "杭州", "unit": "celsius"}
```

+++

### 步骤 6: 常见问题与故障排查

1. **`sglang.serve` 时 MLA 断言报错 (`AssertionError: K must be a multiple of 1024, got 1536`)**：
   - 原因：Blackwell 架构上 SGLang 默认将 MLA Fused A GEMM 路由至 CuteDSL 后端，而 CuteDSL 底层要求维度 K 必须是 1024 整数倍，Ling-3.0-tiny 的 K=1536 会触发断言。
   - 修复：修改 `sglang/python/sglang/kernels/ops/gemm/fused_a_gemm.py` 中 `fused_a_gemm_weight_eligible` 的检查，将 `layer.weight.shape[1] % 256 == 0` 修正为 `layer.weight.shape[1] % 1024 == 0`，使小模型安全回退至标准 PyTorch/CUDA GEMM 算子（官方分支已包含该修复）。

2. **源码构建阶段内存耗尽 (OOM)**：
   - 现象：在编译 C++/CUDA 扩展时进程被系统终止。
   - 解决：通过 `MAX_JOBS=4` 环境变量限制并发编译线程数。

3. **端口冲突 (Port 30000 occupied)**：
   - 现象：服务端启动时报错 `Address already in use`。
   - 解决：通过 `lsof -i :30000` 查询占用进程并停止，或在启动参数中修改 `--port`。

