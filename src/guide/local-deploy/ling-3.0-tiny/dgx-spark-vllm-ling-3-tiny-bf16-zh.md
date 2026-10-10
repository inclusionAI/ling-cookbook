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

+++

# Ling-3.0-tiny on DGX Spark (vLLM BF16) 部署指南

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

Ling-3.0-tiny 是总参数量 7.9B、单 Token 激活仅 1.3B 的轻量型 Sparse MoE 大语言模型。模型采用 KDA 线性注意力 + MLA 混合注意力架构，在保持轻量级计算开销的同时具备强大的长文本理解与推理能力。

本 Notebook 介绍如何在 NVIDIA DGX Spark 上，基于 vLLM 进行全精度 BF16 服务部署。

> [!TIP]
> **环境准备建议**：
> - 推荐使用 **Python 3.12** 环境；
> - 推荐使用 **uv** 创建独立的 Python 虚拟环境，以确保依赖隔离与算子兼容性。

+++

### 步骤 1: 准备 Python 3.12 虚拟环境 (uv)

推荐使用 `uv` 创建独立的 Python 3.12 虚拟环境，确保依赖精准；同时安装 `openai` SDK 用于后续接口调用与测速验证：

```{code-cell}
!pip install -U uv --quiet
!uv venv --python 3.12 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0' requests
```

典型运行输出：
```text
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtualenv at: .venv
Activate with: source .venv/bin/activate
Resolved 20 packages in 110ms
Installed 20 packages in 40ms
 + openai==1.60.0
```

+++

### 步骤 2: 安装 vLLM 官方 ARM64 CUDA 13.0 预编译包

Ling-3.0 架构与 `ling3` 解析器支持已由官方发布至专用 Wheel 仓库。通过 `uv pip` 指定 `--torch-backend=cu130` 安装针对 GB10（sm_121 / cu130）构建的预编译包：

```{code-cell}
!uv pip install \
  --upgrade vllm \
  --torch-backend=cu130 \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130
```

典型运行输出：
```text
⠇ Resolving dependencies...
Installed 180 packages in 120ms
 + vllm==0.19.1+cu130
```

+++

### 步骤 3: 下载 Ling-3.0-tiny BF16 官方模型权重

官方提供了原生 BF16 Safetensors 格式权重：
- [Ling-3.0-tiny on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny)
- [Ling-3.0-tiny on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-tiny)

使用 `modelscope` CLI 工具下载至本地 `~/models/Ling-3.0-tiny` 目录（共 4 个分卷约 15.8 GB，通常 1~2 分钟完成）：

```{code-cell}
!uv pip install -U modelscope --quiet
!uv run modelscope download --model inclusionAI/Ling-3.0-tiny --local-dir ~/models/Ling-3.0-tiny
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

### 步骤 4: 启动 vLLM HTTP 推理服务

启动 vLLM Server，提供标准 OpenAI 兼容 HTTP 接口。关键运行参数说明：
- `--trust-remote-code`：信任百灵 MoE 混合架构模型定义与计算图；
- `--dtype bfloat16`：以 BF16 全精度加载权重与执行推理；
- `--gpu-memory-utilization 0.35`：显存预分配 35%（约 42 GiB），为 128K 超长上下文留出充足 KV 缓存；
- `--max-model-len 131072`：启用 Tiny 模型原生的 128K 上下文窗口；
- `--reasoning-parser ling3`：启用 Ling-3.0 专用的思考链解析器，自动将 `<think>` 标签提取至 `reasoning_content`；
- `--enable-auto-tool-choice --tool-call-parser ling3`：启用百灵专用工具调用解析器并开启自动工具路由（Tool Calling / Function Calling）；
- `--port 30000 --api-key sk-ling-cookbook-test`：指定对外服务端口与鉴权秘钥。

⚠️ 特别注意：

`vllm serve` 以前台常驻模式运行。如果你直接在 Notebook 中运行下方单元格，Jupyter 将阻塞而无法执行后续单元格的代码。建议你在单独的终端会话中执行启动命令。启动成功后即可使用 Jupyter 进行后续验证。

```{code-cell}
!uv run vllm serve ~/models/Ling-3.0-tiny \
  --served-model-name Ling-3.0-tiny-bf16 \
  --trust-remote-code \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key sk-ling-cookbook-test \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.35 \
  --max-num-seqs 16 \
  --reasoning-parser ling3 \
  --enable-auto-tool-choice \
  --tool-call-parser ling3
```

服务端就绪时的典型日志：
```text
INFO 08-17 17:35:10 [server.py:120] Route: /v1/chat/completions, Methods: POST
INFO 08-17 17:35:10 [server.py:120] Route: /v1/models, Methods: GET
INFO 08-17 17:35:12 [model_runner.py:1105] Loading model weights took 14.82 GB memory.
INFO 08-17 17:35:14 [worker.py:245] Memory profiling results: total_gpu_memory=121.00GiB, non_torch_memory=1.20GiB, torch_peak_memory=15.20GiB, available_memory=104.60GiB, kv_cache_memory=27.15GiB.
INFO 08-17 17:35:15 [launcher.py:28] Application startup complete.
INFO 08-17 17:35:15 [launcher.py:29] Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
```

+++

### 步骤 5: 验证部署成功并使用模型服务

服务启动后，即可在各类 LLM 客户端与应用中无缝调用。下面提供的验证测试包括：

1. **健康检查** - 确认模型加载状态与端点连通性。
2. **流式 Reasoning 验证** - 测试 `<think>` 思考链正常生成，同时测算 TTFT 与 Decode TPS 性能指标。
3. **Tool Calling 验证** - 测试 Function Calling 结构化工具调用是否可正常触发。

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
        body = response.read().decode("utf-8")
        print(f"Health check status: {status_code}")
        print(f"Models response: {body}")
except Exception as e:
    print(f"Health check error: {e}")
```

典型输出：
```text
Health check status: 200
Models response: {"object":"list","data":[{"id":"Ling-3.0-tiny-bf16","object":"model","created":1786968363,"owned_by":"vllm","root":"/home/squall/models/Ling-3.0-tiny","parent":null,"max_model_len":131072,"permission":[{"id":"modelperm-b16332c94d7e25c9","object":"model_permission","created":1786968363,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
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
TTFT (首个 Token 延迟): 86.34 ms
Time to First Content (首个正文 Token 延迟): 6054.91 ms
端到端总耗时 (Total Duration): 10.82 s
解码总耗时 (Decode Duration): 10.74 s
生成 Token 总数: 842 (思考链: 467 tokens, 正文: 375 tokens)
端到端平均吞吐 (Overall TPS): 77.79 tokens/s
解码生成速率 (Decode TPS): 78.41 tokens/s

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
Tool Call ID: call_87804adb3c954cb4b5dcc266
Function Name: get_weather
Arguments JSON: {"city": "杭州", "unit": "celsius"}
```

+++

### 步骤 6: 常见问题与故障排查

1. **显存预分配与 128K 上下文**：
   - 说明：Ling-3.0-tiny BF16 权重仅占 ~16 GiB，默认 `--gpu-memory-utilization 0.35`（约 42 GiB）即可支持完整的 128K 上下文与多路并发。如需更大并发可上调此比例。

2. **端口冲突 (Port 30000 occupied)**：
   - 现象：服务端启动时报错 `Address already in use`。
   - 解决：通过 `lsof -i :30000` 查询占用进程并停止，或在启动参数中修改 `--port`。

