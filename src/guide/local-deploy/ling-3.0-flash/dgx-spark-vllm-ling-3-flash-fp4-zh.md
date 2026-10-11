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

# Ling-3.0-flash on DGX Spark (vLLM FP4) 部署指南

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

Ling-3.0-flash 是总参数量 124B、单 Token 激活 5.1B 的 MoE 大语言模型。

本 Notebook 演示如何在 NVIDIA DGX Spark 上，基于 vLLM 官方 ARM64 CUDA 13.0 预编译版本，部署 Ling-3.0-flash 的 FP4 (MXFP4) 量化版本。

> [!TIP]
> **环境准备建议**：
> - 推荐使用 **Python 3.12** 环境；
> - 推荐使用 **uv** 创建独立的 Python 虚拟环境，以确保依赖隔离与算子兼容性。

+++

### 步骤 1: 准备 Python 3.12 虚拟环境 (uv)

推荐使用 `uv` 创建独立的 Python 3.12 虚拟环境，确保依赖准确；同时安装 `openai` package 用于后续的 OpenAI 兼容接口调用。

```{code-cell}
!pip install -U uv
!uv venv --python 3.12 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
```

典型运行输出：
```text
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtualenv at: .venv
Activate with: source .venv/bin/activate
Resolved 20 packages in 120ms
Installed 20 packages in 45ms
 + openai==1.60.0
```

+++

### 步骤 2: 安装 vLLM 官方 ARM64 CUDA 13.0 预编译版本

Ling-3.0 架构与 `ling3` 解析器支持已合并至 vLLM 官方 Wheel 源。通过 `uv pip` 指定 `--torch-backend=cu130` 即可安装针对 DGX Spark GB10（sm_121 / cu130）构建的预编译包：

```{code-cell}
!uv pip install \
  --upgrade vllm \
  --torch-backend=cu130 \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130
```

典型运行输出：
```text
⠇ Resolving dependencies...
Installed 180 packages in 114ms
 + aiohappyeyeballs==2.7.1
 + aiohttp==3.14.3
```

+++

### 步骤 3: 下载 Ling-3.0-flash FP4 模型权重

从 ModelScope 或 Hugging Face 下载我们发布的 `inclusionAI/Ling-3.0-flash-fp4` 模型权重至本地目录。推荐使用 `modelscope` 或 `huggingface` 命令行工具进行下载。以下使用 `modelscope` 示例：

- [Ling-3.0-flash-fp4 on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-fp4)
- [Ling-3.0-flash-fp4 on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-fp4)

```{code-cell}
!uv pip install -U modelscope
!uv run modelscope download --model inclusionAI/Ling-3.0-flash-fp4 --local-dir ~/models/Ling-3.0-flash-fp4
```


典型运行输出：
```text
Downloading [model_files]... 100%|██████████| 62.4G/62.4G [02:15<00:00, 485MB/s]
Successfully downloaded to /home/ubuntu/models/Ling-3.0-flash-fp4
```

+++

### 步骤 4: 启动 vLLM 推理服务

启动 `vllm` 加载模型，并开启 OpenAI 兼容的 HTTP API 服务端。

⚠️注意： `vllm` 默认以前台常驻模式运行。如果在 Jupyter Notebook 单元格中直接运行，将持续占用执行 Kernel，无法操作其他单元格。你需要在独立终端会话中执行下面的启动命令。

```{code-cell}
!uv run vllm serve ~/models/Ling-3.0-flash-fp4 \
  --served-model-name ling-3.0-flash-fp4 \
  --port 30000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --reasoning-parser ling3 \
  --enable-auto-tool-choice \
  --tool-call-parser ling3 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --api-key sk-ling-cookbook-test \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

服务端就绪时的典型日志：

```text
(APIServer pid=1915283) INFO 08-16 16:08:46 [launcher.py:60] Route: /v1/completions/derender, Methods: POST
(APIServer pid=1915283) INFO 08-16 16:08:46 [launcher.py:60] Route: /inference/v1/generate, Methods: POST
(APIServer pid=1915283) INFO:     Started server process [1915283]
(APIServer pid=1915283) INFO:     Waiting for application startup.
(APIServer pid=1915283) INFO:     Application startup complete.
```

+++

### 步骤 5: 验证服务与接口调用

服务启动后，可通过 OpenAI 兼容客户端进行端到端验证：

1. 健康检查 - 确认模型端点连通性；
2. 流式 Reasoning 验证与速度测量 - 测试 `<think>` 思考链解析，并测算 TTFT 与生成速率；
3. Function Calling 测试 - 验证结构化工具调用支持。

```{code-cell}
import urllib.request

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
```
Health check status: 200
Models response: {"object":"list","data":[{"id":"ling-3.0-flash-fp4","object":"model","created":1786868316,"owned_by":"vllm","root":"/home/squall/models/Ling-3.0-flash-fp4","parent":null,"max_model_len":262144,"permission":[{"id":"modelperm-906b7e2582e9ef5e","object":"model_permission","created":1786868316,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-ling-cookbook-test"
)

def verify_streaming_and_thinking():
    prompt = "请推导公式 17 × 23 并详细给出计算步骤。"
    print(f"Sending prompt: '{prompt}' to model 'ling-3.0-flash-fp4'...")

    start_time = time.perf_counter()
    first_token_time = None
    first_content_time = None
    reasoning_chunks = []
    content_chunks = []
    usage_info = None

    try:
        response = client.chat.completions.create(
            model="ling-3.0-flash-fp4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=1024,
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

            # 获取 reasoning 思考链片段 (vLLM ling3 解析器直接返回 delta.reasoning 字段)
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
Sending prompt: '请推导公式 17 × 23 并详细给出计算步骤。' to model 'ling-3.0-flash-fp4'...

=== Latency & Throughput Metrics ===
TTFT (首个 Token 延迟): 256.67 ms
Time to First Content (首个正文 Token 延迟): 6075.82 ms
端到端总耗时 (Total Duration): 18.84 s
解码总耗时 (Decode Duration): 18.59 s
生成 Token 总数: 835 (思考链: 249 tokens, 正文: 586 tokens)
端到端平均吞吐 (Overall TPS): 44.32 tokens/s
解码生成速率 (Decode TPS): 44.93 tokens/s
```

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
    print(f"Testing Function Calling with model 'ling-3.0-flash-fp4'...")
    try:
        response = client.chat.completions.create(
            model="ling-3.0-flash-fp4",
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
            print("\n=== Direct Response ===")
            print(message.content)

    except Exception as e:
        print(f"Tool calling verification failed: {e}")

if __name__ == "__main__":
    verify_tool_calling()
```

典型测试结果：
```text
Testing Function Calling with model 'ling-3.0-flash-fp4'...

=== Function Call Output Detected ===
Tool Call ID: chatcmpl-tool-8ca899170cd1721d
Function Name: get_weather
Arguments JSON: {"city": "杭州", "unit": "celsius"}
```

+++

### 步骤 6: 常见问题与故障排查

1. **Nightly Wheel 下载与网络问题**：
   - 现象：`uv pip install` 提示找不到匹配的 ARM64 cu130 安装包。
   - 解决：确保指定 `--extra-index-url https://wheels.vllm.ai/nightly/cu130` 与 `--torch-backend=cu130`。

2. **显存预分配比例调整**：
   - 现象：启动服务时提示 CUDA Out of Memory 或 KV Cache 分配受限。
   - 解决：调整 `--gpu-memory-utilization`（例如从 0.90 降至 0.85），为系统和投机解码预留充足动态显存。

3. **端口冲突 (Port 30000 occupied)**：
   - 现象：服务端启动时报错 `Address already in use`。
   - 解决：通过 `lsof -i :30000` 查询占用进程并停止，或在启动参数中通过 `--port` 更换服务端口。
