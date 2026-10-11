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

# Ling-3.0-flash on DGX Spark (SGLang INT4 W4A16) 部署指南

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

本 Notebook 将演示如何在 `NVIDIA DGX Spark` (GB10 121GB 统一内存) 上通过源码构建 `SGLang` 部署 Ling-3.0-flash 的 INT4 (W4A16) 权重版本，并开启 `NextN` 投机采样 (Speculative Decoding) 加速。量化后模型权重仅占 ~72 GB，结合 NextN 算法可在单台 DGX Spark 上显著提升 Decode 阶段的生成吞吐并降低首字延迟。

> [!TIP]
> **环境准备建议**：
> - 推荐使用 **Python 3.11 / 3.12** 环境；
> - 推荐使用 **uv** 创建独立的 Python 虚拟环境，以确保依赖隔离与算子兼容性。

+++

### 步骤 1: 准备 Python 虚拟环境 (uv) 与克隆 SGLang 仓库

推荐使用 `uv` 创建独立的 Python 虚拟环境并安装 OpenAI 客户端。同时克隆 SGLang 官方主仓库（Ling-3.0-flash INT4 Marlin MoE 算子已由官方主干原生支持）：

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

在虚拟环境中以可编辑模式（Editable mode）安装分支代码及全量依赖库（`[all]`）。设置 `MAX_JOBS=4` 防止多核并发编译导致内存耗尽：

```{code-cell}
!source .venv/bin/activate && MAX_JOBS=4 uv pip install -e "./sglang/python[all]"
```

典型运行输出：
```text
Successfully built sglang cuda-tile xatlas
Installing collected packages: ....
```

+++

### 步骤 3: 下载 Ling-3.0-flash INT4 模型权重

官方直接提供了预量化的 INT4 格式模型权重：
- [Ling-3.0-flash-int4 on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-int4)
- [Ling-3.0-flash-int4 on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-int4)

如果你在中国，推荐使用 [ModelScope CLI](https://github.com/modelscope/modelscope/blob/master/README_zh.md) 下载；或者，你也可以使用 [Hugging Face CLI](https://huggingface.co/docs/hub/agents-cli) 下载权重至本地目录：

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-int4 --local-dir ~/models/Ling-3.0-flash-int4
```

典型运行输出：
```text
Downloading [model-00024-of-00024.safetensors]: 100%|█| 3.10G/3.10G [01:27<00:00
Processing 35 items: 100%|███████████████████| 35.0/35.0 [11:18<00:00, 19.4s/it]
Successfully downloaded Ling-3.0-flash-int4 to ~/models/Ling-3.0-flash-int4
```

+++

### 步骤 4: 启动 SGLang HTTP 推理服务 (开启 NextN 投机加速)

启动 SGLang Server，提供 OpenAI 兼容的 HTTP 接口。部分部署参数说明：
- `SGLANG_ENABLE_SPEC_V2=1`：开启 Speculative V2 高性能调度内核
- `--speculative-algorithm NEXTN`：启用 NextN 投机采样算法加速解码
- `--max-mamba-cache-size 32`：配置投机采样 Mamba Cache 缓冲大小
- `--mem-fraction-static 0.75`：显存预分配比例设为 75%，为 KV Cache 与投机分支预留充足空间
- `--tool-call-parser ling3 --reasoning-parser ling3`：启用 Ling-3.0 专用的工具调用与思考链解析器
- `--json-model-override-args`：配置 YaRN RoPE Scaling，原生支持 256K 超长上下文窗口

⚠️ 特别注意：

`sglang.launch_server` 以前台常驻模式运行。如果你直接在 Notebook 中运行下方单元格，Jupyter 将阻塞而无法执行后续单元格的代码。建议你在单独的终端会话中执行启动命令。启动成功后即可使用 Jupyter 进行后续验证。

```{code-cell}
!source .venv/bin/activate && \
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
python3 -m sglang.launch_server \
  --model-path ~/models/Ling-3.0-flash-int4 \
  --served-model-name ling-v3-flash-int4 \
  --host 0.0.0.0 --port 30000 \
  --api-key sk-ling-cookbook-test \
  --mem-fraction-static 0.75 --max-running-requests 8 \
  --tp-size 1 --chunked-prefill-size 8192 --allow-auto-truncate \
  --tool-call-parser ling3 --reasoning-parser ling3 --context-length 262144 \
  --speculative-algorithm NEXTN --max-mamba-cache-size 32 --enable-fp32-lm-head \
  --disable-shared-experts-fusion \
  --json-model-override-args '{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}'
```

服务端就绪时的典型日志：
```text
Triton kernel 'chunk_kda_fwd_kernel_inter_solve_fused' took 2.44 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
Prefill batch, #new-seq: 1, #new-token: 6, #cached-token: 0, full token usage: 0.00, mamba usage: 0.09, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 0.25
INFO:     127.0.0.1:58576 - "POST /generate HTTP/1.1" 200 OK
The server is fired up and ready to roll!
```

+++

### 步骤 5: 验证部署成功并使用模型服务

服务启动后，即可在各种 LLM 客户端中使用（如尚未安装 OpenAI 客户端库，可先执行 `pip install openai`）。下面提供的例子包含：

1. 健康检查 - 确认模型加载状态与端点连通性。
2. 流式 Reasoning 验证 - 测试 `<think>` 思考链正常生成，同时测算投机采样加速下的 TTFT 与 TPS 性能指标。
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
```
Health check status: 200
Models response: {"object":"list","data":[{"id":"ling-v3-flash-int4","object":"model","created":1786799113,"owned_by":"sglang","root":"ling-v3-flash-int4","parent":null,"max_model_len":262144}]}
```

+++

#### 步骤 5.2: 流式推理、Reasoning 和速度测量

使用 OpenAI Python SDK 调用接口，通过 `enable_thinking: True` 提取 `<think>` 思考链内容，并实时统计投机采样加速下的 TTFT 与 Decode TPS（全局首个请求需要 JIT 算子，TTFT 可能会长达数十秒。从第二个开始将恢复正常）：

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-ling-cookbook-test"
)

def verify_streaming_and_thinking():
    prompt = "请推导公式 17 × 23 并详细给出计算步骤。"
    print(f"Sending prompt: '{prompt}' to model 'ling-v3-flash-int4'...")
    
    start_time = time.time()
    first_token_time = None
    chunk_count = 0
    exact_completion_tokens = None
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model="ling-v3-flash-int4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            temperature=0.6,
            top_p=0.95,
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
            if first_token_time is None and (delta.content or (hasattr(delta, 'reasoning_content') and delta.reasoning_content)):
                first_token_time = now

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
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

        print("\n=== Latency & Throughput Metrics ===")
        print(f"TTFT (Time to First Token): {ttft:.2f} ms")
        print(f"Decode TPS (Tokens/s): {tps:.2f} t/s")
        print(f"Total Generated Tokens: {total_tokens} (Exact tokens from usage)")
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

典型测试结果：
```text
Sending prompt: '请推导公式 17 × 23 并详细给出计算步骤。' to model 'ling-v3-flash-int4'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 135.51 ms
Decode TPS (Tokens/s): 42.23 t/s
Total Generated Tokens: 726 (Exact tokens from usage)
Total Duration: 17.33 s

=== Extracted Reasoning Chain (<think>) ===
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
    print(f"Testing Function Calling / Tool Use with model 'ling-v3-flash-int4'...")
    try:
        response = client.chat.completions.create(
            model="ling-v3-flash-int4",
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

典型测试结果：
```text
Testing Function Calling / Tool Use with model 'ling-v3-flash-int4'...

=== Function Call Output Detected ===
Tool Call ID: call_87804adb3c954cb4b5dcc266
Function Name: get_weather
Arguments JSON: {"city": "杭州", "unit": "celsius"}
```

+++

### 步骤 6: 常见问题与故障排查

1. **首个请求 JIT 算子编译延迟**：
   - 现象：首次发送推理请求时 TTFT 耗时较长（可能达数十秒），日志提示 `Triton kernel took X.XX s to compile`。
   - 说明：此为 Triton JIT 正常现象，从第二个请求开始将直接复用缓存编译的算子，速度将恢复正常。

2. **源码构建阶段内存耗尽 (OOM)**：
   - 现象：在执行 `MAX_JOBS=4 uv pip install -e "./sglang/python[all]"` 编译 C++/CUDA 扩展时进程被系统强制终止。
   - 解决：通过 `MAX_JOBS=4` 或 `MAX_JOBS=2` 限制并行编译线程数。

3. **端口冲突 (Port 30000 occupied)**：
   - 现象：服务端启动时报错 `Address already in use`。
   - 解决：通过 `lsof -i :30000` 查询占用进程并停止，或在启动参数中修改 `--port`。
