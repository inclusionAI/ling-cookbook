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

# Ling-3.0-tiny on DGX Spark (llama.cpp BF16 GGUF) 部署指南

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


`Ling-3.0-tiny` 是总参数量 7.9B、单 Token 激活仅 1.3B 的轻量型 Sparse MoE 大语言模型。

本 Notebook 将采用 `llama.cpp`，在 `NVIDIA DGX Spark` 设备上对其进行全精度 BF16 GGUF 部署。权重文件约 16 GB，可在单台 DGX Spark 上全精度运行，提供高速低延迟的推理服务。

> [!TIP]
> **环境准备建议**：
> - 推荐使用 **Python 3.11 / 3.12** 环境；
> - 推荐使用 **uv** 创建独立的 Python 虚拟环境，以确保依赖隔离与工具链兼容性。

+++

### 步骤 1: 准备 Python 虚拟环境 (uv) 与克隆 llama.cpp 仓库

推荐使用 `uv` 创建独立虚拟环境并安装 OpenAI 客户端。`llama.cpp` 已合并对 Ling-3.0 架构的支持，直接克隆主干源码即可：

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0' requests
!git clone https://github.com/ggerganov/llama.cpp.git
```

典型输出：
```text
Using CPython 3.11 interpreter at: /usr/bin/python3.11
Creating virtualenv at: .venv
Cloning into 'llama.cpp'...
remote: Enumerating objects: 45210, done.
remote: Counting objects: 100% (210/210), done.
```

+++

### 步骤 2: 构建 llama.cpp (开启 CUDA 加速)

在 DGX Spark 上编译 `llama.cpp` 主干，启用 CUDA 硬件加速：

```{code-cell}
!cd llama.cpp && cmake -B build -DGGML_CUDA=ON . && cmake --build build --parallel 8
```

典型输出：
```text
-- The CUDA compiler identification is NVIDIA 13.0.88
-- Building with CUDA architecture: native (sm_121)
[100%] Built target llama-server
```

+++

### 步骤 3: 下载 Ling-3.0-tiny 官方原始权重 (Shell 快速下载)

直接使用 `modelscope` CLI 下载官方 Safetensors 权重至本地目录 `~/models/Ling-3.0-tiny`（共约 15.8 GB，通常 1~2 分钟内完成）：

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope --quiet
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-tiny --local-dir ~/models/Ling-3.0-tiny
```

典型输出：
```text
Downloading [config.json, model.safetensors.index.json, ...]
Downloading shard 1/4: 100%|██████████| 4.95G/4.95G [00:10<00:00, 480MB/s]
Downloading shard 2/4: 100%|██████████| 4.98G/4.98G [00:10<00:00, 492MB/s]
Downloading shard 3/4: 100%|██████████| 4.92G/4.92G [00:10<00:00, 475MB/s]
Downloading shard 4/4: 100%|██████████| 1.02G/1.02G [00:02<00:00, 460MB/s]
```

+++

### 步骤 4: 将模型转换为全精度 (BF16) GGUF 格式

使用 `convert_hf_to_gguf.py` 转换模型。由于 Tiny 模型体积轻量，**转换完成后可直接用于全精度推理，完全无需耗时的 `llama-quantize` 二次量化**：

```{code-cell}
!source .venv/bin/activate && uv pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt --quiet
!source .venv/bin/activate && python3 ./llama.cpp/convert_hf_to_gguf.py ~/models/Ling-3.0-tiny \
  --outfile ~/models/Ling-3.0-tiny-bf16.gguf \
  --outtype bf16 --model-name Ling-3.0-tiny
```

典型输出：
```text
INFO:hf-to-gguf:Loading model: Ling-3.0-tiny
INFO:hf-to-gguf:Set model parameters
INFO:hf-to-gguf:Writing tensors to /home/squall/models/Ling-3.0-tiny-bf16.gguf
INFO:hf-to-gguf:Done. Output file: /home/squall/models/Ling-3.0-tiny-bf16.gguf (15.82 GiB)
```

+++

### 步骤 5: 用 llama-server 启动模型推理服务

启动 `llama-server`，提供 OpenAI 格式兼容的 HTTP 接口。部分部署参数说明：

- `-m ~/models/Ling-3.0-tiny-bf16.gguf`：加载全精度 BF16 GGUF 模型权重
- `--alias Ling-3.0-tiny-bf16`：指定别名，方便下游客户端调用
- `-ngl all`：将所有网络层部署至 GPU 运算（显存占用约 18 GB）
- `-fa on`：开启 FlashAttention 加速
- `-c 131072`：启用 Tiny 模型原生的 128K 上下文窗口
- `-cb -np 4`：开启连续批处理（Continuous Batching），支持 4 路并发处理
- `--port 9102`：统一选择端口 `9102`，避免与默认 8080 端口冲突
- `--api-key sk-ling-cookbook-test`：配置鉴权秘钥

> [!WARNING]
> **特别注意：前台常驻与阻塞**
> `llama-server` 以前台常驻模式运行。如果你直接在 Notebook 中运行下方单元格，Jupyter 将阻塞而无法执行后续验证单元格。
> 建议你在**单独的终端会话**中执行启动命令，或使用后台命令启动。启动成功后即可在 Notebook 中执行后续测试。

```{code-cell}
!./llama.cpp/build/bin/llama-server \
  -m ~/models/Ling-3.0-tiny-bf16.gguf \
  --alias Ling-3.0-tiny-bf16 \
  -ngl all -fa on -c 131072 -cb -np 4 \
  --host 0.0.0.0 --port 9102 --api-key sk-ling-cookbook-test
```

服务端就绪时的典型日志：
```text
llama_server: HTTP server listening on 0.0.0.0:9102
llama_server: model loaded successfully, n_ctx = 131072, offload = 100% (GPU)
```

+++

### 步骤 6: 验证部署成功并使用模型服务

服务启动后，即可在各种 LLM 客户端中使用（如尚未安装 OpenAI 客户端库，可先执行 `pip install openai requests`）。

下面提供完整的验证测试：
1. **健康检查** - 确认模型加载状态与端点连通性。
2. **流式 Reasoning 验证** - 测试 `<think>` 思考链生成，同时测算 TTFT 与 TPS 性能指标。
3. **Tool Calling 验证** - 测试 Function Calling 结构化输出。

+++

#### 步骤 6.1: 连通性与模型健康检查

请求 `GET /v1/models` 查看当前运行的模型信息：

```{code-cell}
import urllib.request

url = "http://localhost:9102/v1/models"
try:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer sk-ling-cookbook-test"})
    with urllib.request.urlopen(req) as response:
        print(f"Health check status: {response.getcode()}")
        print(f"Models response: {response.read().decode('utf-8')}")
except Exception as e:
    print(f"Health check error: {e}")
```

典型输出：
```text
Health check status: 200
Models response: {"models":[{"name":"Ling-3.0-tiny-bf16","model":"Ling-3.0-tiny-bf16",...}]}
```

+++

#### 步骤 6.2: 流式推理、Reasoning 与速度测量 (TTFT / TPS)

使用 OpenAI Python SDK 调用该接口，通过 `enable_thinking: True` 提取 `<think>` 思考链内容，并统计 TTFT 与 TPS：

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9102/v1",
    api_key="sk-ling-cookbook-test"
)

prompt = "请推导公式 25 × 48 并详细给出计算步骤。"
print(f"Sending prompt: '{prompt}' to model 'Ling-3.0-tiny-bf16'...")

start_time = time.time()
first_token_time = None
total_tokens = 0
reasoning_text = ""
content_text = ""

try:
    response = client.chat.completions.create(
        model="Ling-3.0-tiny-bf16",
        messages=[{"role": "user", "content": prompt}],
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
            print(delta.content, end="", flush=True)
            total_tokens += 1

    end_time = time.time()
    ttft = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
    decode_duration = end_time - first_token_time if first_token_time else 0.001
    tps = total_tokens / decode_duration

    print("")
    print("=== Latency & Throughput Metrics ===")
    print(f"TTFT (Time to First Token): {ttft:.2f} ms")
    print(f"Decode TPS (Tokens/s): {tps:.2f} t/s")
    print(f"Total Generated Tokens: {total_tokens}")
    print(f"Total Duration: {end_time - start_time:.2f} s")

    if reasoning_text:
        print("")
        print("=== Extracted Reasoning Chain (<think>) ===")
        print(reasoning_text)

except Exception as e:
    print(f"Streaming verification failed: {e}")
```

典型测试结果：
```text
=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 58.39 ms
Decode TPS (Tokens/s): 85.21 t/s
Total Generated Tokens: 1315
Total Duration: 15.49 s

=== Extracted Reasoning Chain (<think>) ===
```

+++

#### 步骤 6.3: 测试 Function Calling 和结构化输出

传入标准 Function Calling Schema（以天气查询为例），验证模型对结构化工具调用的支持：

```{code-cell}
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

print("Testing Function Calling / Tool Use with model 'Ling-3.0-tiny-bf16'...")
try:
    response = client.chat.completions.create(
        model="Ling-3.0-tiny-bf16",
        messages=[{"role": "user", "content": "请帮我查一下杭州今天的天气如何？"}],
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
```

典型测试结果：
```text
Testing Function Calling / Tool Use with model 'Ling-3.0-tiny-bf16'...

=== Function Call Output Detected ===
Tool Call ID: pa6MtfWs1LLBkrFJpXaFnjlXKxU7oTi2
Function Name: get_weather
Arguments JSON: {"city":"杭州"}
```

+++

### 步骤 7: 常见问题与故障排查

1. **GGUF 转换依赖安装**：
   - 现象：执行 `convert_hf_to_gguf.py` 提示找不到模块。
   - 解决：通过 `uv pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt` 安装所需依赖。

2. **端口冲突 (Port 9102 occupied)**：
   - 现象：`llama-server` 启动报错端口已被占用。
   - 解决：通过 `lsof -i :9102` 查找并结束进程，或在启动命令中通过 `--port` 修改端口。

