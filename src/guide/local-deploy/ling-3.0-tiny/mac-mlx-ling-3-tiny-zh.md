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

##### Copyright 2026 Ant Group.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

+++

# Ling-3.0-tiny on Apple Silicon Mac (MLX-LM) 部署指南

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

`Ling-3.0-tiny` 是百灵大模型系列中的 7.9B 轻量 Sparse MoE 语言模型，单 Token 激活参数量仅 1.3B，原生支持 128K 长上下文。

MLX 是 Apple 官方专为 Apple Silicon 统一内存架构打造的原生机器学习框架，具备轻量、低抽象开销与直接调用底层 Metal 硬件算子的优势。

针对百灵大模型，`mlx-lm` 项目已在主干（PR #1711）原生合入了对 Ling-3.0 架构（`bailing_hybrid` / `BailingMoeV3ForCausalLM`）的支持。

本指南将介绍从安装、下载、验证到启动 OpenAI 兼容 HTTP 服务的流程；同时针对内存受限设备（8GB / 16GB），提供转换模型权重为 4-bit / MXFP8 的方案。

---

### 设备内存门槛与选型矩阵 (Hardware Matrix)

| 部署规格       | 权重格式     | 存储体积  | 8K 上下文推荐内存 | 适用 Mac 设备建议 | 典型解码速度 (TPS) |
|:---------------|:-------------|:---------:| :---: | :--- |:----------------------:|
| BF16       | 原始精度     | ~14.72 GB | ≥ 24 GB - 32 GB | MacBook Pro 24GB / 36GB / 48GB+ |   ~88.3 tok/s (实测)   |
| MXFP8 量化 | MLX 量化权重 | ~8.06 GB  | ≥ 16 GB - 18 GB | MacBook Pro 16GB / 18GB / 24GB |  ~118.4 tok/s (实测)   |
| 4-bit 量化 | MLX 量化权重 | ~4.83 GB  | ≥ 8 GB - 12 GB | MacBook Air / Mac mini 8GB/16GB |  ~150.5 tok/s (实测)   |

> [!TIP]
> **环境要求**：
> - 推荐使用 **Python 3.11 / 3.12** 与 **uv** 管理虚拟环境；
> - Ling-3.0 架构（`bailing_hybrid`）需安装主干版本（`mlx-lm>=0.32.0`）；
> - 默认端口约定：`mlx_lm.server` 默认使用 `8080` 端口。

+++

### 步骤 1: 准备 Python 虚拟环境与安装 MLX-LM

使用 `uv` 创建 Python 虚拟环境，并安装包含 Ling-3.0 架构支持的最新 `mlx-lm`：

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!mkdir -p ~/.cache/huggingface/hub
!source .venv/bin/activate && uv pip install --upgrade mlx "mlx-lm>=0.32.0" 'openai>=1.52.0,<2.0.0' 'modelscope>=1.18.0'
```

+++

### 步骤 2: 下载官方 Ling-3.0-tiny BF16 基础权重

从 ModelScope 或 Hugging Face 下载官方 `inclusionAI/Ling-3.0-tiny` Safetensors 原始权重（约 14.7 GB）：
- [Ling-3.0-tiny on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny)
- [Ling-3.0-tiny on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-tiny)

```{code-cell}
!mkdir -p ~/models/Ling-3.0-tiny
!source .venv/bin/activate && uv run modelscope download \
  --model inclusionAI/Ling-3.0-tiny \
  --local_dir ~/models/Ling-3.0-tiny
```

> [!IMPORTANT]
> 官方在 ModelScope 上发布的预量化版本专为 vLLM / SGLang 运行时优化。
> MLX-LM 缺少对应算子，若直接下载 ModelScope 上的量化权重传入 `mlx_lm`，会出错。
> 因此在 Apple MLX 生态下，如果需要部署量化版本，必须通过 BF16 基础权重，通过 MLX 内置的 `mlx_lm.convert` 进行本地量化。

+++

### 步骤 3: 使用 MLX-LM CLI 部署模型（以原始权重为例）

如果设备内存有限，希望部署量化版本模型，可参考步骤 6 的 `mlx_lm.convert` 进行本地量化。

`mlx_lm.generate` 可直接验证模型加载与推理速率，无需任何格式转换：

```{code-cell}
!source .venv/bin/activate && python3 -m mlx_lm.generate \
  --model ~/models/Ling-3.0-tiny \
  --prompt "计算 17 × 23 的结果，请给出逐步推导逻辑。" \
  --max-tokens 256 \
  --temp 0.6
```

典型 CLI 输出（Apple Silicon Mac 实测稳态数据）：
```text
Loading model from ~/models/Ling-3.0-tiny...
Prompt: 计算 17 × 23 的结果，请给出逐步推导逻辑。
------
计算 17 × 23 的结果，我们可以使用乘法分配律进行逐步推导：
1. 将 23 拆分为 20 + 3：
   17 × 23 = 17 × (20 + 3)
2. 应用乘法分配律展开：
   = 17 × 20 + 17 × 3
   = 340 + 51
3. 求和得出最终结果：
   = 391
因此，17 × 23 = 391。
------
Prompt: 37 tokens, 421.58 tokens-per-sec
Generation: 256 tokens, 88.33 tokens-per-sec
Peak memory: 15.85 GB
```

+++

### 步骤 4: 启动 MLX-LM HTTP 推理服务 (OpenAI 兼容 API)

启动 `mlx_lm.server`，在本地暴露与 OpenAI 兼容的 HTTP REST 端点。

针对单人使用场景，推荐直接附加 Prompt Prefix Caching（前缀缓存），降低首 Token 延迟。

```bash
python3 -m mlx_lm.server \
  --model ~/models/Ling-3.0-tiny \
  --host 127.0.0.1 \
  --port 8080 \
  --prompt-cache-size 10 \
  --prompt-cache-bytes 2000000000 \
  --prefill-step-size 2048 \
  --trust-remote-code
```

服务端就绪时的典型日志：
```text
Loading model from ~/models/Ling-3.0-tiny...
Model loaded successfully.
INFO:     Started server process [65892]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
```

+++

### 步骤 5: 验证部署成功与接口调用

服务启动后，使用 OpenAI 兼容客户端进行调用验证：

#### 步骤 5.1: 连通性与模型健康检查

请求 `GET /v1/models` 查看当前运行的模型信息：

```{code-cell}
import urllib.request
import json

url = "http://127.0.0.1:8080/v1/models"
try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        status_code = response.getcode()
        body = response.read().decode('utf-8')
        print(f"Health check status: {status_code}")
        print(f"Models response: {body}")
except Exception as e:
    print(f"Health check error: {e}")
```

典型输出：
```json
Health check status: 200
Models response: {"object": "list", "data": [{"id": "~/models/Ling-3.0-tiny", "object": "model", "created": 1788849582}]}
```

+++

#### 步骤 5.2: 流式推理、思考链提取与性能测量

使用 OpenAI Python SDK 发送流式请求。注意：`mlx_lm.server` 将思考链解析在 `delta.reasoning` 字段中：

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="EMPTY"
)

model_id = client.models.list().data[0].id

def verify_streaming_and_thinking():
    prompt = "计算 17 × 23 的结果，请给出逐步推导逻辑。"
    print(f"Sending prompt: '{prompt}' to MLX model '{model_id}'...")

    start_time = time.time()
    first_token_time = None
    total_tokens = 0
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=2048,
            stream=True
        )

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            now = time.time()
            if first_token_time is None:
                first_token_time = now

            reasoning_chunk = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
            if reasoning_chunk:
                reasoning_text += reasoning_chunk
                total_tokens += 1
            elif delta.content:
                content_text += delta.content
                total_tokens += 1

        end_time = time.time()
        ttft = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
        decode_duration = end_time - first_token_time if first_token_time else 0.001
        tps = total_tokens / decode_duration

        print("\n=== Latency & Throughput Metrics ===")
        print(f"TTFT (Time to First Token): {ttft:.2f} ms")
        print(f"Decode TPS (Tokens/s): {tps:.2f} tok/s")
        print(f"Total Generated Tokens: {total_tokens}")
        print(f"Total Duration: {end_time - start_time:.2f} s")

        print("\n=== Extracted Reasoning Chain (<think>) ===")
        print(reasoning_text if reasoning_text else "[Note: Reasoning merged in main content or disabled]")

        print("\n=== Final Response Content ===")
        print(content_text)

    except Exception as e:
        print(f"Streaming verification failed: {e}")

if __name__ == "__main__":
    verify_streaming_and_thinking()
```

+++

#### 步骤 5.3: 测试 Function Calling 工具调用

传入标准工具结构，验证 `mlx_lm.server` 对结构化 Function Calling 的原生支持：

```{code-cell}
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="EMPTY"
)

model_id = client.models.list().data[0].id

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "查询并换算指定币种之间的实时汇率金额",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {
                        "type": "string",
                        "description": "源货币代码，例如 USD、EUR、CNY"
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "目标货币代码，例如 CNY、JPY、USD"
                    },
                    "amount": {
                        "type": "number",
                        "description": "需要转换的金额数额"
                    }
                },
                "required": ["from_currency", "to_currency", "amount"]
            }
        }
    }
]

def verify_tool_calling():
    print(f"Testing Function Calling with MLX-LM model '{model_id}'...")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": "请帮我把 100 美元换成人民币。"}
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
            if hasattr(message, "reasoning") and message.reasoning:
                print(f"\nTool Selection Reasoning: {message.reasoning}")
        else:
            print("\n=== Direct Response ===")
            print(message.content)

    except Exception as e:
        print(f"Tool calling verification failed: {e}")

if __name__ == "__main__":
    verify_tool_calling()
```

+++

### 步骤 6: 适用于低内存设备的 MLX 量化

对于统一内存为 8GB 或 16GB 的 Mac 设备（如 MacBook Air / Mac mini），运行 BF16 模型可能面临显存压力。

可以使用 `mlx_lm.convert` 本地生成 MLX 原生量化权重：

#### 选项 1：转换为 4-bit 量化（内存占用更低）
```bash
python3 -m mlx_lm.convert \
  --hf-path ~/models/Ling-3.0-tiny \
  --mlx-path ~/models/Ling-3.0-tiny-4bit \
  --quantize \
  --q-bits 4 \
  --trust-remote-code
```
转换后运行时内存消耗约 5 GB。

#### 选项 2：转换为 MXFP8 量化（精度较高）
```bash
python3 -m mlx_lm.convert \
  --hf-path ~/models/Ling-3.0-tiny \
  --mlx-path ~/models/Ling-3.0-tiny-mxfp8 \
  --quantize \
  --q-mode mxfp8 \
  --trust-remote-code
```
转换后运行时内存消耗约 8.7 GB。

> [!TIP]
> **部署路径修改提示**：
> 完成量化后，在后续验证或启动服务时需将 `--model` 参数修改为对应的量化目录。例如启动 4-bit 量化模型服务：
> ```bash
> python3 -m mlx_lm.server \
>   --model ~/models/Ling-3.0-tiny-4bit \
>   --host 127.0.0.1 \
>   --port 8080 \
>   --prompt-cache-size 10 \
>   --prompt-cache-bytes 2000000000 \
>   --prefill-step-size 2048 \
>   --trust-remote-code
> ```

#### 性能测算

以下数据基于 Apple Silicon M5 Pro 48GB 实测。

| 部署规格            | 权重格式 / 类型 | 磁盘体积 | Prefill 吞吐 |   解码吞吐 (Decode TPS)    | 首字延迟 (TTFT) | 峰值显存 (Peak RAM) | 推荐 Mac 设备 |
|:--------------------|:----------------| :---: |:------------:|:--------------------------:|:---------------:|:-------------------:| :--- |
| Ling-3.0-tiny       | BF16            | 14.72 GB | 421.58 tok/s |  88.33 tok/s (88.1~88.7)   |    87.93 ms     |      15.85 GB       | MacBook Pro 24GB / 36GB / 48GB+ |
| Ling-3.0-tiny-mxfp8 | MXFP8           | 8.06 GB | 505.23 tok/s | 118.35 tok/s (116.7~119.8) |    73.47 ms     |       8.75 GB       | MacBook Pro 16GB / 18GB / 24GB |
| Ling-3.0-tiny-4bit  | 4-bit           | 4.83 GB | 683.18 tok/s | 150.51 tok/s (149.4~151.5) |    54.16 ms     |       5.30 GB       | MacBook Air / Mac mini 8GB / 16GB |

+++

### 步骤 7: 常见问题与故障排查 (Troubleshooting)

1. **`Model type bailing_hybrid not supported` 错误**：
   - 现象：运行 `mlx_lm.server` 或 `mlx_lm.generate` 时抛出异常。
   - 原因：安装的 `mlx-lm` 为 PyPI 旧版（如 0.31.3），尚未包含 Ling-3.0 架构。
   - 解决：通过 GitHub 主干更新最新版本：
     ```bash
     uv pip install --upgrade "mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git"
     ```

2. **误用官方 `int4` / `fp8` 权重报错**：
   - 现象：传入 `inclusionAI/Ling-3.0-tiny-int4` 提示 `ValueError: Received ... parameters not in model`。
   - 解决：官方 `int4` 与 `fp8` 为 vLLM 专用打包格式。在 MLX 下请下载 BF16 基础模型，并使用 `mlx_lm.convert --q-bits 4` 生成 MLX 原生量化。

3. **端口占用冲突 (Port 8080 occupied)**：
   - 解决：`lsof -i :8080` 查找占用进程并清理，或指定 `--port 8081`。
