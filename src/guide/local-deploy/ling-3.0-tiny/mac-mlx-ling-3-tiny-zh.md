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

`MLX` 是 Apple 官方专为 Apple Silicon 统一内存架构打造的原生机器学习框架，具备轻量、低抽象开销与直接调用底层 Metal 硬件算子的优势。针对百灵大模型，`mlx-lm` 官方已在主干（PR #1711）原生合入了对 Ling-3.0 架构（`bailing_hybrid` / `BailingMoeV3ForCausalLM`）的支持。通过 `mlx-lm`，开发者可以在 Mac 本地直接加载 Safetensors 权重并执行本地 4-bit / 8-bit / MXFP 量化转换，亦可一键拉起与 OpenAI 兼容的高性能 HTTP REST 推理服务（`mlx_lm.server`）。

本指南介绍如何在 Apple Silicon Mac（M1 / M2 / M3 / M4 / M5 系列）上，使用 `mlx-lm` 部署 `Ling-3.0-tiny` 的多种精度版本（BF16 全精度、8-bit/MXFP8 高保真量化、4-bit 紧凑量化）。

---

### 设备内存门槛与选型矩阵 (Hardware & Precision Matrix)

用户可根据自身 Mac 统一内存容量选择适宜的运行精度规格：

| 部署规格 | 权重格式 | 纯权重体积 | 8K 上下文推荐内存 | 适用 Mac 设备建议 | 典型解码实测 (TPS) |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **BF16 (全精度完整版)** | 原生 Safetensors | **~14.72 GB** | **≥ 24 GB - 32 GB** | MacBook Pro 24GB / 36GB / 48GB+ | **~88.3 tok/s (实测)** |
| **8-bit / MXFP8 (高保真量化)** | MLX 量化权重 | **~8.06 - 8.27 GB** | **≥ 16 GB - 18 GB** | MacBook Pro 16GB / 18GB / 24GB | **~118.4 - 119.9 tok/s (实测)** |
| **4-bit (轻量推荐版)** | MLX 量化权重 | **~4.83 GB** | **≥ 8 GB - 12 GB** | MacBook Air / Mac mini 8GB/16GB | **~150.5 tok/s (实测)** |

> [!TIP]
> **环境与版本要求**：
> - 推荐使用 **Python 3.11 / 3.12** 环境；
> - **核心注意**：Ling-3.0 架构（`bailing_hybrid`）由 `mlx-lm` 官方 PR #1711 支持，需安装包含该补丁的最新版本（`git+https://github.com/ml-explore/mlx-lm.git` 或 `mlx-lm>=0.32.0`）；
> - 推荐使用 **uv** 管理 Python 虚拟环境与依赖；
> - 默认端口约定：`mlx_lm.server` 默认使用 `8080` 端口。

+++

### 步骤 1: 准备 Python 虚拟环境与安装 MLX-LM

#### 步骤 1.1: 创建虚拟环境并安装 MLX-LM

使用 `uv` 创建 Python 虚拟环境，并从 GitHub 安装包含 Ling-3.0 原生架构支持的最新 `mlx-lm`：

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!mkdir -p ~/.cache/huggingface/hub
!source .venv/bin/activate && uv pip install --upgrade mlx "mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git" 'openai>=1.52.0,<2.0.0' 'modelscope>=1.18.0'
```

典型安装输出：
```text
Resolved 34 packages in 1.58s
Installed 24 packages in 95ms
 + mlx==0.32.2
 + mlx-lm==0.32.0 (from git+https://github.com/ml-explore/mlx-lm.git)
 + modelscope==1.39.1
 + openai==1.109.1
```

+++

### 步骤 2: 下载官方 Ling-3.0-tiny 基础权重 (Safetensors)

从 ModelScope 或 Hugging Face 下载官方 `inclusionAI/Ling-3.0-tiny` Safetensors 全精度基础权重（共 32 个分片，约 15.8 GB）：
- [Ling-3.0-tiny on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny)
- [Ling-3.0-tiny on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-tiny)

推荐使用 `modelscope` CLI 下载至本地 `~/models/Ling-3.0-tiny`：

> [!NOTE]
> 这一步需从模型托管平台下载约 15.8 GB 权重，耗时取决于网络带宽，请耐心等待。

```{code-cell}
!mkdir -p ~/models/Ling-3.0-tiny
!source .venv/bin/activate && uv run modelscope download \
  --model inclusionAI/Ling-3.0-tiny \
  --local_dir ~/models/Ling-3.0-tiny
```

典型下载输出：
```text
Downloading shards: 100%|██████████| 32/32 [05:20<00:00, 51.2MB/s]
✓ Successfully downloaded Ling-3.0-tiny to /Users/sipan/models/Ling-3.0-tiny
```

+++

### 步骤 3: 准备多精度规格 (BF16 / 8-bit / 4-bit)

`mlx-lm` 具备出色的权重兼容性与原生快速量化能力，用户可根据自身硬件条件选择准备对应精度的权重：

#### 选项 A：BF16 原生全精度版

下载的 Safetensors 基础权重即为 BF16 原生格式，MLX 原生开箱即用，**无需任何转换步骤**，直接指定本地目录路径 `~/models/Ling-3.0-tiny` 即可。

#### 选项 B：8-bit / MXFP8 高保真量化版

使用 `mlx_lm.convert` 快速完成 8-bit 或 MXFP8（Microscaling FP8）量化，权重体积缩减至约 7.9 GB：

```{code-cell}
!source .venv/bin/activate && python3 -m mlx_lm.convert \
  --hf-path ~/models/Ling-3.0-tiny \
  --mlx-path ~/models/Ling-3.0-tiny-mxfp8 \
  --quantize \
  --q-mode mxfp8 \
  --trust-remote-code
```

或使用标准 8-bit 量化：
```bash
python3 -m mlx_lm.convert \
  --hf-path ~/models/Ling-3.0-tiny \
  --mlx-path ~/models/Ling-3.0-tiny-8bit \
  --quantize \
  --q-bits 8 \
  --trust-remote-code
```

#### 选项 C：4-bit 紧凑推荐版

使用 `mlx_lm.convert` 转换为 4-bit 紧凑量化，权重仅约 4.0 GB，适合 8GB / 16GB 消费级 Mac 设备：

```{code-cell}
!source .venv/bin/activate && python3 -m mlx_lm.convert \
  --hf-path ~/models/Ling-3.0-tiny \
  --mlx-path ~/models/Ling-3.0-tiny-4bit \
  --quantize \
  --q-bits 4 \
  --trust-remote-code
```

+++

### 步骤 4: 使用 MLX-LM CLI 工具快速验证生成

`mlx-lm` 提供了极其便捷的命令行交互工具 `mlx_lm.generate`，可直接在终端中验证模型加载、推理正确性与解码速率。根据你所准备的精度规格运行：

#### 选项 A：验证 BF16 原生版
```{code-cell}
!source .venv/bin/activate && python3 -m mlx_lm.generate \
  --model ~/models/Ling-3.0-tiny \
  --prompt "计算 17 × 23 的结果，请给出逐步推导逻辑。" \
  --max-tokens 256 \
  --temp 0.6
```

#### 选项 B：验证 8-bit / MXFP8 版
```{code-cell}
!source .venv/bin/activate && python3 -m mlx_lm.generate \
  --model ~/models/Ling-3.0-tiny-mxfp8 \
  --prompt "计算 17 × 23 的结果，请给出逐步推导逻辑。" \
  --max-tokens 256 \
  --temp 0.6
```

#### 选项 C：验证 4-bit 推荐版
```{code-cell}
!source .venv/bin/activate && python3 -m mlx_lm.generate \
  --model ~/models/Ling-3.0-tiny-4bit \
  --prompt "计算 17 × 23 的结果，请给出逐步推导逻辑。" \
  --max-tokens 256 \
  --temp 0.6
```

典型 CLI 输出（Apple Silicon Mac 实测稳态数据）：
```text
Loading model from /Users/sipan/models/Ling-3.0-tiny...
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
Prompt: 17 tokens, 255.22 tokens-per-sec
Generation: 256 tokens, 86.05 tokens-per-sec
Peak memory: 16.39 GB
```

+++

### 步骤 5: 启动 MLX-LM HTTP 推理服务 (OpenAI 兼容)

启动 `mlx_lm.server`，在本地拉起与 OpenAI 兼容的 HTTP REST 端点。常用参数说明：
- `--model <模型路径>`：指定本地模型权重目录（如 `~/models/Ling-3.0-tiny`）
- `--host 127.0.0.1 --port 8080`：指定服务监听地址与端口（统一采用标准端口 `8080`）
- `--trust-remote-code`：信任模型自定义代码
- `--chat-template-args '{"enable_thinking": true}'`：配置聊天模板默认行为（保留思考链）

⚠️ 特别注意：

`mlx_lm.server` 默认以前台常驻模式运行。若在 Notebook 单元格中直接运行，将持续占用 Kernel。**建议在独立的终端会话中运行下方启动命令**；服务端就绪后即可在 Notebook 中进行接口调用与验证。根据你所准备的精度规格选择对应的启动命令：

#### 选项 A：启动 BF16 全精度服务
```bash
python3 -m mlx_lm.server \
  --model ~/models/Ling-3.0-tiny \
  --host 127.0.0.1 \
  --port 8080 \
  --trust-remote-code
```

#### 选项 B：启动 8-bit / MXFP8 高保真服务
```bash
python3 -m mlx_lm.server \
  --model ~/models/Ling-3.0-tiny-mxfp8 \
  --host 127.0.0.1 \
  --port 8080 \
  --trust-remote-code
```

#### 选项 C：启动 4-bit 轻量推荐服务
```bash
python3 -m mlx_lm.server \
  --model ~/models/Ling-3.0-tiny-4bit \
  --host 127.0.0.1 \
  --port 8080 \
  --trust-remote-code
```

#### 选项 D：单人使用极致加速推荐服务（4-bit + 前缀缓存）

针对个人开发者在本地终端、IDE 编程助手或单人多轮 Agent 调用场景，推荐直接配置 **Prompt Prefix Caching（前缀缓存）**，将历史会话与系统提示词保留于显存中，使得后续多轮交互的首字延迟（TTFT）降至毫秒级：

```bash
python3 -m mlx_lm.server \
  --model ~/models/Ling-3.0-tiny-4bit \
  --host 127.0.0.1 \
  --port 8080 \
  --prompt-cache-size 10 \
  --prompt-cache-bytes 2000000000 \
  --prefill-step-size 2048 \
  --trust-remote-code
```

服务端就绪时的典型日志：
```text
Loading model from /Users/sipan/models/Ling-3.0-tiny...
Model loaded successfully.
INFO:     Started server process [65892]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
```

+++

### 步骤 6: 验证部署成功并使用模型服务

服务启动后，可通过 OpenAI 兼容客户端进行端到端调用验证：

1. **健康检查** - 确认模型加载状态与 HTTP 端点连通性；
2. **流式 Reasoning 验证与速度测算** - 测试 `<think>` 思考链解析，测算 TTFT 与 Decode TPS；
3. **Function Calling 工具调用测试** - 验证结构化工具调用支持；
4. **多精度性能与显存对比表**。

+++

#### 步骤 6.1: 连通性与模型健康检查

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
Models response: {"object": "list", "data": [{"id": "/Users/sipan/models/Ling-3.0-tiny-4bit", "object": "model", "created": 1788849582}]}
```

+++

#### 步骤 6.2: 流式推理、思考链提取与性能测算

使用 OpenAI Python SDK 发送流式请求。注意：`mlx_lm.server` 将思考链解析在 `delta.reasoning` 字段中：

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="EMPTY"
)

# 动态获取已加载的模型 ID
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

典型测试结果（Apple Silicon Mac 实测）：
```text
Sending prompt: '计算 17 × 23 的结果，请给出逐步推导逻辑。' to MLX model '/Users/sipan/models/Ling-3.0-tiny-4bit'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 172.94 ms
Decode TPS (Tokens/s): 105.32 tok/s
Total Generated Tokens: 449
Total Duration: 4.44 s

=== Extracted Reasoning Chain (<think>) ===
1. 分析请求：计算 17 × 23 的结果，给出逐步推导逻辑。
2. 选择分配律方法：
   第一步：将 23 拆分为 20 和 3：17 × (20 + 3)
   第二步：分别计算两个部分：17 × 20 = 340，17 × 3 = 51
   第三步：将两部分相加：340 + 51 = 391

=== Final Response Content ===
计算 17 × 23 的结果，我们可以通过乘法分配律进行逐步推导：
1. 第一步：将 23 拆分为 20 和 3
2. 第二步：分别计算两项：17 × 20 = 340，17 × 3 = 51
3. 第三步：将两部分相加：340 + 51 = 391
最终结果：17 × 23 = 391。
```

+++

#### 步骤 6.3: 测试 Function Calling 工具调用

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

典型测试结果：
```text
Testing Function Calling with MLX-LM model '/Users/sipan/models/Ling-3.0-tiny-4bit'...

=== Function Call Output Detected ===
Tool Call ID: 16f58426-f0cb-4d90-b34d-7d50f82678b9
Function Name: convert_currency
Arguments JSON: {"from_currency": "USD", "to_currency": "CNY", "amount": 100}

Tool Selection Reasoning: 用户想把 100 美元换成人民币。我需要使用 convert_currency 工具来完成这个转换。参数：- from_currency: "USD" - to_currency: "CNY" - amount: 100
```

+++

#### 步骤 6.4: 多档精度解码吞吐与内存实测对比

以下为在 Apple Silicon Mac（M5 Pro，48GB 统一内存，macOS 15）上，通过预热（Metal Shader JIT 编译）与 3 轮稳态采样对 `Ling-3.0-tiny` 多种精度进行端到端推理与显存采样的完整实测基准：

| 部署规格 | 权重格式 / 类型 | 磁盘体积 | 预热冷启动 Prefill | 稳态 Prefill 吞吐 | 稳态解码吞吐 (Decode TPS) | 稳态首字延迟 (TTFT) | 峰值显存 (Peak RAM) | 推荐 Mac 设备 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Ling-3.0-tiny` (BF16)** | Safetensors (全精度) | **14.72 GB** | 300.98 tok/s | **421.58 tok/s** | **88.33 tok/s** (88.1~88.7) | **87.93 ms** | **15.85 GB** | MacBook Pro 24GB / 36GB / 48GB+ |
| **`Ling-3.0-tiny-8bit`** | MLX 8-bit (高保真) | **8.27 GB** | 169.35 tok/s | **538.28 tok/s** | **119.88 tok/s** (119.1~120.4) | **68.88 ms** | **8.98 GB** | MacBook Pro 16GB / 18GB / 24GB |
| **`Ling-3.0-tiny-mxfp8`** | MLX MXFP8 (微缩放FP8) | **8.06 GB** | 163.05 tok/s | **505.23 tok/s** | **118.35 tok/s** (116.7~119.8) | **73.47 ms** | **8.75 GB** | MacBook Pro 16GB / 18GB / 24GB |
| **`Ling-3.0-tiny-4bit`** | MLX 4-bit (轻量推荐) | **4.83 GB** | 438.48 tok/s | **683.18 tok/s** | **150.51 tok/s** (149.4~151.5) | **54.16 ms** | **5.30 GB** | MacBook Air / Mac mini 8GB / 16GB |

> [!TIP]
> **MLX vs llama.cpp 选型与对比建议**：
> - **吞吐对比**：MLX 4-bit 在 CLI 稳态推理场景下跑出了高达 **150.51 tok/s** 的极致解码吞吐，超越了同一硬件下 llama.cpp 的 123.69 tok/s；
> - **开发生态**：MLX 为纯 Python + Metal 原生驱动，可直接利用 Python 生态进行 LoRA 微调、嵌入式应用与数据流转换，灵活性更高；
> - **选型建议**：如果你的技术栈偏向 Python 原生集成、定制化算子或本地微调扩展，首选 **MLX-LM**；如果聚焦多语言跨平台通用服务分发，可结合 **llama.cpp** 指南使用。

+++

#### 步骤 6.5: 单人使用场景下的极致吞吐与低延迟调优技巧

在个人 Mac 工作站上进行日常代码编写、终端助手与单人多轮 Agent 调用时，为了追求极致的响应敏捷度与吞吐效率，建议组合应用以下 4 项专属调优手段：

1. **4-bit 权重量化（第一速度优先）**：
   - **原理**：大模型端侧解码属于显存带宽受限（Memory-Bound）任务。从 BF16 切换至 4-bit 后，单步迭代需从统一内存搬运的激活权重体积锐减 **~70%**；
   - **效果**：解码吞吐直接从 **86.1 tok/s** 跃升至 **147.9 tok/s**（提升超过 **+70%**），同时常驻显存仅需 **5.2 GB**，为日常开发与多任务并发留出充裕空间。

2. **前缀缓存（Prompt Prefix Caching，消灭多轮等待）**：
   - **原理**：在多轮对话或携带固定 System Prompt（如长指令、复杂工具定义或领域知识片段）的场景中，历史前缀的注意力状态完全一致。`mlx_lm.server` 通过 `--prompt-cache-size 10` 自动启用 LRU 缓存命中前缀，直接跳过计算密集型的 Prefill 阶段；
   - **效果**：后续多轮交互的首字延迟（TTFT）从数百毫秒直接降至 **< 10 ms**，交互极其丝滑。

3. **超长上下文下的 KV Cache 量化（MLA 协同）**：
   - **原理**：`Ling-3.0-tiny` 架构本身使用了 **MLA（Multi-head Latent Attention）** 机制，原生将 KV 向量压缩至 `kv_lora_rank = 512`；在需要处理 32K ~ 128K 超长文档时，可在 CLI 生成时追加 `--kv-bits 4` 或 `--kv-bits 8`，进一步将 KV 显存开销压缩 50% ~ 75%；
   - **效果**：避免长文本后期由于 KV Cache 读写带宽被撑满导致的解码速度断崖式下跌。

4. **系统级统一内存锁定（Wired Memory）**：
   - **原理**：避免 macOS 系统的内存管理机制在多任务切换时将模型内存页换出至磁盘 Swap；
   - **配置**：执行系统命令提高 Metal 显存直接分配上限（需管理员权限）：
     ```bash
     sudo sysctl iogpu.wired_mem_limit=30000000000  # 调整为 ~30GB
     ```

+++

### 步骤 7: 常见问题与故障排查 (Troubleshooting)

1. **`Model type bailing_hybrid not supported` 错误**：
   - 现象：运行 `mlx_lm.server` 或 `mlx_lm.generate` 时抛出异常 `ValueError: Model type bailing_hybrid not supported`。
   - 原因：当前安装的 `mlx-lm` 为 PyPI 旧版（如 0.31.3），尚未打包 Ling-3.0 架构（PR #1711 于 2026 年 8 月合入）。
   - 解决：通过 GitHub 主干更新最新版本：
     ```bash
     uv pip install --upgrade "mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git"
     ```

2. **Hugging Face 缓存目录缺失报错 (`CacheNotFound`)**：
   - 现象：访问 `GET /v1/models` 时服务端报错 `huggingface_hub.errors.CacheNotFound: Cache directory not found: ~/.cache/huggingface/hub`。
   - 解决：预先创建该目录即可解决：
     ```bash
     mkdir -p ~/.cache/huggingface/hub
     ```

3. **统一内存不足与量化压缩**：
   - 现象：在 16GB 或 8GB 设备上直接加载 BF16 权重引发系统剧烈 Swap 或卡顿。
   - 解决：使用 `mlx_lm.convert` 将权重压缩至 4-bit（~4.8 GB），大幅降低物理内存占用并提升解码速率：
     ```bash
     python3 -m mlx_lm.convert \
       --hf-path ~/models/Ling-3.0-tiny \
       --mlx-path ~/models/Ling-3.0-tiny-4bit \
       --quantize \
       --q-bits 4
     ```

4. **端口占用冲突 (Port 8080 occupied)**：
   - 现象：启动服务提示 `Errno 48: Address already in use`。
   - 解决：查询占用进程并清理：
     ```bash
     lsof -i :8080
     kill -9 <PID>
     ```
     或在启动参数中修改 `--port 8081`。

