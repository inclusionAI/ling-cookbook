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

# Ling-3.0-tiny on Apple Silicon Mac (Ollama) 部署指南

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

`Ling-3.0-tiny` 是百灵大模型系列中的 7.9B 轻量 Sparse MoE 语言模型，单 Token 激活参数量仅 1.3B，原生支持 128K 长上下文，在低算力与内存占用下提供强劲的端侧智能体、工具调用与深度思考推理能力。

本指南介绍如何在 Apple Silicon Mac 上使用 Ollama 快速部署 `Ling-3.0-tiny`。

---

### 设备内存门槛与量化选型矩阵 (Hardware & Quantization Matrix)

用户可根据自身 Mac 设备的统一内存容量，选择最适合的运行规格：

| 部署规格 | 量化类型 | 纯权重体积 | 8K 上下文推荐内存 | 适用 Mac 设备建议 |
| :--- | :--- | :---: | :---: | :--- |
| BF16 (全精度) | 16-bit GGUF | ~15.80 GB | ≥ 24 GB - 32 GB | MacBook Pro 36GB / 48GB+ |
| Q8_0 (高精度) | 8-bit GGUF | ~8.30 GB | ≥ 16 GB - 24 GB | MacBook Pro 18GB / 24GB |
| Q4_K_M (较高精度) | 4-bit GGUF | ~4.30 GB | ≥ 8 GB | MacBook Air / Pro 16GB |

+++

### 步骤 1: 准备环境与安装 Ollama

#### 步骤 1.1: 通过 Homebrew 安装 Ollama 与 uv

推荐通过 Homebrew 安装 Ollama 与 Python 包管理工具 `uv`：

```{code-cell}
# 安装 Ollama 与 uv（若已安装可跳过）
!brew install ollama uv
```

典型安装输出：
```text
==> Would install 1 formula:
ollama
==> Would install 2 dependencies for ollama:
mlx
mlx-c
==> Do you want to proceed with the installation? [y/n]
==> Fetching downloads for: ollama

==> Installing ollama
==> Pouring ollama--0.33.3.arm64_tahoe.bottle.tar.gz
🍺  /opt/homebrew/Cellar/ollama/0.33.3: 16 files, 52.9MB
==> Caveats
==> ollama
To start ollama now and restart at login:
  brew services start ollama
Or, if you don't want/need a background service you can just run:
  OLLAMA_FLASH_ATTENTION="1" OLLAMA_KV_CACHE_TYPE="q8_0" /opt/homebrew/opt/ollama/bin/ollama serve
```

#### 步骤 1.2: 创建 Python 测试虚拟环境并安装依赖

该 Python 环境用于后续步骤中的 API 调用验证与测速：

```{code-cell}
# 使用 uv 创建独立虚拟环境并安装 OpenAI SDK
!uv venv .venv --python 3.12
!uv pip install --upgrade 'openai>=1.52.0'
```

典型运行输出：
```text
❯ uv venv .venv --python 3.12
Using CPython 3.12.14
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate

❯ source .venv/bin/activate

❯ uv pip install --upgrade 'openai>=1.52.0'
Resolved 14 packages in 559ms
Prepared 14 packages in 358ms
Installed 14 packages in 23ms
 + annotated-types==0.8.0
 + anyio==4.15.1
 + h11==0.16.0
 + httpcore2==2.12.0
 + httpx2==2.12.0
 + idna==3.19
 + jiter==0.16.0
 + openai==3.8.0
 + pydantic==2.13.5
 + pydantic-core==2.46.5
 + sniffio==1.3.1
 + truststore==0.10.4
 + typing-extensions==4.16.0
 + typing-inspection==0.4.4
```

#### 步骤 1.3: 启动 Ollama 服务并检查就绪状态

Ollama 服务需要常驻运行以监听 API 请求（默认监听端口 `11434`）。

打开一个独立的终端窗口，运行以下命令启动服务（建议包含下列 Flash Attention 与 KV Cache 量化参数）：

```bash
OLLAMA_FLASH_ATTENTION="1" OLLAMA_KV_CACHE_TYPE="q8_0" ollama serve
```

终端启动后将输出类似以下日志：
```text
time=2026-09-07T14:41:36.482+08:00 level=INFO source=routes.go:2012 msg="Listening on 127.0.0.1:11434 (version 0.33.3)"
```

服务就绪后，在当前工作终端中检查连通性与版本：

```{code-cell}
# 检查本地 Ollama 服务是否就绪
!curl -s http://127.0.0.1:11434/api/version
```

典型检查输出：
```json
{"version":"0.33.3"}
```

+++

### 步骤 2: 部署模型

Ollama 原生支持直接拉取并运行 Hugging Face 上的 GGUF 仓库。你只需在终端中选择对应的量化档位执行拉取或运行：

#### 选项 A：BF16 全精度版

原始精度权重，适合内存高于 24GB 的设备：

> [!NOTE]
> 这一步需从 Hugging Face 远程下载约 15 GB 的模型权重文件，耗时较长，需要耐心等待。

```{code-cell}
# 拉取并运行 BF16 官方 GGUF 模型文件
!ollama run hf.co/inclusionAI/Ling-3.0-tiny-GGUF:bf16 "请用一句话介绍你自己。"
```

典型拉取与运行输出：
```text
pulling manifest 
pulling b11d4a45d3ad: 100% ▕███████████████████████████████████████████████████████████████████████▏  15 GB                         
verifying sha256 digest 
writing manifest 
success 
嗯，用户让我用一句话介绍自己，这很简单直接。我需要简洁地概括我的身份和核心功能。

想到了可以说明我是蚂蚁集团的AI模型，同时提一下我的名称和主要能力。保持一句话的长度，不加多余解释。

用“作为”开头比较自然，结尾加上“致力于...”的表述能突出服务性。</think>我是一个名为“百灵大模型（Ling）”的AI助手，由蚂蚁集团开发，旨在通过自然语言交互提供信息处理与智能支持。
```

#### 选项 B：Q8_0 量化版

该量化版适合 18GB / 24GB 内存设备：

```{code-cell}
# 拉取并运行 Q8_0 版本
!ollama run hf.co/inclusionAI/Ling-3.0-tiny-GGUF:Q8_0 "请用一句话介绍你自己。"
```

典型拉取与运行输出：
```text
pulling manifest 
pulling 9299a9e5cbc5: 100% ▕███████████████████████████████████████████████████████████████████████▏ 8.4 GB                         
pulling 62edd696268b: 100% ▕███████████████████████████████████████████████████████████████████████▏  223 B                         
pulling a254ca5329e8: 100% ▕███████████████████████████████████████████████████████████████████████▏   65 B                         
verifying sha256 digest 
writing manifest 
success 
嗯，用户让我用一句话介绍自己，这是一个简单的请求。用户可能想要一个简洁但全面的描述，既能说明我的功能，又能让用户快速了解我的核心特点。

考虑到我的身份是百灵大模型（Ling），应该突出其作为通用语言大模型的核心能力，同时保持简洁。想到了用“由蚂蚁集团研发的百灵大模型”开头，然后说明我的功能范围，再补充我的设计理念。

这样既能展示专业性，又能让用户一目了然。想到了“由蚂蚁集团研发的通用语言大模型，擅长多种任务并支持深度推理”这样的表述。
</think>我是由蚂蚁集团研发的百灵大模型（Ling），擅长处理多种语言任务，并支持深度推理与复杂问题求解。
```

#### 选项 C：Q4_K_M 量化版

适合 8GB / 16GB 内存设备。

```{code-cell}
# 拉取并运行 Q4_K_M 版本（按需选择）
!ollama run hf.co/inclusionAI/Ling-3.0-tiny-GGUF:Q4_K_M "请用一句话介绍你自己。"
```

典型拉取与运行输出：
```text
pulling manifest 
pulling 246d67d45f5b: 100% ▕███████████████████████████████████████████████████████████████████████▏ 4.8 GB                         
pulling 62edd696268b: 100% ▕███████████████████████████████████████████████████████████████████████▏  223 B                         
pulling a254ca5329e8: 100% ▕███████████████████████████████████████████████████████████████████████▏   65 B                         
verifying sha256 digest 
writing manifest 
success 
首先，用户说：“请用一句话介绍你自己。”意思是“Please introduce yourself in one sentence.”所以，我需要用一句话来介绍自己。

我是百灵大模型（Ling），由蚂蚁集团开发，是一个通用语言大模型。

我的介绍应该简洁、清晰，符合一句话的要求。

关键点：
- 我的名字：百灵大模型（Ling）
- 由蚂蚁集团开发
- 角色：通用语言大模型

确保只有一句话：不要使用分号或逗号来分隔多个句子。

最终句子：我是由蚂蚁集团开发的通用语言大模型百灵大模型（Ling）。
</think>我是由蚂蚁集团开发的通用语言大模型百灵大模型（Ling），旨在为用户提供广泛的任务支持。
```

+++

### 步骤 3: 检查模型可用性

Ollama 原生兼容 OpenAI API 规范。以下测试代码验证基础对话与端侧工具调用能力：

#### 步骤 3.1: 基础对话与思考链控制 (Hello World)

`Ling-3.0-tiny` 具备原生推理链能力。可通过提示词或系统指令控制是否展开思考：

```{code-cell}
from openai import OpenAI

# 初始化客户端，连接本地 Ollama 服务端口 11434
client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

# 可按需替换为你所拉取的规格（如 :bf16, :Q8_0 或 :Q4_K_M）
MODEL_NAME = "hf.co/inclusionAI/Ling-3.0-tiny-GGUF:bf16"

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "你是一个严谨平实的端侧 AI 助手。"},
        {"role": "user", "content": "计算 17 × 23 的结果，并简要说明计算过程。"}
    ],
    temperature=0.1
)

print("=== 模型输出内容 ===")
print(response.choices[0].message.content)
```

典型输出：
```text
=== 模型输出内容 ===
计算 17 × 23：
17 × 20 = 340
17 × 3 = 51
340 + 51 = 391
因此，17 × 23 = 391。
```

#### 步骤 3.2: 检查端侧工具调用

`Ling-3.0-tiny` 对 Agent 场景适用。我们定义一个天气查询工具函数，验证模型能否正确触发 Function Calling：

```{code-cell}
import json

# 定义工具
tools = [
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
    {"role": "user", "content": "请帮我查一下杭州现在的实时天气怎么样？"}
]

tool_response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    tools=tools,
    temperature=0.0
)

message = tool_response.choices[0].message
if message.tool_calls:
    print("✅ 工具调用检查通过！模型成功发起 Tool Call：")
    for call in message.tool_calls:
        print(f"  - 函数名称: {call.function.name}")
        print(f"  - 传入参数: {call.function.arguments}")
else:
    print("❌ 未触发工具调用，模型直接返回了内容：", message.content)
```

典型输出：
```text
✅ 工具调用检查通过！模型成功发起 Tool Call：
  - 函数名称: get_current_weather
  - 传入参数: {"city":"杭州"}
```

+++

### 步骤 4: 对模型测速

通过调用 Ollama 原生 `/api/generate` 接口，自动统计生成速率 (Decode TPS) 与端到端耗时：

```{code-cell}
import urllib.request
import json
import time

prompt_data = {
    "model": MODEL_NAME,
    "prompt": "请用 200 字左右客观阐述混合注意力机制（如 KDA 线性注意力结合 MLA）在大语言模型长上下文推理中的核心优势与计算复杂度差异。",
    "stream": False,
    "options": {
        "temperature": 0.0,
        "num_predict": 256
    }
}

req = urllib.request.Request(
    "http://127.0.0.1:11434/api/generate",
    data=json.dumps(prompt_data).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

start_time = time.time()
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode("utf-8"))
elapsed_total = time.time() - start_time

# 提取 Ollama 原生统计指标
eval_count = result.get("eval_count", 0)                  # 生成的 Token 数
eval_duration = result.get("eval_duration", 1) / 1e9       # 生成耗时 (秒)

decode_tps = eval_count / eval_duration if eval_duration > 0 else 0

print("========================================")
print(f" Ling-3.0-tiny on Mac (Ollama) 测速")
print("========================================")
print(f"  - 模型版本: {MODEL_NAME}")
print(f"  - Decode 速率 (Generation TPS): {decode_tps:.2f} tokens/s (生成 {eval_count} tokens / {eval_duration:.3f}s)")
print(f"  - 端到端完整耗时: {elapsed_total:.3f}s")
print("========================================")
```

典型运行输出（基于 Apple Silicon Mac 实测）：
```text
========================================
 Ling-3.0-tiny on Mac (Ollama) 测速
========================================
  - 模型版本: hf.co/inclusionAI/Ling-3.0-tiny-GGUF:bf16
  - Decode 速率 (Generation TPS): 72.63 tokens/s (生成 358 tokens / 4.93s)
  - 端到端完整耗时: 4.93s
========================================
```

基于 Apple Silicon Mac 实测的典型性能：

| 硬件配置 | 量化规格 | 上下文长度 | Decode TPS (Generation) | 内存峰值 |
| :--- | :--- | :-: | :-: | :-: |
| M5 Pro (48GB) | BF16 (全精度) | 32K | ~ 72.6 tokens/s | ~ 14.9 GiB |
| M5 Pro (48GB) | Q8_0 (高保真) | 32K | ~ 105.4 tokens/s | ~ 6.0 GiB |
| M5 Pro (48GB) | Q4_K_M | 32K | ~ 127.3 tokens/s | ~ 4.9 GiB |

+++

### 步骤 5: 常见问题

1. 连接拒绝：Failed to connect to 127.0.0.1 port 11434
   - 原因：Ollama 服务未在后台运行。
   - 解决：在独立终端中执行 `OLLAMA_FLASH_ATTENTION="1" OLLAMA_KV_CACHE_TYPE="q8_0" ollama serve` 启动服务。

2. 如何指定与控制模型的上下文长度 (Context Length / `num_ctx`)
   `Ling-3.0-tiny` 原生支持 128K 上下文。Ollama 默认上下文可能较为保守（通常为 2048 或 4096），可按需调整：
   - 方式 A（Modelfile 永久定制，推荐）：
     创建 Modelfile 并指定所需上下文长度（如 32K）：
     ```dockerfile
     FROM hf.co/inclusionAI/Ling-3.0-tiny-GGUF:bf16
     PARAMETER num_ctx 32768
     ```
     在终端中构建并生效：`ollama create ling-3-tiny-32k -f Modelfile`。
   - 方式 B（Python API / OpenAI SDK 动态设置）：
     调用接口时在请求体中传入：
     ```python
     response = client.chat.completions.create(
         model=MODEL_NAME,
         messages=[...],
         extra_body={"options": {"num_ctx": 32768}}
     )
     ```
   - 方式 C（交互式命令行临时设置）：
     在 `ollama run` 提示符下直接执行：`/set parameter num_ctx 32768`。

3. 系统内存交换 (Swap) 导致推理变慢
   - 原因：开启了过大的上下文窗口（如 64K/128K）或同时开启了过多大型应用。
   - 解决：对于 8GB / 16GB 设备，建议搭配 `Q4_K_M` 规格，并将上下文控制在 8192 或 16384 内，以保障系统整体流畅度。
