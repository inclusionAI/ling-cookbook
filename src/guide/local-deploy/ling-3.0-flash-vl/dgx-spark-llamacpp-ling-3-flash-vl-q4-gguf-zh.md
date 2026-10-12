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

# Ling-3.0-flash-VL on DGX Spark (llama.cpp Q4_K_M GGUF) 部署指南

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

Ling-3.0-flash-VL 是多模态 MoE 架构模型，支持图文与视频理解，具备 124B 总参数量与 ~5.5B 激活参数量，采用 5:1 的 KDA 与 MLA 交替注意力结构。

本指南演示在单台 `NVIDIA DGX Spark`（单张 GB10 统一内存 121 GiB 可用）上，基于 `llama.cpp` 部署 4-bit 量化（Q4_K_M）与配套多模态视觉投影（`mmproj`）权重。模型权重与视觉投影静态常驻约 80 GB，在上下文长度设为 32,768 tokens 时为 KV Cache 与系统预留约 40 GiB 物理空间，提供轻量化 C++ 运行时与兼容 OpenAI 的多模态 HTTP 接口。

> [!TIP]
> 环境准备建议：
> - 操作系统：Ubuntu 24.04 LTS (aarch64 / ARM64)；
> - Python 环境：使用 uv 创建 Python 3.12 虚拟环境；
> - 基础依赖：CUDA 13.0 驱动环境，CMake 3.28+，C++17 编译器，llama.cpp 最新主干（>= b11156）。

+++

### 步骤 1: 准备 Python 3.12 虚拟环境 (uv)

使用 `uv` 创建隔离的 Python 3.12 虚拟环境，并安装测试所需的依赖库：

```{code-cell}
!pip install -U uv
!uv venv --python 3.12 .venv
!source .venv/bin/activate && uv pip install openai pillow requests modelscope
```

典型输出：
```text
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtualenv at: .venv
Activate with: source .venv/bin/activate
Installed 4 packages in 12.3s
```

+++

### 步骤 2: 克隆并编译 llama.cpp (CUDA 支持)

Ling-3.0 系列模型的 MoE 混合注意力架构（`bailingmoe3`）及其多模态视觉投影（`mmproj`）已合并至 `llama.cpp` 官方主干（PR #26608 与 PR #29151）。克隆官方仓库并启用 CUDA 加速编译：

```{code-cell}
!git clone https://github.com/ggml-org/llama.cpp.git
!cd llama.cpp && cmake -B build -DGGML_CUDA=ON . && cmake --build build --parallel 8
```

典型输出：
```text
Cloning into 'llama.cpp'...
-- The CXX compiler identification is GNU 13.3.0
-- The CUDA compiler identification is NVIDIA 13.0.88
-- Detecting CUDA compiler ABI info - done
-- Building with CUDA architecture: native
[100%] Built target llama-server
[100%] Built target llama-quantize
```

+++

### 步骤 3: 获取 Ling-3.0-flash-VL GGUF 与 mmproj 权重

全精度 BF16 权重达 ~248 GB，单张 121 GiB 统一内存设备无法直接加载转换。推荐使用 ModelScope CLI 下载由社区预构建的 GGUF 权重与对应的视觉投影文件（`mmproj`）：
- [AtomicChat/Ling-3.0-flash-VL-GGUF on ModelScope](https://modelscope.cn/models/AtomicChat/Ling-3.0-flash-VL-GGUF)
- [bartowski/Ling-3.0-flash-VL-GGUF on Hugging Face](https://huggingface.co/bartowski/Ling-3.0-flash-VL-GGUF)

使用 `modelscope` 命令行工具下载 `Q4_K_M` 权重分卷及 `mmproj-Ling-3.0-flash-VL-F32.gguf`：

```{code-cell}
!source .venv/bin/activate && modelscope download --model AtomicChat/Ling-3.0-flash-VL-GGUF \
  --include "Ling-3.0-flash-VL-AD-Q4_K_M*" "mmproj-Ling-3.0-flash-VL-F32.gguf" \
  --local-dir ~/models/Ling-3.0-flash-VL-GGUF
```

典型输出：
```text
Downloading snapshot of AtomicChat/Ling-3.0-flash-VL-GGUF (model)…
Downloading 7 files from AtomicChat/Ling-3.0-flash-VL-GGUF@master
Ling-3.0-flash-VL-AD-Q4_K_M-00001-of-00006.gguf: 100%|██████████| 15.6G/15.6G [08:12<00:00, 31.7MB/s]
...
mmproj-Ling-3.0-flash-VL-F32.gguf: 100%|██████████| 1.74G/1.74G [00:58<00:00, 29.8MB/s]
Successfully downloaded to ~/models/Ling-3.0-flash-VL-GGUF
```

+++

### 步骤 4: 启动 llama-server 推理服务

启动 `llama-server` 提供兼容 OpenAI 的 HTTP 接口，核心参数说明：
- `-m`: 指定语言模型主权重（可直接指定分片根路径或合并文件）；
- `--mmproj`: 挂载多模态视觉投影权重，用于图像特征编码；
- `--host 0.0.0.0 --port 9102`: 监听端口设为 9102；
- `-ngl 99`: 将全部 42 层 Transformer 与 MoE 算子卸载至 GPU；
- `-c 32768`: 设置上下文长度为 32,768 tokens，预留显存给 KV Cache；
- `--jinja`: 启用 Jinja 模板引擎，解析系统指令与 `<think>` 思考链。

> [!WARNING]
> 前台运行说明：
> `llama-server` 在前台常驻运行。请在独立终端执行以下命令，待输出 `HTTP server listening` 后再回到本 Notebook 执行后续验证。

```{code-cell}
!./llama.cpp/build/bin/llama-server \
  -m ~/models/Ling-3.0-flash-VL-GGUF/Ling-3.0-flash-VL-AD-Q4_K_M-00001-of-00006.gguf \
  --mmproj ~/models/Ling-3.0-flash-VL-GGUF/mmproj-Ling-3.0-flash-VL-F32.gguf \
  --host 0.0.0.0 \
  --port 9102 \
  --api-key sk-ling-cookbook-test \
  -ngl 99 \
  -c 32768 \
  --jinja
```

典型启动日志：
```text
llama_model_loader: loaded meta data with 38 key-value pairs and 685 tensors
llama_model_load: general.architecture       = bailingmoe3
llama_model_load: bailingmoe3.expert_count    = 512
llama_model_load: bailingmoe3.expert_used_count = 8
clip_model_load: loaded multimodal projector
clip_model_load: vision encoder initialized
llama_backend_init: GGML_CUDA enabled
main: HTTP server listening on http://0.0.0.0:9102
```

+++

### 步骤 5: 验证部署与基准测速

服务启动后，使用 OpenAI Python SDK 进行接口可用性与多模态端到端测试：
1. 健康检查：确认模型接口连通性；
2. 流式推理与基准测速：对比纯文本与图文多模态推理，统计 TTFT、TPOT 与连续解码速率；
3. 视觉工具调用：验证结构化参数抽取能力。

+++

#### 步骤 5.1: 连通性与模型健康检查

请求 `GET /v1/models` 端点验证服务状态：

```{code-cell}
import urllib.request
import json

url = "http://127.0.0.1:9102/v1/models"
headers = {"Authorization": "Bearer sk-ling-cookbook-test"}
try:
    req = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req) as response:
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
Models response: {"object":"list","data":[{"id":"Ling-3.0-flash-VL-Q4_K_M","object":"model","created":1791788200,"owned_by":"llamacpp"}]}
```

+++

#### 步骤 5.2: 流式推理与基准测速

对比纯文本与图像多模态输入，解析 `<think>` 思考链并记录延迟与解码速率：

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9102/v1",
    api_key="sk-ling-cookbook-test"
)

def benchmark_stream(name, messages):
    print(f"\n============================================================")
    print(f"Sending [{name}] request to llama-server...")
    print("-" * 60)
    
    start_time = time.time()
    first_token_time = None
    chunk_count = 0
    exact_completion_tokens = None
    exact_prompt_tokens = None
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model="Ling-3.0-flash-VL-Q4_K_M",
            messages=messages,
            temperature=0.7,
            top_p=0.95,
            stream=True,
            stream_options={"include_usage": True}
        )

        for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                exact_completion_tokens = chunk.usage.completion_tokens
                exact_prompt_tokens = chunk.usage.prompt_tokens

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            r_text = getattr(delta, "reasoning_content", None)
            if not r_text and delta.model_extra:
                r_text = delta.model_extra.get("reasoning") or delta.model_extra.get("reasoning_content")

            if r_text:
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                reasoning_text += r_text
                chunk_count += 1

            if delta.content:
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                content_text += delta.content
                chunk_count += 1

        total_time = time.time() - start_time
        tokens = exact_completion_tokens if exact_completion_tokens else chunk_count
        decode_time = total_time - (first_token_time if first_token_time else 0)
        tps = (tokens - 1) / decode_time if decode_time > 0 and tokens > 1 else 0
        tpot = (decode_time / (tokens - 1) * 1000) if tokens > 1 else 0

        if reasoning_text:
            print(f"[Thinking Process]:\n{reasoning_text.strip()}\n")
            print("-" * 60)
        print(f"[Model Answer]:\n{content_text.strip()}\n")
        print("=" * 60)
        print(f"Metrics:")
        print(f"  Input Prompt Tokens: {exact_prompt_tokens}")
        print(f"  TTFT: {first_token_time * 1000:.2f} ms ({first_token_time:.3f} s)")
        print(f"  Total Duration: {total_time:.3f} s")
        print(f"  Generated Tokens: {tokens}")
        print(f"  Decoding Throughput (TPS): {tps:.2f} tokens/s")
        print(f"  Time Per Output Token (TPOT): {tpot:.2f} ms/tok")
        print("=" * 60)
        return {"ttft": first_token_time, "tps": tps, "tokens": tokens, "prompt_tokens": exact_prompt_tokens}

    except Exception as e:
        print(f"Request failed: {e}")
        return None

# 1. 纯文本基准测试
text_msg = [
    {"role": "user", "content": "请用简练的一段话阐述人机协同的未来演进趋势。"}
]
text_res = benchmark_stream("纯文本基准", text_msg)

# 2. 图像多模态推理测试
image_url = "https://raw.githubusercontent.com/inclusionAI/ling-cookbook/main/resources/assets/ling-flash-vl-benchmark.jpg"
mm_msg = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "请观察并分析这张图片中的视觉元素、构图层次与主要配色关系。"}
        ]
    }
]
mm_res = benchmark_stream("图像多模态", mm_msg)
```

典型输出：
```text
============================================================
Sending [纯文本基准] request to llama-server...
------------------------------------------------------------
[Model Answer]:
人机协同的未来演进正从指令交互迈向认知共生。AI 模型在海量数据整合、规律洞察与自动化执行中展现优势，而人类则在价值判断、跨领域综合与创新直觉中发挥主导作用...
============================================================
Metrics:
  Input Prompt Tokens: 32
  TTFT: 245.30 ms (0.245 s)
  Total Duration: 6.820 s
  Generated Tokens: 135
  Decoding Throughput (TPS): 20.38 tokens/s
  Time Per Output Token (TPOT): 49.07 ms/tok
============================================================

============================================================
Sending [图像多模态] request to llama-server...
------------------------------------------------------------
[Model Answer]:
## 1. 视觉元素识别
画面由三个带文字标注的独立几何图形构成：
1. 红色矩形（RED RECT）：位于左上方；
2. 蓝色圆形（BLUE CIRCLE）：位于右上方；
3. 绿色三角形（GREEN TRI）：居中位于下方。

## 2. 构图层次
整体呈倒三角稳定布局，下方的绿色三角形作为视觉基座，上方两个图形保持横向对称平衡。

## 3. 配色关系
采用红、蓝、绿三原色高对比配置，底色为中性浅蓝灰色，层次分明。
============================================================
Metrics:
  Input Prompt Tokens: 296
  TTFT: 1780.20 ms (1.780 s)
  Total Duration: 24.150 s
  Generated Tokens: 450
  Decoding Throughput (TPS): 20.12 tokens/s
  Time Per Output Token (TPOT): 49.70 ms/tok
============================================================
```

#### 实测性能基准对比分析

| 评测指标 | 纯文本流式 | 图像多模态流式 | 增量/差值 |
| :--- | :--- | :--- | :--- |
| 输入 Prompt Tokens | 32 tokens | 296 tokens | +264 tokens |
| 首字延迟 (TTFT) | 245.30 ms (0.25 s) | 1780.20 ms (1.78 s) | +1534.90 ms (~1.53 s) |
| 单 Token 耗时 (TPOT) | 49.07 ms/tok | 49.70 ms/tok | +0.63 ms/tok |
| 连续解码速率 (TPS) | 20.38 tokens/s | 20.12 tokens/s | -0.26 tokens/s (-1.3%) |
| 生成 Token 总数 | 135 tokens | 450 tokens | +315 tokens |
| 端到端总用时 | 6.82 s | 24.15 s | - |

> [!TIP]
> 测试指标说明：
> 1. 首字延迟 (TTFT)：图像输入（含视觉特征投影）首字响应为 1780.20 ms，较纯文本增加约 1.53 秒；
> 2. 连续解码速率：纯文本解码速率为 20.38 tokens/s，图像输入为 20.12 tokens/s；
> 3. 多模态投影配置：运行必须显式提供 `--mmproj` 参数，否则模型仅能处理纯文本请求。

+++

#### 步骤 5.3: 测试视觉工具调用 (Tool Calling)

测试基于图片内容的结构化工具调用，验证模型提取视觉参数与触发目标函数的能力：

```{code-cell}
def test_multimodal_tool_calling():
    print("测试多模态场景下的 Tool Calling...")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_design_palette",
                "description": "提取设计图中的主色调、视觉模式与构图元素数量",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "primary_color": {"type": "string", "description": "十六进制主色值"},
                        "theme_mode": {"type": "string", "enum": ["dark", "light"], "description": "色彩主题风格"},
                        "element_count": {"type": "integer", "description": "识别到的几何元素数量"}
                    },
                    "required": ["primary_color", "theme_mode", "element_count"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="Ling-3.0-flash-VL-Q4_K_M",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/inclusionAI/ling-cookbook/main/resources/assets/ling-flash-vl-benchmark.jpg"}},
                    {"type": "text", "text": "请分析图片的配色方案并调用工具提取设计元数据。"}
                ]
            }
        ],
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        print(f"成功触发工具调用: {tool_call.function.name}")
        print(f"参数 JSON: {tool_call.function.arguments}")
    else:
        print("直接文本回答 (未触发工具):", message.content)

test_multimodal_tool_calling()
```

典型输出：
```text
测试多模态场景下的 Tool Calling...
成功触发工具调用: generate_design_palette
参数 JSON: {"primary_color": "#E53935", "theme_mode": "light", "element_count": 3}
```

+++

### 步骤 6: 常见问题与运行调优

- 显存不足与上下文长度控制：
  - 现象：启动时分配 KV Cache 失败或报 `CUDA out of memory`；
  - 原因：`llama-server` 默认会尝试分配尽可能大的上下文缓存。在 121 GiB 统一内存设备上加载 80 GB 权重后，若上下文设为 131,072 会超出物理内存；
  - 解决：显式传入 `-c 32768`（或按需设为 `16384`），为系统预留 40 GiB 左右物理空间。
- 多模态 mmproj 配对要求：
  - 现象：传入图片时服务返回 `image input is not supported` 或忽略图片内容；
  - 原因：未在启动参数中配置 `--mmproj`；
  - 解决：启动命令必须包含 `--mmproj ~/models/Ling-3.0-flash-VL-Q4_K_M/mmproj-Ling-3.0-flash-VL-f16.gguf`。
- 构建编译架构匹配：
  - 现象：CMake 编译出的二进制在 Blackwell 架构上执行较慢；
  - 解决：在构建时指定 `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native`，启用本地 GPU 硬件原生架构优化。
