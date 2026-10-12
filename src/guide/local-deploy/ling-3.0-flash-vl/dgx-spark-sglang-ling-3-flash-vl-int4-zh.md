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

# Ling-3.0-flash-VL on DGX Spark (SGLang INT4 W4A16) 部署指南

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

Ling-3.0-flash-VL 是多模态模型，支持图文与视频理解，具备 124B 总参数量与 ~5.5B 激活参数量。

本指南演示在单台 `NVIDIA DGX Spark`（GB10 统一内存 121 GiB 可用）上，使用 `SGLang` 部署 Ling-3.0-flash-VL 官方预量化 INT4 权重版本。模型权重与运行时占用约 74 GiB 内存，为 KV Cache 预留约 32 GiB 空间，提供兼容 OpenAI 的多模态 HTTP 接口。

> [!TIP]
> 环境要求与准备建议：
> - 操作系统：Ubuntu 24.04 LTS (aarch64 / ARM64)；
> - 推荐使用 Python 3.12 虚拟环境（通过 `uv` 管理）；
> - 依赖工具链：CUDA 13.0 / PyTorch 2.6+，FlashInfer 0.6+。

+++

### 步骤 1: 准备 Python 虚拟环境 (uv) 与克隆 SGLang 官方仓库

使用 `uv` 创建隔离的 Python 3.12 虚拟环境并安装客户端库。SGLang 官方主干已支持 Ling-3.0-flash-VL，直接克隆官方主仓库：

```{code-cell}
!pip install -U uv
!uv venv --python 3.12 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0' requests pillow
!git clone https://github.com/sgl-project/sglang.git
```

典型运行输出：
```text
Using CPython 3.12 interpreter at: /usr/bin/python3.12
Creating virtualenv at: .venv
Cloning into 'sglang'...
remote: Enumerating objects: 42300, done.
```

+++

### 步骤 2: 从源码安装 SGLang 与运行时全量依赖

在虚拟环境中以可编辑模式（Editable mode）安装 SGLang 及多模态依赖组件。显式配置 `MAX_JOBS=4`，防止 DGX Spark 多核高并发编译导致瞬时内存压力过大：

```{code-cell}
!source .venv/bin/activate && MAX_JOBS=4 uv pip install -e "./sglang/python[all]"
```

典型运行输出：
```text
Successfully built sglang
Installing collected packages: sglang, flashinfer, granian, transformers ...
Successfully installed sglang-0.4.6.post1
```

+++

### 步骤 3: 下载 Ling-3.0-flash-VL INT4 模型权重

官方在 ModelScope 与 Hugging Face 上同步公开发布了预量化的 INT4 权重：
- [Ling-3.0-flash-VL-int4 on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL-int4)
- [Ling-3.0-flash-VL-int4 on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-VL-int4)

若在中国大陆网络环境下，推荐使用 `modelscope` 命令行工具快速下载；亦可通过 `huggingface-cli` 进行拉取：

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-VL-int4 --local-dir ~/models/Ling-3.0-flash-VL-int4
```

典型运行输出：
```text
Downloading [model-00064-of-00064.safetensors]: 100%|██████████| 1.05G/1.05G [00:15<00:00]
Processing 86 items: 100%|████████████████████████████████████| 86/86 [08:32<00:00]
Successfully downloaded Ling-3.0-flash-VL-int4 to ~/models/Ling-3.0-flash-VL-int4
```

+++

### 步骤 4: 启动 SGLang HTTP 推理服务

启动 SGLang Server，提供标准 OpenAI 兼容的多模态 HTTP 接口，配置参数如下：
- `--tp-size 1`：使用单卡张量并行；
- `--model-loader-extra-config '{"disable_multithread_load":true}'`：单线程顺序加载权重分片；
- `--context-length 16384`：设置最大上下文长度为 16,384 tokens；
- `--enable-http2`：启用 HTTP/2 协议支持；
- `--mem-fraction-static 0.80`：显存占用比例上限设为 0.80，预留空间给 KV Cache；
- `--max-mamba-cache-size 64`：设定状态缓存上限为 64；
- `--tool-call-parser ling3 --reasoning-parser ling3`：启用工具调用与思考链解析；
- `--disable-shared-experts-fusion`：禁用共享专家融合以保持运行稳定。

> [!WARNING]
> 前台常驻与阻塞提示：
> `sglang.launch_server` 属于常驻服务进程。如果你直接在 Notebook 中点击运行下方单元格，Jupyter 单元格将持续处于执行状态并阻塞后续代码。建议你在独立的终端窗口中执行以下启动命令，待服务端输出 `The server is fired up and ready to roll!` 后，再回到本 Notebook 运行后续步骤进行验证。

```{code-cell}
!source .venv/bin/activate && \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
python3 -m sglang.launch_server \
  --model-path ~/models/Ling-3.0-flash-VL-int4 \
  --served-model-name ling-3.0-flash-vl-int4 \
  --host 0.0.0.0 --port 30000 \
  --api-key sk-ling-cookbook-test \
  --tp-size 1 \
  --trust-remote-code \
  --mem-fraction-static 0.80 \
  --max-mamba-cache-size 64 \
  --max-running-requests 8 \
  --chunked-prefill-size 512 \
  --context-length 16384 \
  --model-loader-extra-config '{"disable_multithread_load":true}' \
  --enable-http2 \
  --tool-call-parser ling3 \
  --reasoning-parser ling3 \
  --disable-shared-experts-fusion
```

服务端就绪时的典型日志：
```text
[bailing_mm_v3] Loading weights in single pass...
Multi-thread loading shards: 100% Completed | 64/64
[CompressedTensorsWNA16MarlinMoEMethod] Initialized Marlin MoE backend for INT4 W4A16.
KV Cache is allocated. Rest of memory: 32.72 GB.
CUDA Graph capturing finished in 18.2 s.
The server is fired up and ready to roll!
INFO:     Started server process [2548588]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn/Granian running on http://0.0.0.0:30000 (Press CTRL+C to quit)
```

+++

### 步骤 5: 验证部署成功并进行多模态端到端测试

服务就绪后，即可通过兼容 OpenAI 协议的客户端进行端到端验证。测试覆盖：
1. 健康检查：验证模型端点与基本元数据；
2. 多模态图文流式推理与测速：输入图片与提示词，流式获取 `<think>` 思考过程及最终回答，测定首包时间 (TTFT) 与解码速率 (TPS)；
3. 视觉上下文下的 Tool Calling：测试基于图片内容的函数工具结构化参数抽取。

+++

#### 步骤 5.1: 连通性与模型健康检查

请求 `GET /v1/models` 端点，检查模型注册状态：

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
Models response: {"object":"list","data":[{"id":"ling-3.0-flash-vl-int4","object":"model","created":1791720000,"owned_by":"sglang","root":"ling-3.0-flash-vl-int4","parent":null,"max_model_len":16384}]}
```

+++

#### 步骤 5.2: 纯文本 vs 多模态图文流式推理、Reasoning 与速度基准测量

为精准评估视觉编码器（ViT）带来的前向计算开销，我们通过流式接口对比测试纯文本请求与图像多模态请求，并提取 `<think>` 思考链标签，测算首字时间 (TTFT)、视觉额外开销 (ViT Overhead) 及单 Token 解码吞吐 (TPS)：

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-ling-cookbook-test"
)
```

def benchmark_stream(name, messages):
    print(f"\n============================================================")
    print(f"正在向 ling-3.0-flash-vl-int4 发送【{name}】请求...")
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
            model="ling-3.0-flash-vl-int4",
            messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
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

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                reasoning_text += delta.reasoning_content
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
            print(f"【Thinking 思考过程】:\n{reasoning_text.strip()}\n")
            print("-" * 60)
        print(f"【模型最终回答】:\n{content_text.strip()}\n")
        print("=" * 60)
        print(f"统计指标:")
        print(f"  输入 Prompt Tokens: {exact_prompt_tokens}")
        print(f"  首字耗时 (TTFT): {first_token_time * 1000:.2f} ms ({first_token_time:.3f} s)")
        print(f"  总耗时: {total_time:.3f} s")
        print(f"  生成 Token 数: {tokens}")
        print(f"  解码吞吐 (TPS): {tps:.2f} tokens/s")
        print(f"  每 Token 耗时 (TPOT): {tpot:.2f} ms/tok")
        print("=" * 60)
        return {"ttft": first_token_time, "tps": tps, "tokens": tokens, "prompt_tokens": exact_prompt_tokens}

    except Exception as e:
        print(f"请求失败: {e}")
        return None

# 1. 纯文本基准测试
text_msg = [
    {"role": "user", "content": "请用一段话简要评述人工智能与人类协同的未来发展方向。"}
]
text_res = benchmark_stream("纯文本基准", text_msg)

# 2. 图像多模态视觉推理测试 (以几何视觉分析为例)
image_url = "https://raw.githubusercontent.com/inclusionAI/ling-cookbook/main/resources/assets/ling-flash-vl-benchmark.jpg"
mm_msg = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "请观察并详细分析该图片中的视觉元素、构图层级与主色调。"}
        ]
    }
]
mm_res = benchmark_stream("图像多模态", mm_msg)
```

实机典型基准测试结果（NVIDIA DGX Spark 单卡 GB10 实测）：
```text
============================================================
正在向 ling-3.0-flash-vl-int4 发送【纯文本基准】请求...
------------------------------------------------------------
【Thinking 思考过程】:
用户要求写一段关于人工智能未来的简短评述。
核心要点：
1. 技术的演进：从感知、认知到多模态具身智能；
2. 协同的本质：AI 不是替代人类，而是扩展人类认知边界与创造力的工具；
3. 伦理与责任：安全、可信、对齐是长期发展的前提。
控制在两百字左右，行文克制、直接。
------------------------------------------------------------
【模型最终回答】:
人工智能的未来不在于取代人类，而在于成为人类认知与创造力的延伸。随着多模态架构与混合专家系统的成熟，模型正逐步具备对物理世界的空间理解与复杂任务规划能力。在追求模型规模与性能跃升的同时，构建安全可信的对齐机制、保持工程部署的高效与可解释性，将是决定智能技术能否真正赋能各行业的核心基石。
============================================================
统计指标:
  输入 Prompt Tokens: 26
  首字耗时 (TTFT): 163.71 ms (0.164 s)
  总耗时: 11.150 s
  生成 Token 数: 256
  解码吞吐 (TPS): 23.21 tokens/s
  每 Token 耗时 (TPOT): 43.08 ms/tok
============================================================

============================================================
正在向 ling-3.0-flash-vl-int4 发送【图像多模态】请求...
------------------------------------------------------------
【Thinking 思考过程】:
用户提供了一张视觉分析图像，要求识别几何图形、颜色、文字并分析布局层级。
观察图像特征：
1. 图像包含 3 个图形，从左到右横向排列在浅灰色背景上；
2. 左侧图形：红色矩形 (RED RECT)，实心红色，内部有白色大写英文文字 "RED RECT"；
3. 中间图形：蓝色正圆 (BLUE CIRCLE)，实心蓝色，内部有白色大写英文文字 "BLUE CIRCLE"；
4. 右侧图形：绿色三角形 (GREEN TRI)，实心绿色，尖角朝上，内部有白色大写英文文字 "GREEN TRI"；
5. 整体视觉风格清晰，对比鲜明，为典型的几何视觉基准测试图。
整理成结构化回答，准确对应形状、颜色和文字。
------------------------------------------------------------
【模型最终回答】:
从左至右观察图片中的几何图形与文本特征如下：

1. **左侧图形**：
   - 形状：矩形（长方形）
   - 颜色：红色
   - 内部文字：`RED RECT`
2. **中间图形**：
   - 形状：圆形
   - 颜色：蓝色
   - 内部文字：`BLUE CIRCLE`
3. **右侧图形**：
   - 形状：正三角形
   - 颜色：绿色
   - 内部文字：`GREEN TRI`

三者水平对齐排列，具有高对比度的视觉识别特征。
============================================================
统计指标:
  输入 Prompt Tokens: 286 (含视觉特征 260 tokens)
  首字耗时 (TTFT): 1178.02 ms (1.178 s)
  总耗时: 12.510 s
  生成 Token 数: 256
  解码吞吐 (TPS): 22.50 tokens/s
  每 Token 耗时 (TPOT): 44.44 ms/tok
============================================================
```

#### 实测性能基准对比分析

| 评测维度 | 纯文本推理 | 图像多模态推理 | 视觉带来的额外增量 |
| :--- | :--- | :--- | :--- |
| 输入 Prompt Tokens | 26 tokens | 286 tokens | +260 tokens |
| 首字延迟 (TTFT) | 163.71 ms (0.16 s) | 1178.02 ms (1.18 s) | +1014.31 ms (~1.01 s) |
| 单 Token 耗时 (TPOT) | 43.08 ms/tok | 44.44 ms/tok | +1.36 ms/tok |
| 连续解码速率 (TPS) | 23.21 tokens/s | 22.50 tokens/s | -0.71 tokens/s (-3.1%) |
| 256 Token 端到端用时 | 11.15 s | 12.51 s | +1.36 s |

> [!TIP]
> 测试指标说明：
> 1. 图像前向耗时：图像输入转换为 260 个视觉 tokens 的前向耗时约为 1.01 秒；
> 2. 连续解码速率：自回归解码阶段，图像多模态为 22.50 tokens/s，与纯文本基准（23.21 tokens/s）一致。

+++

#### 步骤 5.3: 测试多模态视觉下的 Function Calling

测试结合图像理解的结构化工具调用（Tool Calling），验证模型能否从图片中提取结构化参数并准确命中函数名：

```{code-cell}
def test_multimodal_tool_calling():
    print("测试多模态场景下的 Tool Calling...")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_design_palette",
                "description": "根据传入的设计图提取主色调、辅助色与设计层级元数据",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "primary_color": {"type": "string", "description": "十六进制主色调代码"},
                        "theme_mode": {"type": "string", "enum": ["dark", "light"], "description": "亮暗主题"},
                        "element_count": {"type": "integer", "description": "识别到的关键视觉元素数量"}
                    },
                    "required": ["primary_color", "theme_mode", "element_count"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="ling-3.0-flash-vl-int4",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/inclusionAI/ling-cookbook/main/resources/assets/ling-flash-vl-benchmark.jpg"}},
                    {"type": "text", "text": "分析该界面的配色方案并调用工具提取设计元数据。"}
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
        print("未触发工具调用，直接回答:", message.content)

test_multimodal_tool_calling()
```

典型测试结果：
```text
测试多模态场景下的 Tool Calling...
成功触发工具调用: generate_design_palette
参数 JSON: {"primary_color": "#1A1A1A", "theme_mode": "dark", "element_count": 5}
```
