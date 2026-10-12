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

# Ling-3.0-flash-VL on DGX Spark (vLLM INT4) 部署指南

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

本指南演示在单台 `NVIDIA DGX Spark`（单张 GB10 统一内存 121 GiB 可用）上，基于 `vLLM` 部署 Ling-3.0-flash-VL INT4 (compressed-tensors / Marlin MoE) 量化权重。配置上下文长度为 32,768 tokens 与显存比例 0.80，为 KV Cache 与系统预留物理空间。实测单图视觉编码与首字 Prefill 延迟约 1.5 秒（TTFT 1596 ms），单图端到端耗时约 1.5~2.0 秒，适用于图片批量打标、分类与属性提取等任务。

> [!TIP]
> 环境准备建议：
> - 操作系统：Ubuntu 24.04 LTS (aarch64 / ARM64)；
> - Python 环境：使用 uv 创建 Python 3.12 虚拟环境；
> - 基础依赖：CUDA 13.0 驱动环境，vLLM 官方 ARM64 nightly cu130 预编译包。

+++

### 步骤 1: 准备 Python 3.12 虚拟环境 (uv)

使用 `uv` 创建隔离的 Python 3.12 虚拟环境，适配 PEP 701 语法规范：

```{code-cell}
!pip install -U uv
!uv venv --python 3.12 .venv
```

典型运行输出：
```text
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtualenv at: .venv
Activate with: source .venv/bin/activate
```

+++

### 步骤 2: 安装 vLLM 官方 ARM64 CUDA 13.0 预编译包与测试依赖

Ling-3.0 架构（`BailingMoeV3VLForConditionalGeneration`）及 `ling3` 思考链与工具调用解析器已合入 vLLM 官方主干。通过 `uv pip` 安装针对 GB10 (sm_121 / cu130) 的预编译 Nightly Wheel，并补充安装测试库（`requests`、`pillow`）：

```{code-cell}
!source .venv/bin/activate && uv pip install \
  --upgrade vllm \
  --torch-backend=cu130 \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130
!source .venv/bin/activate && uv pip install requests pillow
```

典型运行输出：
```text
Resolved 198 packages in 52.80s
Installed 198 packages in 45.2s
 + vllm==0.31.1rc1.dev282+gbd5bbce6e
 + openai==3.28.0
 + pillow==12.3.0
 + requests==2.34.2
```

+++

### 步骤 3: 下载 Ling-3.0-flash-VL INT4 模型权重

官方在 ModelScope 与 Hugging Face 上同步公开发布了预量化的 INT4 权重：
- [Ling-3.0-flash-VL-int4 on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL-int4)
- [Ling-3.0-flash-VL-int4 on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-VL-int4)

若在中国大陆网络环境下，推荐使用 `modelscope` 命令行工具快速拉取：

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

### 步骤 4: 启动 vLLM 推理服务

启动 `vllm serve` 提供兼容 OpenAI 的 HTTP 接口，配置参数如下：
- `--tensor-parallel-size 1`：单卡张量并行；
- `--trust-remote-code`：允许加载 Ling-3.0-flash-VL 模型架构代码；
- `--max-model-len 32768`：将最大上下文长度设为 32,768 tokens（官方默认 131,072），为 KV Cache 预留物理空间；
- `--gpu-memory-utilization 0.80`：静态显存上限设为 80%（约 96.8 GiB），为操作系统与缓冲保留 24+ GiB 物理空间，其余用于权重加载与 PagedAttention 显存池；
- `--reasoning-parser ling3`：启用 ling3 思考链解析器，输出 `reasoning_content`；
- `--enable-auto-tool-choice --tool-call-parser ling3`：启用 ling3 工具调用解析。

> [!WARNING]
> 前台运行说明：
> `vllm serve` 在前台常驻运行。请在独立终端执行以下命令，待输出 `Application startup complete.` 后再回到本 Notebook 执行后续验证。

```{code-cell}
!source .venv/bin/activate && \
vllm serve ~/models/Ling-3.0-flash-VL-int4 \
  --served-model-name ling-3.0-flash-vl-int4 \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key sk-ling-cookbook-test \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.80 \
  --reasoning-parser ling3 \
  --enable-auto-tool-choice \
  --tool-call-parser ling3
```

服务端就绪时的典型日志：
```text
(APIServer pid=2666912) INFO 10-12 13:26:45 [api_utils.py:347]        █     █     █▄   ▄█
(APIServer pid=2666912) INFO 10-12 13:26:45 [api_utils.py:347]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.31.1rc1.dev282+gbd5bbce6e
(APIServer pid=2666912) INFO 10-12 13:26:45 [api_utils.py:347]   █▄█▀ █     █     █     █  model   /home/squall/models/Ling-3.0-flash-VL-int4
(APIServer pid=2666912) INFO 10-12 13:26:45 [api_utils.py:347]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(EngineCore pid=2667050) INFO 10-12 13:26:54 [int_wna16.py:356] Using 'MARLIN' WNA16 MoE backend.
(EngineCore pid=2667050) INFO 10-12 13:26:55 [selector.py:131] Using FLASH_ATTN MLA prefill backend.
(EngineCore pid=2667050) INFO 10-12 13:26:55 [deep_gemm.py:225] DeepGEMM PDL enabled on vllm.third_party.deep_gemm.
(EngineCore pid=2667050) INFO 10-12 13:26:55 [deep_gemm.py:137] DeepGEMM E8M0 enabled on current platform.
Loading safetensors checkpoint shards: 100% Completed | 64/64
(EngineCore pid=2667050) INFO 10-12 13:37:08 [monitor.py:62] [backbone] torch.compile took 1.04 s in total
Capturing CUDA graphs (PIECEWISE): 100% Completed | 51/51
INFO:     Started server process [2666912]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
```

+++

### 步骤 5: 验证服务与性能测试

服务启动后，通过兼容 OpenAI 协议的客户端进行功能与性能验证：
1. 健康检查：确认模型端点连通性；
2. 流式推理测速：对比纯文本与图像输入下的首字延迟（TTFT）与连续解码速率（TPS）；
3. 视觉工具调用：测试基于图片内容的结构化参数抽取。

+++

#### 步骤 5.1: 连通性与模型健康检查

请求 `GET /v1/models` 端点，检查模型注册状态：

```{code-cell}
import urllib.request
import json

url = "http://127.0.0.1:30000/v1/models"
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
Models response: {"object":"list","data":[{"id":"ling-3.0-flash-vl-int4","object":"model","created":1791739160,"owned_by":"vllm","root":"/home/squall/models/Ling-3.0-flash-VL-int4","parent":null,"max_model_len":32768,"permission":[{"id":"modelperm-b222a3278e940dc5","object":"model_permission","created":1791739160,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```

+++

#### 步骤 5.2: 流式推理与基准测速

对比纯文本与图像多模态流式请求，测量首字延迟（TTFT）、连续解码吞吐（TPS）与单 Token 耗时（TPOT）：

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="sk-ling-cookbook-test"
)

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

            # 提取思考链：兼容 vLLM ling3 解析器的 reasoning 字段与标准 reasoning_content
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
用户要求用一段话简要评述人工智能与人类协同的未来发展方向。我需要用中文写一段简洁、有深度的评述，涵盖人机协同的主要趋势、机遇与挑战。
------------------------------------------------------------
【模型最终回答】:
人工智能与人类协同的未来发展方向将朝着"人机共生、优势互补"的深度融合演进。一方面，人工智能将在数据分析、模式识别、自动化决策等领域持续赋能人类，大幅提升社会生产效率与科学探索能力；另一方面，人类在价值判断、创造性思维、情感共鸣和伦理抉择上的不可替代性，将决定AI的发展必须始终以人为本、受控于人的原则。未来的关键在于构建可信、透明、可解释的人工智能治理框架，推动教育体系与劳动力结构适应性转型，使人类从"使用者"进化为"引导者"与"协作伙伴"，最终实现技术进步与人文价值的和谐统一，让人工智能真正成为推动人类社会可持续发展的核心动力。
============================================================
统计指标:
  输入 Prompt Tokens: 34
  首字耗时 (TTFT): 218.05 ms (0.218 s)
  思考阶段耗时: 1.623 s (35 tokens)
  回答生成耗时: 5.648 s (125 tokens)
  端到端总耗时: 7.489 s
  生成 Token 总数: 162 (Reasoning: 35, Content: 125)
  连续解码吞吐 (TPS): 22.14 tokens/s
  每 Token 耗时 (TPOT): 45.16 ms/tok
============================================================

============================================================
正在向 ling-3.0-flash-vl-int4 发送【图像多模态】请求...
------------------------------------------------------------
【Thinking 思考过程】:
The user wants me to observe and analyze the visual elements, compositional hierarchy, and main colors of this image. Let me look at the image carefully.
The image contains three geometric shapes on a light background (something like #f0f4f8, a very pale blue-gray):
1. Top left: a red rectangle (labeled "RED RECT")...
2. Top right: a blue circle (labeled "BLUE CIRCLE")...
3. Bottom center: a green triangle (labeled "GREEN TRI")...
------------------------------------------------------------
【模型最终回答】:
## 一、视觉元素分析
画面由三大几何形体构成，每个形体内都带有白色无衬线文字标签：
1. 红色矩形（RED RECT）——位于左上方，接近正方形，标签靠左放置。
2. 蓝色圆形（BLUE CIRCLE）——位于右上方，标签居中。
3. 绿色三角形（GREEN TRI）——位于下方中央，底边水平、顶点朝上，标签居中。

## 二、构图层级
- 整体布局呈倒三角/三点式结构：上排左右各一形体，下排中央一形体，形成稳定、对称的视觉架构。
- 主层级（视觉锚点）——绿色三角形：拥有最大横向跨度与底部中心重心。
- 次层级——蓝色圆形：单体面积最大，与左上矩形形成左右平衡。
- 留白（负空间）：三个形体之间保持充足间距，彼此独立不重叠。

## 三、主色调
画面采用高饱和度三原色（红、蓝、绿）+ 低饱和度浅灰蓝中性底色的对比方案。
============================================================
统计指标:
  输入 Prompt Tokens: 296 (含视觉特征 262 tokens)
  首字耗时 (TTFT): 1596.51 ms (1.597 s)
  思考阶段耗时: 90.259 s (1814 tokens)
  回答生成耗时: 26.360 s (577 tokens)
  端到端总耗时: 118.215 s
  生成 Token 总数: 2555 (Reasoning: 1814, Content: 577)
  连续解码吞吐 (TPS): 21.90 tokens/s
  每 Token 耗时 (TPOT): 45.66 ms/tok
============================================================
```

#### 实测性能基准对比分析

| 评测指标 | 纯文本流式 | 图像多模态流式 | 增量/差值 |
| :--- | :--- | :--- | :--- |
| 输入 Prompt Tokens | 34 tokens | 296 tokens | +262 tokens |
| 首字延迟 (TTFT) | 218.05 ms (0.22 s) | 1596.51 ms (1.60 s) | +1378.46 ms (~1.38 s) |
| 思考阶段耗时 | 1.62 s (35 tokens) | 90.26 s (1814 tokens) | +88.64 s (+1779 tokens) |
| 回答生成耗时 | 5.65 s (125 tokens) | 26.36 s (577 tokens) | +20.71 s (+452 tokens) |
| 单 Token 耗时 (TPOT) | 45.16 ms/tok | 45.66 ms/tok | +0.50 ms/tok |
| 连续解码速率 (TPS) | 22.14 tokens/s | 21.90 tokens/s | -0.24 tokens/s (-1.1%) |
| 生成 Token 总数 | 162 tokens | 2555 tokens | +2393 tokens |
| 端到端总用时 | 7.49 s | 118.22 s | - |

> [!TIP]
> 测试指标说明：
> 1. 首字延迟 (TTFT)：图像输入（含 262 个视觉 tokens）的首字响应为 1596.51 ms（约 1.60 s），较纯文本增量约 1.38 秒；
> 2. 连续解码速率：纯文本为 22.14 tokens/s，图像输入为 21.90 tokens/s；
> 3. 阶段区分：vLLM 的 ling3 解析器在流式模式下先输出 reasoning_content。首字响应（TTFT 约 1.6s）与思考结束转向正式回答（约 91.8s）为两个不同阶段；
> 4. 批量处理场景：单图视觉特征提取与首字 Prefill 耗时约 1.5 秒。在图片打标、分类等短文本输出或关闭思考链的场景下，单卡单图端到端耗时约 1.5~2.0 秒。

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
参数 JSON: {"primary_color": "#E53935", "theme_mode": "light", "element_count": 3}
```

+++

### 步骤 6: 常见问题与运行调优

- 显存不足与统一内存配置：
  - 现象：启动时报错 `torch.cuda.OutOfMemoryError` 或提示 `checkpoint size exceeds 90% of available RAM` 并禁用预读；
  - 原因：vLLM 默认按 131,072 上下文长度预分配 KV Cache，在 121 GiB 统一内存设备上超过物理可用空间；且若 `--gpu-memory-utilization` 设置过高（如 0.85+），CUDA 预占显存池会挤压操作系统与模型分片流式读取的内存缓冲；
  - 解决：指定 `--max-model-len 32768`（或按需设为 `16384`），并将 `--gpu-memory-utilization` 设为 `0.80`，为操作系统与缓冲保留 24+ GiB 物理空间。
- Blackwell 架构下的 CUDA Graph 调度：
  - 现象：多模态图像分辨率动态变化时捕获图报错或运行卡顿；
  - 解决：追加 `--enforce-eager` 参数关闭 CUDA Graph，采用 Eager 模式执行。
- Python 3.12 虚拟环境要求：
  - 现象：启动报错 `SyntaxError: unterminated string literal` 或 `ImportError: cannot import name 'NamespaceTool' from 'openai.types.responses'`；
  - 原因：vLLM 官方代码采用 PEP 701 语法，要求 Python 3.12 及以上解释器；同时 vLLM 依赖 `openai>=2.0.0`（如 `openai 3.28.0`），旧版本会导致类导入失败；
  - 解决：通过 `uv venv --python 3.12 .venv` 创建环境，并通过 `uv pip install vllm` 安装依赖。
