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

### 步骤 3: 获取官方权重并构建 Q4_K_M GGUF 与 mmproj

为了确保多模态投影层与门控张量 100% 契合 `llama.cpp` upstream 规范（避免早期社区草案分支的键名漂移问题），推荐从 ModelScope 官方下载 BF16 原生权重并执行单次全量量化：
- [inclusionAI/Ling-3.0-flash-VL on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-VL)

```bash
# 1. 下载官方原生 BF16 权重 (233 GB)
source .venv/bin/activate
python3 -c "from modelscope import snapshot_download; snapshot_download('inclusionAI/Ling-3.0-flash-VL', local_dir='/home/squall/models/Ling-3.0-flash-VL')"

# 2. 导出官方原生 mmproj 视觉投影层 (838 MB)
python3 llama.cpp/convert_hf_to_gguf.py /home/squall/models/Ling-3.0-flash-VL \
  --mmproj --outfile /home/squall/models/mmproj-Ling-3.0-flash-VL-BF16.gguf

# 3. 转换语言主干为 BF16 GGUF 并使用 16 线程量化至 Q4_K_M (71 GB)
python3 llama.cpp/convert_hf_to_gguf.py /home/squall/models/Ling-3.0-flash-VL \
  --outtype bf16 --outfile /home/squall/models/Ling-3.0-flash-VL-BF16.gguf
./llama.cpp/build/bin/llama-quantize /home/squall/models/Ling-3.0-flash-VL-BF16.gguf \
  /home/squall/models/Ling-3.0-flash-VL-Q4_K_M.gguf Q4_K_M 16

# 4. 清理中间 248 GB 浮点 GGUF，释放磁盘空间
rm -f /home/squall/models/Ling-3.0-flash-VL-BF16.gguf
```

典型量化输出：
```text
llama_model_quantize_impl: model size  = 237405.30 MiB (16.01 BPW)
llama_model_quantize_impl: quant size  = 71783.71 MiB (4.84 BPW)
llama_quantize: quantize time = 469529.20 ms
llama_quantize:    total time = 469529.20 ms
```

+++

### 步骤 4: 启动 llama-server 推理服务

启动 `llama-server` 提供兼容 OpenAI 的 HTTP 接口，核心参数说明：
- `-m`: 指定语言模型主权重 `Ling-3.0-flash-VL-Q4_K_M.gguf`；
- `--mmproj`: 挂载多模态视觉投影权重 `mmproj-Ling-3.0-flash-VL-BF16.gguf`；
- `--alias ling-3.0-flash-vl-q4`: 注册模型别名，便于客户端直接调用；
- `--host 0.0.0.0 --port 9102`: 锁定监听端口为 9102；
- `-ngl 99`: 将全部 42 层 Transformer 与 MoE 算子卸载至 GPU；
- `-fa on`: 启用 FlashAttention 优化；
- `-c 32768`: 设置上下文长度为 32,768 tokens，预留显存给 KV Cache；
- `--jinja`: 启用 Jinja 模板引擎，原生支持 `<think>` 思考链解析与 Tool Calling。

> [!WARNING]
> 前台运行说明：
> `llama-server` 在前台常驻运行。请在独立终端执行以下命令，待输出 `HTTP server listening` 后再回到本 Notebook 执行后续验证。

```{code-cell}
!./llama.cpp/build/bin/llama-server \
  -m ~/models/Ling-3.0-flash-VL-Q4_K_M.gguf \
  --mmproj ~/models/mmproj-Ling-3.0-flash-VL-BF16.gguf \
  --alias ling-3.0-flash-vl-q4 \
  --host 0.0.0.0 \
  --port 9102 \
  --api-key sk-ling-cookbook-test \
  -ngl 99 \
  -fa on \
  -c 32768 \
  --jinja
```

典型启动日志：
```text
load_model: loading model '/home/squall/models/Ling-3.0-flash-VL-Q4_K_M.gguf'
load_model: loaded multimodal model, '/home/squall/models/mmproj-Ling-3.0-flash-VL-BF16.gguf'
load_model: initializing, n_slots = 4, n_ctx_slot = 32768, kv_unified = 'true'
init: chat template supports preserving reasoning, it is enabled by default
llama_server: listening on http://0.0.0.0:9102
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
Models response: {"object":"list","data":[{"id":"ling-3.0-flash-vl-q4","object":"model","architecture":{"input_modalities":["text","image","video"],"output_modalities":["text"]},"meta":{"vocab_type":2,"n_ctx":32768,"n_params":124414211552,"size":75270677376,"ftype":"Q4_K - Medium"},"capabilities":["completion","multimodal"]}]}
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
            model="ling-3.0-flash-vl-q4",
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
[Thinking Process]:
用户要求我用一段话简要评述人工智能与人类协同的未来发展方向。这是一个开放性的评述题，需要涵盖AI与人类协作的关键趋势、机遇与挑战，语言要精炼、有深度，同时保持客观平衡的视角...
------------------------------------------------------------
[Model Answer]:
人工智能与人类协同的未来发展方向，核心在于构建"以人类价值为主导、以技术能力为延伸"的共生关系。随着大模型、具身智能等技术的持续演进，人工智能将从当前的辅助工具逐步深化为各领域的"协作伙伴"——在医疗诊断、科学研究、创意生产及复杂决策中，AI负责处理海量数据与模式识别，人类则把控价值判断、伦理底线与战略方向，二者形成优势互补的闭环。与此同时，这一进程也伴随着对就业结构的重塑、数据隐私的保护以及算法透明度的严峻考验，因此建立完善的治理框架与人机信任机制至关重要。总体而言，未来人机协同的终极目标并非让AI取代人类，而是通过技术赋能拓展人类的认知边界与行动能力，使社会在效率提升的同时，始终守住以人为本的文明根基。
============================================================
Metrics:
  Input Prompt Tokens: 34
  TTFT: 684.60 ms (0.685 s)
  Total Duration: 6.821 s
  Generated Tokens: 276
  Decoding Throughput (TPS): 44.81 tokens/s
  Time Per Output Token (TPOT): 22.31 ms/tok
============================================================

============================================================
Sending [图像多模态] request to llama-server...
------------------------------------------------------------
[Model Answer]:
该图片呈现了一幅简洁的几何构成设计，以下是详细分析：

## 一、视觉元素分析
1. 核心图形元素：
- 红色矩形（RED RECT）：位于左上角，呈标准正方形；
- 蓝色圆形（BLUE CIRCLE）：位于右上角，纯蓝色填充；
- 绿色三角形（GREEN TRI）：位于底部中央，面积最大。
2. 文字信息：每个几何图形内部均有白色英文标签，指明颜色与形状名称。

## 二、构图层级与空间关系
呈现稳定的"品"字形倒三角构图，绿色三角形作为画面的视觉锚点与重心，上方两元素保持对称呼应。

## 三、主色调与色彩关系
采用经典三原色（RGB）高对比组合，整体呈现包豪斯（Bauhaus）极简主义风格。
============================================================
Metrics:
  Input Prompt Tokens: 295
  TTFT: 2378.58 ms (2.379 s)
  Total Duration: 27.869 s
  Generated Tokens: 1162
  Decoding Throughput (TPS): 45.55 tokens/s
  Time Per Output Token (TPOT): 21.96 ms/tok
============================================================
```

#### 实测性能基准对比分析

| 评测指标 | 纯文本流式 | 图像多模态流式 | 增量/差值 |
| :--- | :--- | :--- | :--- |
| 输入 Prompt Tokens | 34 tokens | 295 tokens | +261 tokens |
| 首字延迟 (TTFT) | 684.60 ms (0.68 s) | 2378.58 ms (2.38 s) | +1693.98 ms (~1.69 s) |
| 单 Token 耗时 (TPOT) | 22.31 ms/tok | 21.96 ms/tok | -0.35 ms/tok |
| 连续解码速率 (TPS) | 44.81 tokens/s | 45.55 tokens/s | +0.74 tokens/s (+1.7%) |
| 生成 Token 总数 | 276 tokens | 1162 tokens | +886 tokens |
| 端到端总用时 | 6.82 s | 27.87 s | - |

> [!TIP]
> 测试指标说明：
> 1. 首字延迟 (TTFT)：纯文本响应仅需 684.60 ms（服务端提示词计算仅 542.45 ms）；图像输入（含图片拉取与视觉 Tower 特征编码）首字响应为 2378.58 ms；
> 2. 连续解码速率：纯文本与图像多模态解码速率均稳定在 **44.8 ~ 45.6 tokens/s**，发挥了 MoE 稀疏激活的高效特性；
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
        model="ling-3.0-flash-vl-q4",
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
参数 JSON: {"primary_color": "#1976D2", "theme_mode": "light", "element_count": 3}
```

+++

### 步骤 6: 常见问题与运行调优

- 显存不足与上下文长度控制：
  - 现象：启动时分配 KV Cache 失败或报 `CUDA out of memory`；
  - 原因：`llama-server` 默认会尝试分配尽可能大的上下文缓存。在 121 GiB 统一内存设备上加载 71 GB 权重后，若上下文设为 131,072 会超出物理内存；
  - 解决：显式传入 `-c 32768`（或按需设为 `16384`），为系统预留 40+ GiB 物理空间。
- 社区早期 GGUF 投影层兼容报错 (`unknown projector type: bailing3vl_merger`)：
  - 现象：挂载早期社区 GGUF 的 `mmproj` 时报错 `unknown projector type` 或 LayerNorm 张量缺失；
  - 原因：社区早期权重基于 PR 合并前的草案分支导出。上游 PR #29151 最终合入时规范投影类型为 `"ling3vl"`，并将归一化层定名为 `mm.input_norm`；
  - 解决：直接从官方原生 BF16 权重调用官方 `convert_hf_to_gguf.py --mmproj` 导出，确保元数据与主干完全对齐。
- KDA 递归门控张量缺失 (`check_tensor_dims: tensor 'blk.0.ssm_f_a.weight' not found`)：
  - 现象：加载语言模型 GGUF 分片时报门控张量缺失；
  - 原因：上游 `bailingmoe3.cpp` 规范门控张量带有 `_a` 后缀（`ssm_f_a` / `ssm_g_a`），社区早期版本未带后缀；
  - 解决：采用官方转换脚本从 BF16 safetensors 统一导出并量化，即可自动生成带 `_a` 的全秩门控张量。
- 构建编译架构匹配：
  - 现象：CMake 编译出的二进制在 Blackwell 架构上执行较慢；
  - 解决：在构建时指定 `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native`，启用本地 GPU 硬件原生架构优化。
