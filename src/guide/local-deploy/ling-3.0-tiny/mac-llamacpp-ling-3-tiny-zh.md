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

# Ling-3.0-tiny on Apple Silicon Mac (llama.cpp Metal GGUF) 部署指南

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

`Ling-3.0-tiny` 是百灵大模型系列中的 7.9B 轻量 Sparse MoE 语言模型，单 Token 激活参数量仅 1.3B，原生支持 128K 长上下文。针对 Apple Silicon Mac 的统一内存架构，通过 `llama.cpp` 原生 Metal 硬件加速结合 GGUF 量化，可在消费级 Mac 上实现极致低内存占用、高吞吐的端侧推理与智能体工具调用。

本指南介绍如何在 Apple Silicon Mac（M1 / M2 / M3 / M4 系列）上，通过 Homebrew 安装 `llama.cpp` 并启动 `Ling-3.0-tiny` GGUF 版本的推理服务，提供 OpenAI 兼容的 HTTP API。

---

### 设备内存门槛与量化选型矩阵 (Hardware & Quantization Matrix)

`llama.cpp` 官方已在 `master` 主干（PR #26608）合入了对 Ling-3.0 架构的原生支持。用户可根据自身 Mac 统一内存容量选择适宜的 GGUF 量化版本：

| 部署规格 | 量化类型 | 纯权重体积 | 8K 上下文推荐内存 | 适用 Mac 设备建议 | 典型性能实测 (TPS) |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Q4_K_M (轻量推荐)** | 4-bit GGUF | **~4.80 GB** | **≥ 8 GB** | MacBook Air / Mac mini / MacBook Pro 8GB/16GB | **~123.7 tok/s (实测)** |
| **Q8_0 (高保真)** | 8-bit GGUF | **~7.80 GB** | **≥ 16 GB - 18 GB** | MacBook Pro 16GB / 18GB / 24GB | **~110.2 tok/s (实测)** |
| **BF16 (全精度完整版)** | 16-bit GGUF | **~15.80 GB** | **≥ 16 GB - 24 GB** | MacBook Pro 18GB / 24GB / 36GB+ | **~81.6 tok/s (实测)** |

> [!TIP]
> **环境与依赖建议**：
> - 推荐使用 **macOS 14+ (Sonoma / Sequoia)** 系统；
> - 推荐使用 **uv** 创建隔离的 Python 3.11 / 3.12 虚拟环境；
> - `llama.cpp` 支持全量 GPU 卸载（`-ngl all`），利用 Apple Metal API 加速矩阵计算。

+++

### 步骤 1: 准备环境与安装 llama.cpp

#### 步骤 1.1: 通过 Homebrew 安装 llama.cpp 与 uv

在 Apple Silicon Mac 上，推荐直接通过 Homebrew 安装开箱即用、原生支持 Metal 硬件加速的预编译版本：

```{code-cell}
# 安装 Homebrew 版 llama.cpp 与 uv（若已安装可跳过）
!brew install llama.cpp uv
```

典型安装输出：
```text
==> Downloading bottle manifests
✔︎ Bottle Manifest llama.cpp (0.4.0)                 Downloaded   94.0KB/ 94.0KB
==> Would install 1 formula:
llama.cpp
==> Would install 2 dependencies for llama.cpp:
libomp
ggml
==> Installing dependencies for llama.cpp: libomp and ggml
==> Installing llama.cpp dependency: libomp
==> Pouring libomp--23.1.0.arm64_tahoe.bottle.tar.gz
🍺  /opt/homebrew/Cellar/libomp/23.1.0: 11 files, 1.8MB
==> Installing llama.cpp dependency: ggml
==> Pouring ggml--0.23.0.arm64_tahoe.bottle.tar.gz
🍺  /opt/homebrew/Cellar/ggml/0.23.0: 38 files, 4.8MB
==> Installing llama.cpp
==> Pouring llama.cpp--0.4.0.arm64_tahoe.bottle.tar.gz
🍺  /opt/homebrew/Cellar/llama.cpp/0.4.0: 70 files, 19MB
```

+++

#### 步骤 1.2: 准备 Python 虚拟环境与客户端依赖

使用 `uv` 创建 Python 虚拟环境并安装测试所需的 `openai` 与 `modelscope` 库：

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0' 'modelscope>=1.18.0'
```

典型运行输出：
```text
Resolved 27 packages in 506ms
Prepared 17 packages in 903ms
Installed 17 packages in 51ms
 + httpcore==1.0.9
 + httpx==0.28.1
 + modelscope==1.39.1
 + modelscope-hub==0.4.0
 + openai==1.109.1
 + requests==2.34.2
 + tqdm==4.70.0
```

+++

### 步骤 2: 下载官方 Ling-3.0-tiny GGUF 模型权重

官方在 ModelScope 与 Hugging Face 上直接提供了预量化的 GGUF 格式模型权重：
- [Ling-3.0-tiny-GGUF on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny-GGUF)
- [Ling-3.0-tiny-GGUF on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-tiny-GGUF)

推荐直接使用 `modelscope` CLI 按需选择对应精度下载至本地 `~/models/Ling-3.0-tiny-GGUF`：

#### 选项 A：BF16 全精度版（完整版本）

原始全精度权重，无量化损失，适合统一内存 ≥ 24GB / 32GB 的 Mac 设备（如 MacBook Pro M3/M4 系列）：

> [!NOTE]
> 这一步需从 ModelScope 下载约 15.8 GB 的全精度模型权重文件，耗时较长，请耐心等待。

```{code-cell}
!mkdir -p ~/models/Ling-3.0-tiny-GGUF
!source .venv/bin/activate && uv run modelscope download \
  --model inclusionAI/Ling-3.0-tiny-GGUF Ling-3.0-tiny-bf16.gguf \
  --local-dir ~/models/Ling-3.0-tiny-GGUF
```

典型下载输出：
```text
Ling-3.0-tiny-bf16.gguf: 100%|██████████| 15.8G/15.8G [14:53<00:00, 18.5MB/s]
✓ Ling-3.0-tiny-bf16.gguf → /Users/sipan/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-bf16.gguf
```

#### 选项 B：Q8_0 量化版

高保真 8-bit 量化，适合 18GB / 24GB 内存设备：

```{code-cell}
!mkdir -p ~/models/Ling-3.0-tiny-GGUF
!source .venv/bin/activate && uv run modelscope download \
  --model inclusionAI/Ling-3.0-tiny-GGUF Ling-3.0-tiny-Q8_0.gguf \
  --local-dir ~/models/Ling-3.0-tiny-GGUF
```

典型下载输出：
```text
✓ Ling-3.0-tiny-Q8_0.gguf → /Users/sipan/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q8_0.gguf
```

#### 选项 C：Q4_K_M 量化版

4-bit 紧凑量化，适合 8GB / 16GB 消费级 Mac（如 MacBook Air / Mac mini）：

```{code-cell}
!mkdir -p ~/models/Ling-3.0-tiny-GGUF
!source .venv/bin/activate && uv run modelscope download \
  --model inclusionAI/Ling-3.0-tiny-GGUF Ling-3.0-tiny-Q4_K_M.gguf \
  --local-dir ~/models/Ling-3.0-tiny-GGUF
```

典型下载输出：
```text
✓ Ling-3.0-tiny-Q4_K_M.gguf → /Users/sipan/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q4_K_M.gguf
```

+++

### 步骤 3: 启动 llama-server HTTP 推理服务 (Metal 加速)

启动 `llama-server` 载入 GGUF 权重并开启 OpenAI 兼容的 HTTP REST 端点。部分关键参数说明：
- `-m ~/models/.../Ling-3.0-tiny-*.gguf`：指定本地 GGUF 模型文件路径
- `--alias <模型别名>`：设置对应精度的规范模型名称，统一方便下游客户端精确调用
- `-ngl all`：将所有网络层全量卸载至 Apple Metal GPU 显存
- `-fa on`：启用 FlashAttention 算子加速
- `-c 32768`：设置上下文窗口大小为 32K（可根据可用内存调整为 16K ~ 128K）
- `-np 2`：配置 2 路并发处理槽位
- `--host 127.0.0.1 --port 9102`：设置服务监听地址与端口（统一采用规范端口 `9102`）
- `--api-key sk-ling-cookbook-test`：配置客户端访问鉴权密钥

⚠️ 特别注意：

`llama-server` 默认以前台常驻模式运行。如果在 Notebook 单元格中直接运行，将阻塞执行 Kernel。**建议在独立的终端会话中运行下方启动命令**；启动成功后即可在 Notebook 中进行验证。根据你在步骤 2 中下载的权重规格，选择对应选项启动服务：

#### 选项 A：启动 BF16 全精度服务

```bash
llama-server \
  -m ~/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-bf16.gguf \
  --alias Ling-3.0-tiny-bf16 \
  -ngl all \
  -fa on \
  -c 32768 \
  -np 2 \
  --host 127.0.0.1 \
  --port 9102 \
  --api-key sk-ling-cookbook-test
```

服务端就绪时的典型日志：
```text
0.00.117.743 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.121.571 I srv    load_model: loading model '/Users/sipan/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-bf16.gguf'
0.00.311.892 W load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
0.03.146.450 I cmn          init: llama threadpool init, n_threads = 5
0.03.310.140 I srv    load_model: initializing, n_slots = 2, n_ctx_slot = 16384, kv_unified = 'false'
0.03.319.396 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
0.03.319.402 I srv  llama_server: model loaded
0.03.319.405 I srv  llama_server: listening on http://127.0.0.1:9102
```

#### 选项 B：启动 Q8_0 高保真服务

```bash
llama-server \
  -m ~/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q8_0.gguf \
  --alias Ling-3.0-tiny-Q8_0 \
  -ngl all \
  -fa on \
  -c 32768 \
  -np 2 \
  --host 127.0.0.1 \
  --port 9102 \
  --api-key sk-ling-cookbook-test
```

服务端就绪时的典型日志：
```text
0.00.128.707 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.130.297 I srv    load_model: loading model '/Users/sipan/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q8_0.gguf'
0.00.321.842 W load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
0.01.720.776 I cmn          init: llama threadpool init, n_threads = 5
0.01.762.667 I srv    load_model: initializing, n_slots = 2, n_ctx_slot = 16384, kv_unified = 'false'
0.01.770.021 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
0.01.770.030 I srv  llama_server: model loaded
0.01.770.033 I srv  llama_server: listening on http://127.0.0.1:9102
```

#### 选项 C：启动 Q4_K_M 轻量推荐服务

```bash
llama-server \
  -m ~/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q4_K_M.gguf \
  --alias Ling-3.0-tiny-Q4_K_M \
  -ngl all \
  -fa on \
  -c 32768 \
  -np 2 \
  --host 127.0.0.1 \
  --port 9102 \
  --api-key sk-ling-cookbook-test
```

服务端就绪时的典型日志：
```text
0.00.079.933 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.081.423 I srv    load_model: loading model '/Users/sipan/models/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q4_K_M.gguf'
0.00.280.360 W load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
0.01.339.747 I cmn          init: llama threadpool init, n_threads = 5
0.01.379.077 I srv    load_model: initializing, n_slots = 2, n_ctx_slot = 16384, kv_unified = 'false'
0.01.387.025 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
0.01.387.035 I srv  llama_server: model loaded
0.01.387.037 I srv  llama_server: listening on http://127.0.0.1:9102
```

+++

### 步骤 4: 验证部署成功并使用模型服务

服务启动后，即可在任意 OpenAI 兼容客户端中使用。下面提供 3 组完整端到端测试：

1. **健康检查** - 确认模型加载状态与 HTTP 端点连通性；
2. **流式推理与 `<think>` 深度思考测试** - 验证思考链生成，并测算 TTFT 与 Decode TPS 吞吐基准；
3. **Function Calling 工具调用测试** - 验证结构化参数解析与工具调用功能。

+++

#### 步骤 4.1: 连通性与模型健康检查

请求 `GET /v1/models` 查看当前运行的模型信息：

```{code-cell}
import urllib.request
import json

url = "http://127.0.0.1:9102/v1/models"
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
```json
Health check status: 200
Models response: {"models":[{"name":"Ling-3.0-tiny","model":"Ling-3.0-tiny","modified_at":"","size":"","digest":"","type":"model","description":"","tags":[""],"capabilities":["completion"],"parameters":"","details":{"parent_model":"","format":"gguf","family":"","families":[""],"parameter_size":"","quantization_level":""}}],"object":"list","data":[{"id":"Ling-3.0-tiny","aliases":["Ling-3.0-tiny"],"tags":[],"object":"model","created":1788805461,"owned_by":"llamacpp","meta":{"vocab_type":true,"n_vocab":157184,"n_ctx":16384,"n_ctx_train":131072,"n_embd":1536,"n_params":7893392800,"size":15796958848,"ftype":"BF16"}}]}
```

+++

#### 步骤 4.2: 流式推理、思考链提取与速度测量

使用 OpenAI Python SDK 调用接口，通过 `enable_thinking: True` 提取 `<think>` 思考链内容，并统计端侧 TTFT 与 Decode TPS 吞吐表现：

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9102/v1",
    api_key="sk-ling-cookbook-test"
)

# 可按需替换为你所启动的规格（如 "Ling-3.0-tiny-bf16", "Ling-3.0-tiny-Q8_0" 或 "Ling-3.0-tiny-Q4_K_M"）
MODEL_NAME = "Ling-3.0-tiny-bf16"

def verify_streaming_and_thinking():
    prompt = "计算 17 × 23 的结果，请给出逐步推导逻辑。"
    print(f"Sending prompt: '{prompt}' to model '{MODEL_NAME}'...")

    start_time = time.time()
    first_token_time = None
    total_tokens = 0
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
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
                total_tokens += 1

        end_time = time.time()
        ttft = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
        decode_duration = end_time - first_token_time if first_token_time else 0.001
        tps = total_tokens / decode_duration

        print("\n=== Latency & Throughput Metrics ===")
        print(f"TTFT (Time to First Token): {ttft:.2f} ms")
        print(f"Decode TPS (Tokens/s): {tps:.2f} t/s")
        print(f"Total Generated Tokens: {total_tokens}")
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

典型测试结果（Apple Silicon Mac 实测）：
```text
Sending prompt: '计算 17 × 23 的结果，请给出逐步推导逻辑。' to model 'Ling-3.0-tiny'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 395.72 ms
Decode TPS (Tokens/s): 81.57 t/s
Total Generated Tokens: 2377
Total Duration: 29.54 s

=== Extracted Reasoning Chain (<think>) ===
1. 分析请求：
   - 任务：计算 17 × 23。
   - 要求：提供逐步推导逻辑。

2. 选择方法：
   - 方法一：拆分因数法（分配律），适合展示数学逻辑。
   - 方法二：列竖式乘法（标准算法），适合展示计算步骤。

3. 起草逐步推导：
   - 将 17 拆分为 10 + 7
   - 应用乘法分配律：(10 + 7) × 23 = 10 × 23 + 7 × 23
   - 计算两项：10 × 23 = 230，7 × 23 = 161
   - 相加：230 + 161 = 391

=== Final Response Content ===
计算 17 × 23 的结果，可以通过以下两种逐步推导逻辑来完成：

### 方法一：拆分因数法（乘法分配律）
1. 将 17 拆分为 10 和 7：17 = 10 + 7
2. 根据乘法分配律拆开：17 × 23 = (10 + 7) × 23 = 10 × 23 + 7 × 23
3. 分别计算：10 × 23 = 230，7 × 23 = 161
4. 相加求和：230 + 161 = 391

### 方法二：列竖式乘法（标准算法）
1. 用个位数 3 乘以 17 得到第一行：51
2. 用十位数 2 乘以 17 得到第二行：340
3. 两行相加：51 + 340 = 391

### 最终结果
17 × 23 = 391
```

+++

#### 步骤 4.3: 测试 Function Calling 工具调用

传入标准 Function Calling Schema（以汇率换算为例），验证模型对结构化工具调用的原生支持：

```{code-cell}
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9102/v1",
    api_key="sk-ling-cookbook-test"
)

# 可按需替换为你所启动的规格（如 "Ling-3.0-tiny-bf16", "Ling-3.0-tiny-Q8_0" 或 "Ling-3.0-tiny-Q4_K_M"）
MODEL_NAME = "Ling-3.0-tiny-bf16"

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
    print(f"Testing Function Calling with model '{MODEL_NAME}'...")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "请帮我把 100 美元兑换成人民币，算一下能换多少？"}
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
Testing Function Calling with model 'Ling-3.0-tiny-bf16'...

=== Function Call Output Detected ===
Tool Call ID: call_cur_9823f0a1
Function Name: convert_currency
Arguments JSON: {"from_currency": "USD", "to_currency": "CNY", "amount": 100}
```

+++

#### 步骤 4.4: 三档精度解码性能与常驻内存实测对比

以下为在 Apple Silicon Mac（M5 Pro，48GB 统一内存，macOS 15）上对 `Ling-3.0-tiny` 三种精度（BF16、Q8_0、Q4_K_M）进行端到端流式推理与显存采样的完整实测基准数据：

| 模型规格 | 精度类型 | 权重磁盘体积 | 首 Token 延迟 (TTFT) | 解码吞吐 (Decode TPS) | 初始常驻内存 (Idle RSS) | 推理峰值内存 (Peak RSS) | 动态显存增量 (KV Cache) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Ling-3.0-tiny-bf16`** | BF16 (全精度完整版) | **15.8 GB** | 395.72 ms | **81.57 tok/s** | 11.88 GB | **12.07 GB** | 0.18 GB (~188 MB) |
| **`Ling-3.0-tiny-Q8_0`** | 8-bit GGUF (高保真) | **7.8 GB** | 271.69 ms | **110.18 tok/s** | 8.13 GB | **8.17 GB** | 0.04 GB (~43 MB) |
| **`Ling-3.0-tiny-Q4_K_M`** | 4-bit GGUF (轻量推荐) | **4.8 GB** | 248.72 ms | **123.69 tok/s** | 4.17 GB | **4.21 GB** | 0.04 GB (~46 MB) |

> [!TIP]
> **选型与部署建议**：
> - **8GB / 16GB 内存设备**（如 MacBook Air / Mac mini / MacBook Pro 8GB/16GB）：首选 **`Q4_K_M`**，峰值物理内存仅需 **~4.2 GB**，解码吞吐高达 **~123.7 tok/s**，响应极其迅捷；
> - **18GB / 24GB 内存设备**（如 MacBook Pro M3/M4 系列）：推荐 **`Q8_0`**，高保真 8-bit 量化保留了近乎无损的数学与逻辑推理能力，解码吞吐达到 **~110.2 tok/s**，峰值物理显存仅需 **~8.2 GB**；
> - **32GB+ 内存设备**（如 MacBook Pro M-Max 系列）：可直接使用 **`BF16`** 完整版本，无任何精度损失，实测解码吞吐依然稳定在 **~81.6 tok/s**，物理常驻显存仅需 **~12.1 GB**。

+++

### 步骤 5: 常见问题与故障排查 (Troubleshooting)

1. **Metal 统一内存分配上限限制**：
   - 现象：macOS 系统默认将单个进程的 Metal 显存分配上限限制为物理统一内存的约 75%。在加载超大上下文时可能提示 `ggml_metal_init: error allocating buffer`。
   - 解决：如需突破默认显存限制，可通过系统命令临时调整上限（需管理员权限）：
     ```bash
     sudo sysctl iogpu.wired_mem_limit=30000000000  # 调整为 ~30GB
     ```

2. **上下文显存溢出 (Context OOM)**：
   - 现象：在 8GB 或 16GB 内存设备上，配置过高的 `-c 131072` 与 `-np 4` 导致启动崩溃或卡死。
   - 解决：轻量设备推荐将上下文长度收敛至 `-c 32768` 或 `-c 16384`，并发槽位保持 `-np 1` 或 `-np 2`。

3. **端口冲突 (Port 9102 occupied)**：
   - 现象：`llama-server` 启动报错 `failed to bind port 9102` 或 `Address already in use`。
   - 解决：通过终端查询占用进程并清理：
     ```bash
     lsof -i :9102
     kill -9 <PID>
     ```
     或在启动参数中修改 `--port 9103`。
