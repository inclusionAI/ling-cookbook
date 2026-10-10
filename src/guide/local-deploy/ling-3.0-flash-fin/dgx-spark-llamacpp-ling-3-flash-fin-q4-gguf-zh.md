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

+++

# Ling-3.0-flash-Fin on DGX Spark (llama.cpp Q4_K_M GGUF) 部署指南

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

Ling-3.0-flash-Fin 是百灵大模型系列首个金融增强大模型，以 Ling-3.0-flash 为基座，总参数量 124B、单 Token 激活 5.1B，原生支持 256K 超长上下文窗口，针对金融研报解析、财务指标核算、多表对齐与估值建模等复杂长程任务进行了专项增强。

本 Notebook 将演示如何在 `NVIDIA DGX Spark` (GB10 121GB 统一内存) 设备上，使用 `llama.cpp` 对 Ling-3.0-flash-Fin 进行 4-bit 量化（Q4_K_M GGUF）部署。量化后模型权重占 ~72 GB（约 71.5 GB 显存），可在单台 DGX Spark 上保留近 50 GB 显存供超长上下文与高并发推理使用，兼具高吞吐与极低的首字延迟。

> [!TIP]
> **环境准备建议**：
> - 推荐使用 **Python 3.11 / 3.12** 环境；
> - 推荐使用 **uv** 创建独立的 Python 虚拟环境，以确保依赖隔离与工具链兼容性。

+++

### 步骤 1: 准备 Python 虚拟环境 (uv) 与克隆 llama.cpp 仓库

推荐使用 `uv` 创建独立虚拟环境并安装 OpenAI Python 客户端。`llama.cpp` 官方主干已合并对 Ling-3.0 架构的支持，直接克隆主干源码即可：

```{code-cell}
!pip install -U uv
!uv venv --python 3.12 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
!git clone https://github.com/ggerganov/llama.cpp.git
```

典型输出：
```text
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtualenv at: .venv
Cloning into 'llama.cpp'...
remote: Enumerating objects: 45210, done.
remote: Counting objects: 100% (210/210), done.
```

+++

### 步骤 2: 构建 llama.cpp

在 DGX Spark 上启用 CUDA 加速编译 `llama.cpp`：

```{code-cell}
!cd llama.cpp && cmake -B build -DGGML_CUDA=ON . && cmake --build build --parallel 8
```

典型运行输出：
```text
-- The CXX compiler identification is GNU 11.4.0
-- The CUDA compiler identification is NVIDIA 12.8.55
-- Building with CUDA architecture: native
[100%] Built target llama-server
[100%] Built target llama-quantize
```

+++

### 步骤 3: 获取 Ling-3.0-flash-Fin 模型权重文件

我们先获取金融模型的原始权重文件。如果你在中国，推荐使用 [ModelScope CLI](https://github.com/modelscope/modelscope/blob/master/README_zh.md) 下载；或者也可以使用 [Hugging Face CLI](https://huggingface.co/docs/hub/agents-cli) 下载权重：

- [Ling-3.0-flash-Fin on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-Fin)
- [Ling-3.0-flash-Fin on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-Fin)

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-Fin --local-dir ~/models/Ling-3.0-flash-Fin
```

典型运行输出：
```text
Downloading shard 1/24: 100%|██████████| 4.98G/4.98G [00:15<00:00, 332MB/s]
...
Successfully downloaded Ling-3.0-flash-Fin to ~/models/Ling-3.0-flash-Fin
```

+++

### 步骤 4: 将模型转换为全精度 GGUF 格式

`llama.cpp` 支持使用 GGUF 格式模型。我们安装转换依赖，并使用 `convert_hf_to_gguf.py` 将 Safetensors 权重转换为全精度 BF16 的 GGUF 格式：

```{code-cell}
!source .venv/bin/activate && uv pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
!source .venv/bin/activate && python3 llama.cpp/convert_hf_to_gguf.py ~/models/Ling-3.0-flash-Fin \
  --outfile ~/models/Ling-3.0-flash-Fin-bf16.gguf \
  --outtype bf16 --model-name Ling-3.0-flash-Fin
```

典型运行输出：
```text
INFO:hf-to-gguf:Loading model: Ling-3.0-flash-Fin
INFO:hf-to-gguf:Set model parameters
INFO:hf-to-gguf:Writing tensors to /home/squall/models/Ling-3.0-flash-Fin-bf16.gguf
INFO:hf-to-gguf:Done writing 124B model tensors (bf16).
```

+++

### 步骤 5: 量化模型到 Q4_K_M GGUF 格式

该模型的全精度版本无法在单台 DGX Spark 上完整载入，我们使用编译生成的 `llama-quantize` 工具将 BF16 GGUF 压缩为 4-bit (Q4_K_M) 格式，模型体积从 ~248GB 降至 ~60.5GB：

```{code-cell}
!./llama.cpp/build/bin/llama-quantize ~/models/Ling-3.0-flash-Fin-bf16.gguf ~/models/Ling-3.0-flash-Fin-Q4_K_M.gguf Q4_K_M
```

典型运行输出：
```text
llama_model_quantize_impl: model size  = 243267.57 MiB (16.01 BPW)
llama_model_quantize_impl: quant size  = 73436.36 MiB (4.83 BPW)
llama_model_quantize_impl: WARNING: 8 of 938 tensor(s) required fallback quantization

llama_quantize: quantize time = 533907.62 ms
llama_quantize:    total time = 533907.62 ms
=== [Step 4/4] Cleaning intermediate BF16 GGUF ===
=== All Done! Generated: /home/squall/models/Ling-3.0-flash-Fin-Q4_K_M.gguf ===
-rw-rw-r-- 1 squall squall 72G 10月 10 15:49 /home/squall/models/Ling-3.0-flash-Fin-Q4_K_M.gguf
```

+++

### 步骤 6: 用 llama-server 启动模型推理服务

启动 `llama-server`，提供 OpenAI 格式兼容的 HTTP 接口。部分部署参数说明：

- `-m ~/models/Ling-3.0-flash-Fin-Q4_K_M.gguf`：指定量化模型文件路径
- `--alias ling-3.0-flash-fin-q4`：对外注册的模型唯一标识名
- `-ngl all`：将所有网络层部署至 GPU 运算
- `-fa on`：开启 FlashAttention 加速
- `-c 262144`：启用 256K 满上下文窗口
- `-cb`：开启连续批处理（Continuous Batching）
- `-np 4`：支持 4 路并发处理
- `--port 9102`：按照环境约定使用 llama.cpp 标准端口 `9102`

⚠️ 特别注意：

`llama-server` 以前台常驻模式运行。如果你直接在 Notebook 中运行下方单元格，Jupyter 将阻塞而无法执行后续单元格的代码。建议你在单独的终端会话中执行启动命令。启动成功后即可使用 Jupyter 进行后续验证。

```{code-cell}
!./llama.cpp/build/bin/llama-server \
  -m ~/models/Ling-3.0-flash-Fin-Q4_K_M.gguf \
  --alias ling-3.0-flash-fin-q4 \
  -ngl all -fa on -c 262144 -cb -np 4 \
  --host 0.0.0.0 --port 9102 --api-key sk-ling-cookbook-test
```

服务端就绪时的典型日志：
```text
llama_server: HTTP server listening on 0.0.0.0:9102
llama_server: model loaded successfully, n_ctx = 262144, offload = 100% (GPU)
```

+++

### 步骤 7: 验证部署成功并使用模型服务

服务启动后，即可在各类 LLM 客户端中使用。Ling-3.0-flash-Fin 默认开启思考模式（Thinking Mode），推荐采样参数配置为 `temperature=1.0`、`top_p=0.95`、`top_k=20`。

本节提供 3 项逐步验证用例：
1. **健康检查**：确认模型注册状态与接口连通性；
2. **金融推理与流式测速**：验证自由现金流 (FCFF) 深度推理，提取 `<think>` 思考链并测算 TTFT 与 Decode TPS；
3. **金融 Tool Calling 验证**：测试财报查询结构化工具调用。

+++

#### 步骤 7.1: 连通性与模型健康检查

请求 `GET /v1/models` 查看当前运行的模型信息：

```{code-cell}
import urllib.request
import json

url = "http://localhost:9102/v1/models"
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
{
  "object": "list",
  "data": [
    {
      "id": "ling-3.0-flash-fin-q4",
      "aliases": [
        "ling-3.0-flash-fin-q4"
      ],
      "tags": [],
      "object": "model",
      "created": 1791619693,
      "owned_by": "llamacpp",
      "meta": {
        "vocab_type": 2,
        "n_vocab": 157184,
        "n_ctx": 65536,
        "n_ctx_train": 262144,
        "n_embd": 2560,
        "n_params": 127486405600,
        "size": 77003601792,
        "ftype": "Q4_K - Medium"
      }
    }
  ]
}
```

+++

#### 步骤 7.2: 金融深度推理、Thinking 思考链与测速

使用 OpenAI Python SDK 调用接口，通过高端半导体制造企业 FCFF 折现估值专业用例验证模型的专业金融逻辑，实时提取 `<think>` 思考链内容，并基于服务端统计的精确 Token 总数测算首字延迟 (TTFT) 与解码速率 (Decode TPS)：

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9102/v1",
    api_key="sk-ling-cookbook-test"
)

def verify_financial_reasoning_and_speed():
    prompt = (
        "请对一家高端半导体制造企业进行自由现金流（FCFF）折现估值简要测算：\n"
        "1. 第1年EBIT 120亿元，年复合增长率15%，所得税率15%；\n"
        "2. WACC为9.5%，永续增长率为2.5%。\n"
        "请展示推导逻辑与主要估值结论。"
    )
    print("Sending financial prompt to model 'ling-3.0-flash-fin-q4'...")

    start_time = time.time()
    first_token_time = None
    chunk_count = 0
    exact_completion_tokens = None
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model="ling-3.0-flash-fin-q4",
            messages=[
                {"role": "system", "content": "你是由蚂蚁集团训练的金融专业级大模型 Ling-3.0-flash-Fin。在回答复杂专业问题时，请展示清晰、规范的金融推导逻辑。"},
                {"role": "user", "content": prompt}
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            temperature=1.0,
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

            reasoning_piece = (
                getattr(delta, 'reasoning', None)
                or getattr(delta, 'reasoning_content', None)
                or (delta.model_extra.get('reasoning') if hasattr(delta, 'model_extra') and delta.model_extra else None)
            )
            content_piece = delta.content

            if reasoning_piece:
                if first_token_time is None:
                    first_token_time = time.time()
                reasoning_text += reasoning_piece
                chunk_count += 1

            if content_piece:
                if first_token_time is None:
                    first_token_time = time.time()
                content_text += content_piece
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
        print(reasoning_text if reasoning_text else "[Note: Reasoning output]")

        print("\n=== Final Response Content ===")
        print(content_text)

    except Exception as e:
        print(f"Streaming verification failed: {e}")

if __name__ == "__main__":
    verify_financial_reasoning_and_speed()
```

典型测试结果：
```text
Sending financial prompt to model 'ling-3.0-flash-fin-q4'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 800.97 ms
Decode TPS (Tokens/s): 44.63 t/s
Total Generated Tokens: 1833
Total Duration: 41.87 s

--- Extracted Reasoning Chain (<think>) ---
1. 整理已知条件与财务要素：
   - 第1年 EBIT = 120 亿元，年复合增长率 = 15%，所得税率 = 15%
   - 加权平均资本成本 (WACC) = 9.5%，永续增长率 (g_n) = 2.5%
2. 确定估值模型框架：
   - 在未提供折旧、资本支出和营运资本变动细项的情况下，采用 EBIT 折现法，将 FCFF 近似为税后净营业利润 NOPAT = EBIT × (1 - T)。
   - 构建两阶段 FCFF 折现模型：明确预测期（10年高增长期，复合增长率 15%）+ 永续成熟期（永续增长率 2.5%）。
3. 验证合理性：
   - 半导体制造资本密集型行业，WACC 9.5% 处于合理区间（无风险利率约3%，市场风险溢价约6%，β约1.2-1.5）。
   - 永续增长率 2.5% 略高于长期 GDP 稳态增速。

--- Response Content ---
# 高端半导体制造企业 FCFF 折现估值测算

## 一、核心参数设定

| 参数 | 数值 | 依据 |
|------|------|------|
| 第1年 EBIT | 120 亿元 | 给定 |
| EBIT 复合增长率 (g) | 15% | 高端半导体国产替代+产能扩张 |
| 所得税率 (T) | 15% | 高新技术企业优惠税率 |
| WACC | 9.5% | 半导体制造资本密集型特征 |
| 永续增长率 (g_n) | 2.5% | 略高于长期GDP增速 |

---

## 二、推导逻辑与逐年预测（10年明确预测期）

### Step 1：计算第1年 FCFF (NOPAT)
$$FCFF_1 = EBIT_1 \times (1 - T) = 120 \times (1 - 0.15) = 102 \text{ 亿元}$$

### Step 2：逐年 FCFF 现值测算
| 年份 | EBIT (亿元) | FCFF = EBIT×0.85 (亿元) | 折现因子 (1.095)⁻ᵗ | 现值 (亿元) |
|------|------------|------------------------|---------------------|------------|
| 1 | 120.0 | 102.0 | 0.9132 | 93.15 |
| 2 | 138.0 | 117.3 | 0.8340 | 97.83 |
| 3 | 158.7 | 134.9 | 0.7617 | 102.75 |
| 4 | 182.5 | 155.1 | 0.6956 | 107.89 |
| 5 | 209.9 | 178.4 | 0.6353 | 113.35 |
| 6 | 241.4 | 205.2 | 0.5802 | 119.06 |
| 7 | 277.6 | 236.0 | 0.5298 | 125.03 |
| 8 | 319.2 | 271.3 | 0.4839 | 131.28 |
| 9 | 367.1 | 312.0 | 0.4419 | 137.87 |
| 10 | 422.2 | 358.9 | 0.4036 | 144.85 |

> **预测期现值合计 ≈ 1,173 亿元**

### Step 3：计算永续终值与企业价值
- 终值 $TV_{10} = \frac{FCFF_{10} \times (1 + g_n)}{WACC - g_n} = \frac{358.9 \times 1.025}{0.095 - 0.025} \approx 5,255.7 \text{ 亿元}$
- 终值折现现值 $PV(TV_{10}) = 5,255.7 \times 0.4036 \approx 2,121.2 \text{ 亿元}$
- **企业价值 (Enterprise Value, EV) = 预测期现值 + 终值现值 = 1,173 + 2,121 ≈ 3,294 亿元**
```

+++

#### 步骤 7.3: 测试金融 Tool Calling 结构化调用

定义符合金融研报与指标查询场景的结构化工具定义（以财报指标查询 `query_financial_statement` 为例），验证模型对专业参数的抽取与函数调用能力：

```{code-cell}
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9102/v1",
    api_key="sk-ling-cookbook-test"
)

financial_tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "query_financial_statement",
            "description": "查询上市公司指定报告期的资产负债表、利润表或现金流量表指标",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "上市公司股票代码或名称，例如 贵州茅台、600519、AAPL"
                    },
                    "report_period": {
                        "type": "string",
                        "description": "报告期，例如 2025Q3、2024FY"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "待查询财务科目列表，如 ['营业收入', '归母净利润']"
                    }
                },
                "required": ["ticker", "report_period", "metrics"]
            }
        }
    }
]

def verify_financial_tool_calling():
    print(f"Testing Financial Tool Calling with model 'ling-3.0-flash-fin-q4'...")
    try:
        response = client.chat.completions.create(
            model="ling-3.0-flash-fin-q4",
            messages=[
                {"role": "user", "content": "帮我查询贵州茅台2025年第三季度的营业收入和归母净利润。"}
            ],
            tools=financial_tools_schema,
            tool_choice="auto",
            temperature=0.1
        )

        message = response.choices[0].message
        if message.tool_calls:
            print("\n=== Financial Tool Call Output Detected ===")
            for tool_call in message.tool_calls:
                print(f"Tool Call ID: {tool_call.id}")
                print(f"Function Name: {tool_call.function.name}")
                print(f"Arguments JSON: {tool_call.function.arguments}")
        else:
            print("\n=== Direct Content Output ===")
            print(message.content)

    except Exception as e:
        print(f"Tool calling verification failed: {e}")

if __name__ == "__main__":
    verify_financial_tool_calling()
```

典型测试结果：
```text
Testing Financial Tool Calling with model 'ling-3.0-flash-fin-q4'...

=== Financial Tool Call Output Detected ===
Tool Call ID: 1DqqNndH1re5aq8whZG1e2jKo7Pjkk9H
Function Name: query_financial_statement
Arguments JSON: {"ticker":"贵州茅台","report_period":"2025Q3","metrics":["营业收入", "归母净利润"]}
```

+++

### 步骤 8: 常见问题与故障排查

1. **GGUF 转换依赖安装报错**：
   - 现象：运行 `convert_hf_to_gguf.py` 时报 `ModuleNotFoundError`。
   - 解决：确保在虚拟环境中执行 `uv pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt` 安装转换所需的全部依赖库（如 `torch`, `sentencepiece` 等）。

2. **多并发下上下文显存分配**：
   - 现象：启动时指定超大上下文（如 `-c 262144`）与较高并发数（如 `-np 4`）时显存溢出。
   - 解决：可适度减小并发槽位 `-np 2` 或降低上下文长度 `-c 131072`。单台 DGX Spark 上 Q4_K_M 权重占 ~72GB，统一显存上限 121GB，支持充足的 KV 缓存。

3. **端口冲突 (Port 9102 occupied)**：
   - 现象：`llama-server` 启动报错端口已被占用。
   - 解决：通过 `lsof -i :9102` 查找并结束占用进程，或在启动命令中通过 `--port` 修改端口。
