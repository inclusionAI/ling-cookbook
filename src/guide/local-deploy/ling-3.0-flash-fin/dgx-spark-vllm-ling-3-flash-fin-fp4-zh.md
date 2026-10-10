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

# Ling-3.0-flash-Fin on DGX Spark (vLLM FP4) 部署指南

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

本 Notebook 将演示如何在 `NVIDIA DGX Spark` (GB10 121GB 统一内存) 上，基于 vLLM 官方针对 ARM64 CUDA 13.0 构建的预编译版本，部署 Ling-3.0-flash-Fin 的 FP4 (MXFP4) 量化版本。量化后模型权重仅占 ~60.5 GB，结合 DeepGEMM MXFP4 硬件加速算子与 Triton MLA 注意力实现，可在单台 DGX Spark 上高效兼顾高推理吞吐、低首字延迟与超长上下文显存富余。

> [!TIP]
> **环境准备建议**：
> - 推荐使用 **Python 3.12** 环境；
> - 推荐使用 **uv** 创建独立的 Python 虚拟环境，以确保依赖隔离与算子兼容性。

+++

### 步骤 1: 准备 Python 3.12 虚拟环境 (uv)

推荐使用 `uv` 创建独立的 Python 3.12 虚拟环境，确保依赖隔离；同时安装 `openai` 客户端用于后续服务接口测试：

```{code-cell}
!pip install -U uv
!uv venv --python 3.12 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
```

典型运行输出：
```text
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtualenv at: .venv
Activate with: source .venv/bin/activate
Resolved 20 packages in 120ms
Installed 20 packages in 45ms
 + openai==1.60.0
```

+++

### 步骤 2: 安装 vLLM 官方 ARM64 CUDA 13.0 预编译版本

Ling-3.0 架构支持与 `ling3` 解析器已合并至 vLLM 官方源。通过 `uv pip` 指定 `--torch-backend=cu130` 即可快速安装针对 DGX Spark GB10（sm_121 / cu130）构建的优化预编译包：

```{code-cell}
!uv pip install \
  --upgrade vllm \
  --torch-backend=cu130 \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130
```

典型运行输出：
```text
⠇ Resolving dependencies...
Installed 180 packages in 114ms
 + aiohappyeyeballs==2.7.1
 + aiohttp==3.14.3
Successfully installed vllm-0.7.0
```

+++

### 步骤 3: 下载 Ling-3.0-flash-Fin FP4 模型权重

从 ModelScope 或 Hugging Face 下载官方发布的 `inclusionAI/Ling-3.0-flash-Fin-fp4` 预量化权重至本地统一模型目录 `~/models/Ling-3.0-flash-Fin-fp4`：

- [Ling-3.0-flash-Fin-fp4 on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-Fin-fp4)
- [Ling-3.0-flash-Fin-fp4 on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-Fin-fp4)

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-Fin-fp4 --local-dir ~/models/Ling-3.0-flash-Fin-fp4
```

典型运行输出：
```text
Downloading snapshot of inclusionAI/Ling-3.0-flash-Fin-fp4 (model)...
Downloading [model-00065-of-00065.safetensors]: 100%|██████████| 1.02G/1.02G [00:12<00:00]
Processing 77 items: 100%|███████████████████| 77.0/77.0 [12:35<00:00, 9.8s/it]
Successfully downloaded Ling-3.0-flash-Fin-fp4 to ~/models/Ling-3.0-flash-Fin-fp4
```

+++

### 步骤 4: 启动 vLLM HTTP 推理服务

启动 `vllm` 加载模型，并开启 OpenAI 兼容的 HTTP API 服务端。

关键启动参数说明：
- `--served-model-name ling-3.0-flash-fin-fp4`：对外注册的模型唯一标识名
- `--port 30000`：推理服务统一监听端口
- `--gpu-memory-utilization 0.85`：预留充足显存（约 108GB 上限），除 ~64.15GB 权重外提供近 35GB 空间供 PagedAttention KV Cache（支持 4,288,818 tokens 缓存，可支撑 256K 满上下文 16.36x 并发）
- `--trust-remote-code`：允许加载百灵混合注意力 (KDA + MLA) 结构代码
- `--reasoning-parser ling3`：将 `<think>` 思考链自动分离至 `delta.reasoning`
- `--tool-call-parser ling3` 与 `--enable-auto-tool-choice`：将原生 `<tool_call>` 自动解析为 OpenAI 标准的 `tool_calls` 结构化对象

⚠️注意： `vllm` 默认以前台常驻模式运行。如果在 Jupyter Notebook 单元格中直接运行，将持续占用执行 Kernel，无法操作其他单元格。你需要在独立终端会话中执行下面的启动命令。

```{code-cell}
!uv run vllm serve ~/models/Ling-3.0-flash-Fin-fp4 \
  --served-model-name ling-3.0-flash-fin-fp4 \
  --port 30000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --reasoning-parser ling3 \
  --enable-auto-tool-choice \
  --tool-call-parser ling3 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --api-key sk-ling-cookbook-test
```

服务端就绪时的典型日志：
```text
(EngineCore pid=992724) INFO 10-10 13:41:42 [__init__.py:733] Selected DeepGemmFp8BlockScaledMMKernel for Fp8LinearMethod
(EngineCore pid=992724) INFO 10-10 13:41:42 [deep_gemm.py:225] DeepGEMM PDL enabled on vllm.third_party.deep_gemm.
(EngineCore pid=992724) INFO 10-10 13:41:42 [deep_gemm.py:137] DeepGEMM E8M0 enabled on current platform.
(EngineCore pid=992724) INFO 10-10 13:41:42 [mxfp4.py:806] Using 'DEEPGEMM_MXFP4' Mxfp4 MoE backend.
(EngineCore pid=992724) INFO 10-10 13:41:44 [cuda.py:506] Using TRITON_MLA attention backend out of potential backends: ['TRITON_MLA'].
(EngineCore pid=992724) INFO 10-10 13:41:44 [selector.py:131] Using FLASH_ATTN MLA prefill backend.
Loading safetensors checkpoint shards: 100% Completed | 65/65 [01:28<00:00,  1.37s/it]
(EngineCore pid=992724) INFO 10-10 13:43:16 [default_loader.py:498] Loading weights took 89.00 seconds
(EngineCore pid=992724) INFO 10-10 13:43:21 [model_runner.py:416] Model loading took 64.15 GiB memory and 99.891893 seconds
(EngineCore pid=992724) INFO 10-10 13:44:07 [worker.py:230] # GPU blocks: 268051, # CPU blocks: 1092
(EngineCore pid=992724) INFO 10-10 13:44:07 [worker.py:235] Maximum concurrency for 262144 tokens per request: 16.36x
(APIServer pid=992488) INFO 10-10 13:49:19 [entry.py:140] Starting vLLM server on http://0.0.0.0:30000
(APIServer pid=992488) INFO 10-10 13:49:19 [launcher.py:195] Route: /v1/chat/completions, Methods: POST
(APIServer pid=992488) INFO 10-10 13:49:19 [launcher.py:195] Route: /v1/models, Methods: GET
INFO:     Started server process [992488]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     vLLM API server running on http://0.0.0.0:30000
```

+++

### 步骤 5: 验证服务与接口调用

服务启动后，可通过 OpenAI 兼容客户端进行端到端验证：

1. **健康检查**：确认模型端点连通性；
2. **金融深度推理与流式测速**：验证自由现金流 (FCFF) 折现估值专业推理，实时分离 `<think>` 思考链，并精确测算 TTFT 与 Decode TPS；
3. **金融 Tool Calling 验证**：测试上市公司财报查询与结构化参数抽取。

+++

#### 步骤 5.1: 连通性与模型健康检查

请求 `GET /v1/models` 查看当前运行的模型信息：

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
Models response: {"object":"list","data":[{"id":"ling-3.0-flash-fin-fp4","object":"model","created":1791611671,"owned_by":"vllm","root":"/home/squall/models/Ling-3.0-flash-Fin-fp4","parent":null,"max_model_len":262144,"permission":[{"id":"modelperm-a33a0bcbef60f74b","object":"model_permission","created":1791611671,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```

+++

#### 步骤 5.2: 金融深度推理、Thinking 思考链与测速

使用 OpenAI Python SDK 调用接口，通过高端半导体制造企业 FCFF 折现估值用例验证模型的专业金融逻辑，实时提取 `<think>` 思考链内容，并基于服务端统计的精确 Token 总数测算首字延迟 (TTFT) 与解码速率 (Decode TPS)：

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-ling-cookbook-test"
)

def verify_financial_reasoning_and_speed():
    prompt = (
        "请对一家高端半导体制造企业进行自由现金流（FCFF）折现估值简要测算：\n"
        "1. 第1年EBIT 120亿元，年复合增长率15%，所得税率15%；\n"
        "2. WACC为9.5%，永续增长率为2.5%。\n"
        "请展示推导逻辑与主要估值结论。"
    )
    print("Sending financial prompt to model 'ling-3.0-flash-fin-fp4'...")

    start_time = time.time()
    first_token_time = None
    chunk_count = 0
    exact_completion_tokens = None
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model="ling-3.0-flash-fin-fp4",
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
Sending financial prompt to model 'ling-3.0-flash-fin-fp4'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 384.62 ms
Decode TPS (Tokens/s): 34.43 t/s
Total Generated Tokens: 2048 (Exact tokens from usage)
Total Duration: 59.76 s

=== Extracted Reasoning Chain (<think>) ===
1. 整理已知条件与财务要素：
   - 第1年 EBIT = 120 亿元，年复合增长率 = 15%，所得税率 = 15%
   - 加权平均资本成本 (WACC) = 9.5%，永续增长率 (g) = 2.5%

2. 确定估值模型框架：
   - 财务逻辑校验：当前高增长率 15% > WACC 9.5%。若假设 15% 为永续增长，则分母 (WACC - g) 为负，折现估值发散至无穷大。因此采用两阶段 FCFF 估值模型（有限高增长预测期 + 永续增长终值期）。
   - 参考高端半导体制造企业资本开支与研发周期，设定前 5 年为显式高增长预测期，第 6 年起进入 2.5% 永续稳态增长阶段。

3. 计算各期现金流 (NOPAT 与 FCFF 估算)：
   - 税后净营业利润公式：NOPAT = EBIT × (1 - t) = EBIT × (1 - 15%) = EBIT × 0.85
   - 第 1 年 NOPAT = 120 × 0.85 = 102.00 亿元
   - 第 2 年 NOPAT = 102.00 × (1 + 15%) = 117.30 亿元
   - 第 3 年 NOPAT = 117.30 × 1.15 = 134.90 亿元
   - 第 4 年 NOPAT = 134.90 × 1.15 = 155.13 亿元
   - 第 5 年 NOPAT = 155.13 × 1.15 = 178.40 亿元

4. 折现测算 (WACC = 9.5%)：
   - PV₁ = 102.00 / (1.095)¹ = 93.15 亿元
   - PV₂ = 117.30 / (1.095)² = 97.92 亿元
   - PV₃ = 134.90 / (1.095)³ = 102.93 亿元
   - PV₄ = 155.13 / (1.095)⁴ = 108.19 亿元
   - PV₅ = 178.40 / (1.095)⁵ = 113.72 亿元
   - 预测期现值合计 (PV_explicit) = 93.15 + 97.92 + 102.93 + 108.19 + 113.72 = 515.92 亿元

=== Final Response Content ===
针对该高端半导体制造企业，采用两阶段企业自由现金流（FCFF）折现估值法的测算结果如下：

### 一、估值模型与基本假设
1. **模型选择**：企业当前处于 15% 高增长阶段，因高增长率高于资本成本（15% > 9.5%），若直接永续折现将导致模型发散，故采用两阶段 FCFF 折现模型。
2. **预测期假设**：设定未来 5 年（T=1~5）为高速成长期，复合增长率为 15%；自第 6 年起进入永续成熟期，永续增长率为 2.5%。
3. **资本成本与税率**：WACC 设定为 9.5%，企业所得税率 15%。

### 二、测算明细
- **预测期现金流（NOPAT）现值**：
  第 1 至第 5 年各期折现现值分别为 93.15 亿、97.92 亿、102.93 亿、108.19 亿与 113.72 亿元，**5 年预测期现金流现值总和为 515.92 亿元**。
- **终值（Terminal Value, TV）**：
  第 6 年 FCFF = 178.40 × (1 + 2.5%) = 182.86 亿元；
  第 5 年末终值 $TV_5 = \frac{182.86}{9.5\% - 2.5\%} = 2612.29$ 亿元；
  终值折现至当期现值 $PV(TV) = \frac{2612.29}{(1.095)^5} = 1659.35$ 亿元。

### 三、主要估值结论
企业价值 (Enterprise Value, EV) = 预测期现值 + 终值现值 = $515.92 + 1659.35 = \mathbf{2175.27 \text{ 亿元}}$。
```

+++

#### 步骤 5.3: 专业级金融 Tool Calling (财报查询与结构化参数解析)

Ling-3.0-flash-Fin 针对金融智能体（Agent）工作流与外部数据交互进行了专项对齐。通过在服务端开启 `--tool-call-parser ling3` 与 `--enable-auto-tool-choice`，vLLM 会将模型输出的原生 XML 结构自动解析为标准 OpenAI `tool_calls` JSON 数组：

```{code-cell}
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-ling-cookbook-test"
)

tools_schema = [
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
                        "description": "公司股票代码或标准名称，如 600519.SH 或 贵州茅台"
                    },
                    "report_period": {
                        "type": "string",
                        "description": "财务报告期，如 2025Q3, 2025FY"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "需要查询的财务科目名称列表，如 ['营业收入', '归母净利润', '经营活动产生的现金流量净额']"
                    }
                },
                "required": ["ticker", "report_period", "metrics"]
            }
        }
    }
]

def verify_tool_calling():
    print(f"Testing Financial Tool Calling with model 'ling-3.0-flash-fin-fp4'...")
    try:
        response = client.chat.completions.create(
            model="ling-3.0-flash-fin-fp4",
            messages=[
                {"role": "user", "content": "请帮我查一下贵州茅台在 2025 年第三季度的营业收入和归母净利润。"}
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
            print("\n=== Direct Response ===")
            print(message.content)

    except Exception as e:
        print(f"Tool calling verification failed: {e}")

if __name__ == "__main__":
    verify_tool_calling()
```

典型测试结果：
```text
Testing Financial Tool Calling with model 'ling-3.0-flash-fin-fp4'...

=== Function Call Output Detected ===
Tool Call ID: chatcmpl-tool-90787ece219bd20b
Function Name: query_financial_statement
Arguments JSON: {"ticker": "600519.SH", "report_period": "2025Q3", "metrics": ["营业收入", "归母净利润"]}
```

> **专业对齐解析**：模型在理解自然语言提问「贵州茅台」时，不仅准确触发了 `query_financial_statement` 工具并提取对应财务科目，更展现了金融专业大模型的实体消歧与标准化能力——自动将「贵州茅台」映射转换为 A 股官方股票代码 `600519.SH`，完美契合金融数据库的底层查询规范。

+++

### 步骤 6: 常见问题与故障排查

1. **Nightly cu130 Wheel 下载与网络问题**：
   - 现象：`uv pip install` 提示找不到匹配的 ARM64 cu130 安装包。
   - 解决：确保指定 `--extra-index-url https://wheels.vllm.ai/nightly/cu130` 与 `--torch-backend=cu130`。如果需要离线环境，可提前缓存 wheel 包。

2. **显存预分配比例建议 (`--gpu-memory-utilization`)**：
   - 现象：启动服务时提示 CUDA Out of Memory 或 KV Cache 分配空间不足。
   - 解决：DGX Spark 可用统一内存约为 121 GB。Ling-3.0-flash-Fin FP4 权重加载占用 64.15 GiB，推荐配置 `--gpu-memory-utilization 0.85`，可稳定分配 34.69 GiB KV Cache，容纳超过 420 万 tokens 缓存，支持 256K 满上下文的超大并发。

3. **端口冲突 (Port 30000 occupied)**：
   - 现象：服务端启动时报错 `Address already in use`。
   - 解决：通过 `lsof -i :30000` 查询占用进程（如先前启动的 SGLang 服务）并平稳退出，或在启动参数中通过 `--port` 更换服务端口。

4. **投机采样 MTP 支持核验**：
   - 现象：指定 `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` 报错。
   - 解决：Ling-3.0-flash-Fin-fp4 模型结构中未封装独立 MTP 投机头（`mtp_loss_scaling_factor: 0`），启动时无需配置 `--speculative-config`。在单卡单并发自回归模式下，Blackwell 原生解码吞吐即可达 ~35 tokens/s。

5. **Python 3.12 运行环境要求 (PEP 701 语法兼容)**：
   - 现象：启动 vLLM 时报 `SyntaxError: unterminated string literal`，指向 `sharded_state_loader.py`。
   - 解决：vLLM 0.31+ nightly 源码引入了 PEP 701 规范的多行嵌套 f-string 语法，在 Python 3.11 及以下版本解析会报错。请确保部署环境使用 Python 3.12（如通过 `uv venv --python 3.12` 创建环境）。
