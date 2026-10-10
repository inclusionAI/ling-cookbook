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

# Ling-3.0-flash-Fin on DGX Spark (vLLM FP4) Deployment Guide

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

Ling-3.0-flash-Fin is the first finance-specialized large language model in the Ling series. Built on the Ling-3.0-flash base, it features 124B total parameters and 5.1B active parameters per token, natively supporting a 256K long-context window. The model is specifically optimized for complex, long-horizon tasks including financial research report analysis, financial statement reconciliation, multi-table data alignment, and DCF/valuation modeling.

This notebook demonstrates how to deploy the FP4 (MXFP4) quantized version of Ling-3.0-flash-Fin on `NVIDIA DGX Spark` (GB10 with 121GB unified memory) using the official vLLM prebuilt wheel for ARM64 CUDA 13.0. The quantized model weights require only ~60.5 GB. Powered by DeepGEMM MXFP4 hardware-accelerated kernels and Triton MLA attention, it delivers high inference throughput, low first-token latency, and ample memory headroom for long contexts on a single DGX Spark node.

> [!TIP]
> **Environment Recommendations**:
> - **Python 3.12** is strongly recommended.
> - Using **uv** to manage an isolated Python virtual environment is recommended to ensure clean dependency management and operator compatibility.

+++

### Step 1: Set Up Python 3.12 Virtual Environment (uv)

We recommend creating an isolated virtual environment with `uv` to ensure clean dependencies. Install the `openai` client for subsequent service endpoint testing:

```{code-cell}
!pip install -U uv
!uv venv --python 3.12 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
```

Typical output:
```text
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtualenv at: .venv
Activate with: source .venv/bin/activate
Resolved 20 packages in 120ms
Installed 20 packages in 45ms
 + openai==1.60.0
```

+++

### Step 2: Install Official vLLM ARM64 CUDA 13.0 Prebuilt Wheel

Ling-3.0 architecture support and the `ling3` parser have been merged upstream into official vLLM. Specify `--torch-backend=cu130` with `uv pip` to install the prebuilt wheel optimized for DGX Spark GB10 (`sm_121` / `cu130`):

```{code-cell}
!uv pip install \
  --upgrade vllm \
  --torch-backend=cu130 \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130
```

Typical output:
```text
⠇ Resolving dependencies...
Installed 180 packages in 114ms
 + aiohappyeyeballs==2.7.1
 + aiohttp==3.14.3
Successfully installed vllm-0.7.0
```

+++

### Step 3: Download Ling-3.0-flash-Fin FP4 Model Weights

Download the official `inclusionAI/Ling-3.0-flash-Fin-fp4` pre-quantized weights from ModelScope or Hugging Face into the local model directory `~/models/Ling-3.0-flash-Fin-fp4`:

- [Ling-3.0-flash-Fin-fp4 on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-Fin-fp4)
- [Ling-3.0-flash-Fin-fp4 on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-Fin-fp4)

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-Fin-fp4 --local-dir ~/models/Ling-3.0-flash-Fin-fp4
```

Typical output:
```text
Downloading snapshot of inclusionAI/Ling-3.0-flash-Fin-fp4 (model)...
Downloading [model-00065-of-00065.safetensors]: 100%|██████████| 1.02G/1.02G [00:12<00:00]
Processing 77 items: 100%|███████████████████| 77.0/77.0 [12:35<00:00, 9.8s/it]
Successfully downloaded Ling-3.0-flash-Fin-fp4 to ~/models/Ling-3.0-flash-Fin-fp4
```

+++

### Step 4: Launch vLLM HTTP Inference Server

Start the `vllm` service to load the model and launch an OpenAI-compatible HTTP API server.

Key configuration parameters:
- `--served-model-name ling-3.0-flash-fin-fp4`: External model registration identifier.
- `--port 30000`: Unified service port.
- `--gpu-memory-utilization 0.85`: Allocates up to ~108GB unified memory. With ~64.15GB weights, nearly 35GB remains for PagedAttention KV Cache (accommodating 4,288,818 cached tokens, supporting 16.36x concurrent full 256K contexts).
- `--trust-remote-code`: Enables loading Ling hybrid attention (KDA + MLA) architectural definitions.
- `--reasoning-parser ling3`: Automatically separates `<think>` reasoning chains into `delta.reasoning`.
- `--tool-call-parser ling3` & `--enable-auto-tool-choice`: Automatically converts native `<tool_call>` tags into standard OpenAI `tool_calls` structured objects.

⚠️ Note: `vllm` runs in foreground persistent mode by default. Running it directly within a Jupyter Notebook cell will block the kernel. Launch the command below in a separate persistent terminal session.

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

Typical startup log:
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

### Step 5: Verify Service and API Endpoints

Once the server is up and running, verify the deployment using standard OpenAI-compatible client calls:

1. **Health Check**: Confirm endpoint connectivity and model registration.
2. **Financial Deep Reasoning & Streaming Benchmarks**: Verify complex FCFF valuation reasoning, separate `<think>` reasoning chains, and benchmark TTFT and Decode TPS.
3. **Specialized Financial Tool Calling**: Test structured parameter extraction against corporate financial statements.

+++

#### Step 5.1: Endpoint Connectivity & Model Health Check

Query `GET /v1/models` to verify active model status:

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

Typical output:
```text
Health check status: 200
Models response: {"object":"list","data":[{"id":"ling-3.0-flash-fin-fp4","object":"model","created":1791611671,"owned_by":"vllm","root":"/home/squall/models/Ling-3.0-flash-Fin-fp4","parent":null,"max_model_len":262144,"permission":[{"id":"modelperm-a33a0bcbef60f74b","object":"model_permission","created":1791611671,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```

+++

#### Step 5.2: Financial Deep Reasoning, Thinking Chain, and Speed Verification

Use the OpenAI Python SDK to benchmark a complex financial valuation scenario (Free Cash Flow to Firm, FCFF) for a semiconductor manufacturing enterprise. Real-time `<think>` reasoning chunks are extracted, and TTFT alongside Decode TPS are measured:

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

Typical output:
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

#### Step 5.3: Specialized Financial Tool Calling (Financial Statement Query & Argument Parsing)

Ling-3.0-flash-Fin is specifically aligned for financial agent workflows and external data interaction. With `--tool-call-parser ling3` and `--enable-auto-tool-choice` enabled on the server, vLLM automatically parses the model's native XML tool call structure into standard OpenAI `tool_calls` JSON arrays:

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
            "description": "Query balance sheet, income statement, or cash flow metrics for listed companies",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock symbol or company name, e.g., 600519.SH or Kweichow Moutai"
                    },
                    "report_period": {
                        "type": "string",
                        "description": "Financial reporting period, e.g., 2025Q3, 2025FY"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Financial line item names, e.g., ['Operating Revenue', 'Net Profit']"
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

Typical output:
```text
Testing Financial Tool Calling with model 'ling-3.0-flash-fin-fp4'...

=== Function Call Output Detected ===
Tool Call ID: chatcmpl-tool-90787ece219bd20b
Function Name: query_financial_statement
Arguments JSON: {"ticker": "600519.SH", "report_period": "2025Q3", "metrics": ["营业收入", "归母净利润"]}
```

> **Financial Alignment Highlight**: When handling natural language queries mentioning company names like "贵州茅台" (Kweichow Moutai), the model not only triggers the `query_financial_statement` tool accurately, but also resolves and disambiguates company entities into standard A-share ticker symbols (`600519.SH`), perfectly adhering to institutional database conventions.

+++

### Step 6: Troubleshooting

1. **Nightly cu130 Wheel Download and Network Issues**:
   - Symptoms: `uv pip install` cannot find matching ARM64 cu130 wheels.
   - Solution: Ensure `--extra-index-url https://wheels.vllm.ai/nightly/cu130` and `--torch-backend=cu130` are specified. Pre-download wheel packages if deploying in air-gapped environments.

2. **Memory Utilization Guidelines (`--gpu-memory-utilization`)**:
   - Symptoms: Service startup reports CUDA Out of Memory or insufficient KV Cache allocation space.
   - Solution: DGX Spark provides ~121GB usable unified memory. Ling-3.0-flash-Fin FP4 weights occupy ~64.15 GiB. We recommend setting `--gpu-memory-utilization 0.85`, which safely allocates ~34.69 GiB for KV Cache (over 4.2 million tokens capacity).

3. **Port Conflict (Port 30000 occupied)**:
   - Symptoms: Startup fails with `Address already in use`.
   - Solution: Identify conflicting processes using `lsof -i :30000` (such as earlier SGLang instances) and terminate them gracefully, or pass `--port <new_port>` to specify an alternative port.

4. **Speculative Decoding MTP Verification**:
   - Symptoms: Passing `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` causes an initialization error.
   - Solution: Ling-3.0-flash-Fin-fp4 does not package standalone MTP speculative prediction heads (`mtp_loss_scaling_factor: 0`). Do not pass `--speculative-config`. In standard single-batch autoregressive generation, Blackwell native decoding achieves ~34-35 tokens/s.

5. **Python 3.12 Runtime Requirement (PEP 701 Syntax Compatibility)**:
   - Symptoms: vLLM fails to import with `SyntaxError: unterminated string literal` pointing to `sharded_state_loader.py`.
   - Solution: Recent vLLM nightly releases incorporate PEP 701 nested multi-line f-string syntax, which requires Python 3.12+. Ensure your environment is created with Python 3.12 (e.g., `uv venv --python 3.12`).
