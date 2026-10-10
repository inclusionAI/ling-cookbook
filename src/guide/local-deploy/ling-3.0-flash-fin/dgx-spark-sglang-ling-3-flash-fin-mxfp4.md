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

# Ling-3.0-flash-Fin on DGX Spark (SGLang MXFP4) Deployment Guide

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

This notebook demonstrates how to build `SGLang` from source to deploy the MXFP4 quantized version of Ling-3.0-flash-Fin on a single `NVIDIA DGX Spark` (GB10 Grace Blackwell with 121GB unified memory). The quantized weights require approximately ~60.5 GB, enabling high-throughput inference while preserving ample memory for ultra-long context windows and KV cache.

> [!TIP]
> **Environment Recommendations**:
> - We recommend using **Python 3.11 or 3.12**.
> - We recommend using **uv** to manage an isolated Python virtual environment, ensuring clean dependency management and kernel compatibility.

+++

### Step 1: Set Up Python Virtual Environment (uv) and Clone Repository

We recommend creating an isolated virtual environment with `uv` and installing the OpenAI Python SDK. Then clone the official Ling-3.0 support branch (`inclusionAI/sglang:ling_v3_support_mxfp4`):

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
!git clone -b ling_v3_support_mxfp4 https://github.com/inclusionAI/sglang.git
```

Typical output:
```text
Using CPython 3.11 interpreter at: /usr/bin/python3.11
Creating virtualenv at: .venv
Cloning into 'sglang'...
remote: Enumerating objects: 38200, done.
Switched to a new branch 'ling_v3_support_mxfp4'
```

+++

### Step 2: Install SGLang and Dependencies from Source

Install the branch in editable mode along with all required dependencies (`[all]`). Set `MAX_JOBS=4` to avoid running out of memory during multi-core C++/CUDA kernel compilation:

```{code-cell}
!source .venv/bin/activate && MAX_JOBS=4 uv pip install -e "./sglang/python[all]"
```

Typical output:
```text
Requirement already satisfied: pip in ...
Installing collected packages: sglang
  Running setup.py develop for sglang
Successfully installed sglang
```

+++

### Step 3: Download Model Weights

The pre-quantized FP4 (MXFP4) finance model weights are officially hosted at:
- [Ling-3.0-flash-Fin-fp4 on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-Fin-fp4)
- [Ling-3.0-flash-Fin-fp4 on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-Fin-fp4)

We recommend downloading the weights to `~/models/Ling-3.0-flash-Fin-fp4` using either ModelScope CLI or Hugging Face CLI:

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-Fin-fp4 --local-dir ~/models/Ling-3.0-flash-Fin-fp4
```

Typical output:
```text
Downloading [model-00065-of-00065.safetensors]: 100%|██████████| 1.02G/1.02G [00:12<00:00]
Processing 77 items: 100%|███████████████████| 77.0/77.0 [12:35<00:00, 9.8s/it]
Successfully downloaded Ling-3.0-flash-Fin-fp4 to ~/models/Ling-3.0-flash-Fin-fp4
```

+++

### Step 4: Launch HTTP Inference Service (Marlin MoE Backend)

Launch the SGLang Server to provide an OpenAI-compatible HTTP interface. For MXFP4 weights, deployment standardizes on the Marlin MoE kernel backend (`--moe-runner-backend marlin`), delivering high throughput and numerical stability.

Key parameter descriptions:
- `--model-path ~/models/Ling-3.0-flash-Fin-fp4`: Path to the local model weights
- `--served-model-name ling-v3-flash-fin-fp4`: Model name registered for client invocation
- `--moe-runner-backend marlin`: Enables the optimized Marlin hardware kernel for MXFP4 experts
- `--cuda-graph-backend-decode full --cuda-graph-max-bs-decode 1`: Captures a complete CUDA Graph during the decode phase for batch size 1 to minimize kernel launch overhead
- `--attention-backend flashinfer --fp8-gemm-backend cutlass`: Enables FlashInfer attention and CUTLASS GEMM acceleration
- `--mem-fraction-static 0.8`: Reserves 80% of GPU memory for static weights and base buffers, ensuring sufficient KV cache headroom
- `--tool-call-parser ling3 --reasoning-parser ling3`: Enables native Ling-3.0 tool calling and reasoning chain parsers
- `--json-model-override-args`: Configures YaRN RoPE Scaling to support the 256K context window

⚠️ Important Note:

`sglang.launch_server` runs as a persistent foreground process. Executing the cell directly inside a Notebook will block subsequent cells from running. We recommend launching the server in a separate terminal session. Once the server is ready, proceed with the verification cells below.

```{code-cell}
!source .venv/bin/activate && \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_DSV4_FP4_DEQUANT=0 \
SGLANG_FP8_IGNORED_LAYERS="" \
python3 -m sglang.launch_server \
  --model-path ~/models/Ling-3.0-flash-Fin-fp4 \
  --served-model-name ling-v3-flash-fin-fp4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 1 \
  --ep-size 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key sk-ling-cookbook-test \
  --max-running-requests 1 \
  --max-mamba-cache-size 64 \
  --chunked-prefill-size 8192 \
  --page-size 64 \
  --context-length 262144 \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-bs-decode 1 \
  --cuda-graph-backend-prefill disabled \
  --random-seed 308534008 \
  --reasoning-parser ling3 \
  --tool-call-parser ling3 \
  --attention-backend flashinfer \
  --disable-flashinfer-autotune \
  --mem-fraction-static 0.8 \
  --fp8-gemm-backend cutlass \
  --moe-runner-backend marlin \
  --disable-shared-experts-fusion \
  --enable-fp32-lm-head \
  --json-model-override-args '{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}'
```

Typical server startup logs:
```text
[2026-10-10 11:53:57] Detected mixed checkpoint layout: routed experts are MXFP4.
[2026-10-10 11:54:00] Auto-detected template features: reasoning_config=ReasoningToggleConfig(toggle_param='enable_thinking', default_enabled=True, special_case=None, effort_kwarg=None), reasoning_parser=qwen3, tool_call_parser=glm45
[2026-10-10 11:54:06] Load weight begin. avail mem=109.98 GB
[2026-10-10 11:54:06] Detected fp8 checkpoint.
[2026-10-10 11:54:06] attention type of layers:[0, 0, 0, 0, 0, 1, 0, ...], 0 is linear layer and 1 is softmax layer!
[2026-10-10 11:54:07] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=marlin, quant_method=Mxfp4MarlinMoEMethod).
Multi-thread loading shards: 100% Completed | 65/65 [05:15<00:00, 4.85s/it]
[2026-10-10 11:59:36] Load weight end. elapsed=329.53 s, type=BailingMoeV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=29.67 GB, mem usage=80.31 GB.
[2026-10-10 11:59:39] Mamba Cache is allocated. max_mamba_cache_size: 117, conv_state size: 0.28GB, ssm_state size: 8.07GB 
[2026-10-10 11:59:41] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 1226984, KV size: 9.21 GB
[2026-10-10 11:59:41] Linear attention kernel backend: decode=triton, prefill=triton
[2026-10-10 11:59:41] KDA kernel dispatcher: decode=TritonKDAKernel, extend=TritonKDAKernel packed_decode=True
[2026-10-10 11:59:51] Capture target decode CUDA graph end. elapsed=10.10 s, mem usage=0.79 GB, avail mem=10.39 GB.
[2026-10-10 11:59:51] max_total_num_tokens=1226984, chunked_prefill_size=8192, context_len=262144, available_gpu_mem=10.39 GB
[2026-10-10 11:59:52] INFO:     Started server process [980369]
[2026-10-10 11:59:52] INFO:     Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
[2026-10-10 12:00:13] The server is fired up and ready to roll!
```

+++

### Step 5: Verify Deployment and Test API Calls

Once the server is running, it can be accessed using standard LLM client libraries. Ling-3.0-flash-Fin defaults to thinking mode enabled, with recommended generation parameters: `temperature=1.0`, `top_p=0.95`, and `top_k=20`.

This section provides 3 verification steps:
1. **Connectivity & Health Check**: Confirm model registration and endpoint responsiveness;
2. **Financial Deep Reasoning & Streaming Benchmark**: Verify Free Cash Flow to Firm (FCFF) discounted valuation reasoning, extract the `<think>` chain, and measure TTFT and Decode TPS;
3. **Financial Tool Calling**: Test structured function call generation for financial statement queries.

+++

#### Step 5.1: Connectivity & Model Health Check

Query `GET /v1/models` to verify active models:

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
```json
{
  "object": "list",
  "data": [
    {
      "id": "ling-v3-flash-fin-fp4",
      "object": "model",
      "created": 1760068792,
      "owned_by": "sglang"
    }
  ]
}
```

+++

#### Step 5.2: Financial Deep Reasoning, Thinking Chain, and Speed Benchmark

Use the OpenAI Python SDK to test financial logic on a two-stage FCFF valuation for a semiconductor enterprise. Stream the output, extract the `<think>` reasoning chain, and compute TTFT and Decode TPS using server-reported token counts:

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
    print(f"Sending financial prompt to model 'ling-v3-flash-fin-fp4'...")
    
    start_time = time.time()
    first_token_time = None
    chunk_count = 0
    exact_completion_tokens = None
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model="ling-v3-flash-fin-fp4",
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
            now = time.time()
            if first_token_time is None and (delta.content or (hasattr(delta, 'reasoning_content') and delta.reasoning_content)):
                first_token_time = now

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_text += delta.reasoning_content
                chunk_count += 1
            elif delta.content:
                content_text += delta.content
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

Typical test results:
```text
Sending financial prompt to model 'ling-v3-flash-fin-fp4'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 127.94 ms
Decode TPS (Tokens/s): 34.56 t/s
Total Generated Tokens: 1024 (Exact tokens from usage)
Total Duration: 29.82 s

=== Extracted Reasoning Chain (<think>) ===
1. 整理已知条件与财务要素：
   - 第1年 EBIT = 120 亿元，年复合增长率 = 15%，所得税率 = 15%
   - 加权平均资本成本 (WACC) = 9.5%，永续增长率 (g) = 2.5%

2. 确定估值模型框架：
   - 核心财务校验：当前高增长率 15% > WACC 9.5%。若假设 15% 为永续增长，则分母 (WACC - g) 为负，折现估值发散至无穷大。因此必须采用两阶段 FCFF 估值模型（有限高增长预测期 + 永续增长终值期）。
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

#### Step 5.3: Tool Calling and Structured Outputs Test

Define a structured function call schema for querying financial statements (`query_financial_statement`) to verify parameter extraction and structured output capabilities:

```{code-cell}
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
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
                        "description": "待查询财务科目列表，如 ['营业收入', '净利润']"
                    }
                },
                "required": ["ticker", "report_period", "metrics"]
            }
        }
    }
]

def verify_financial_tool_calling():
    print(f"Testing Financial Tool Calling with model 'ling-v3-flash-fin-fp4'...")
    try:
        response = client.chat.completions.create(
            model="ling-v3-flash-fin-fp4",
            messages=[
                {"role": "user", "content": "帮我查询贵州茅台2025年第三季度的营业收入和净利润。"}
            ],
            tools=financial_tools_schema,
            tool_choice="auto",
            temperature=0.1
        )

        message = response.choices[0].message
        if message.tool_calls:
            print("\n=== Financial Tool Call Output Detected (Parsed via --tool-call-parser ling3) ===")
            for tool_call in message.tool_calls:
                print(f"Tool Call ID: {tool_call.id}")
                print(f"Function Name: {tool_call.function.name}")
                print(f"Arguments JSON: {tool_call.function.arguments}")
        else:
            print("\n=== Direct Content Output (Raw Tags) ===")
            print(message.content)

    except Exception as e:
        print(f"Tool calling verification failed: {e}")

if __name__ == "__main__":
    verify_financial_tool_calling()
```

Typical test results:
```text
Testing Financial Tool Calling with model 'ling-v3-flash-fin-fp4'...

=== Financial Tool Call Output Detected (Parsed via --tool-call-parser ling3) ===
Tool Call ID: call_0_query_financial_statement_20261010
Function Name: query_financial_statement
Arguments JSON: {"ticker": "贵州茅台", "report_period": "2025Q3", "metrics": ["营业收入", "净利润"]}
```

+++

### Step 6: Troubleshooting & FAQ

1. **Tool Call Parsing Configuration (`--tool-call-parser ling3`)**:
   - Symptom: When making client requests with the `tools` parameter, `<tool_call>` tags appear mixed in `message.content` instead of being parsed into `message.tool_calls`.
   - Resolution: Ensure `--tool-call-parser ling3` (or `glm45`) is explicitly passed in the SGLang startup command. This instructs the SGLang serving layer to intercept and convert native XML tags into standard OpenAI JSON tool call objects.

2. **Cold Start Latency (TTFT) and Triton JIT Compilation**:
   - Symptom: The initial inference request after server readiness may take 20–30 seconds.
   - Resolution: This is expected behavior. SGLang uses Triton-optimized kernels tailored for Ling-3.0's 42-layer hybrid architecture (alternating KDA linear attention and MLA softmax attention). The first request triggers JIT kernel compilation and CUDA Graph capture; subsequent requests achieve stable TTFT within ~120–170 ms.

3. **Compilation Memory Protection (`MAX_JOBS=4`)**:
   - Symptom: An Out of Memory (OOM) error occurs while building C++/CUDA extensions via `MAX_JOBS=4 uv pip install -e "./sglang/python[all]"`.
   - Resolution: Limit parallel compilation jobs by setting `MAX_JOBS=4` (or `MAX_JOBS=2`) to reduce concurrent compiler memory usage.

4. **Static Memory Fraction Planning (`--mem-fraction-static 0.8`)**:
   - Symptom: When deploying on NVIDIA DGX Spark (GB10 with 121GB unified memory), setting `--mem-fraction-static 0.75` causes an error: `Loaded weights leave no GPU memory for the KV cache`.
   - Explanation: Ling-3.0-flash-Fin FP4 weights and initial Marlin/Mamba states require approximately 85 GB, which exceeds 75% of the ~110 GB available GPU memory (82.5 GB). Set `--mem-fraction-static 0.8` (reserving 80%, approximately ~88 GB) to guarantee sufficient memory pool for KV Cache allocation.
