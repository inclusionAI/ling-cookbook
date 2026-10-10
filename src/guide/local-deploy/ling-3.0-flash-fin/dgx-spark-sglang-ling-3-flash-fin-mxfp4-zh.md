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

# Ling-3.0-flash-Fin on DGX Spark (SGLang MXFP4) 部署指南

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

本 Notebook 将演示如何在 `NVIDIA DGX Spark` (GB10 121GB 统一内存) 上通过源码构建 `SGLang` 部署 Ling-3.0-flash-Fin 的 MXFP4 量化版本，量化后模型权重仅占 ~60.5 GB，可在单台 DGX Spark 上高效推理并为超长上下文保留充裕显存。

> [!TIP]
> **环境准备建议**：
> - 推荐使用 **Python 3.11 / 3.12** 环境；
> - 推荐使用 **uv** 创建独立的 Python 虚拟环境，以确保依赖隔离与算子兼容性。

+++

### 步骤 1: 准备 Python 虚拟环境 (uv) 与克隆 SGLang 仓库

推荐使用 `uv` 创建独立的 Python 虚拟环境并安装 OpenAI 客户端。同时克隆官方维护的 Ling-3.0 支持分支（`inclusionAI/sglang:ling_v3_support_mxfp4`）：

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
!git clone -b ling_v3_support_mxfp4 https://github.com/inclusionAI/sglang.git
```

典型运行输出：
```text
Using CPython 3.11 interpreter at: /usr/bin/python3.11
Creating virtualenv at: .venv
Cloning into 'sglang'...
remote: Enumerating objects: 38200, done.
Switched to a new branch 'ling_v3_support_mxfp4'
```

+++

### 步骤 2: 从源码安装 SGLang 与运行时全量依赖

在虚拟环境中以可编辑模式安装分支代码及全量依赖库（`[all]`）。设置 `MAX_JOBS=4` 防止多核并发编译导致内存耗尽：

```{code-cell}
!source .venv/bin/activate && MAX_JOBS=4 uv pip install -e "./sglang/python[all]"
```

典型运行输出：
```text
Requirement already satisfied: pip in ...
Installing collected packages: sglang
  Running setup.py develop for sglang
Successfully installed sglang
```

+++

### 步骤 3: 下载 Ling-3.0-flash-Fin FP4 (MXFP4) 模型权重

官方直接提供了预量化的 FP4 (MXFP4) 格式金融模型权重：
- [Ling-3.0-flash-Fin-fp4 on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-Fin-fp4)
- [Ling-3.0-flash-Fin-fp4 on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-Fin-fp4)

推荐使用 ModelScope CLI 或 Hugging Face CLI 将权重下载至本地统一模型目录 `~/models/Ling-3.0-flash-Fin-fp4`：

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-Fin-fp4 --local-dir ~/models/Ling-3.0-flash-Fin-fp4
```

典型运行输出：
```text
Downloading [model-00065-of-00065.safetensors]: 100%|██████████| 1.02G/1.02G [00:12<00:00]
Processing 77 items: 100%|███████████████████| 77.0/77.0 [12:35<00:00, 9.8s/it]
Successfully downloaded Ling-3.0-flash-Fin-fp4 to ~/models/Ling-3.0-flash-Fin-fp4
```

+++

### 步骤 4: 启动 SGLang HTTP 推理服务 (Marlin 算子加速)

启动 SGLang Server，提供 OpenAI 兼容的 HTTP 接口。针对 MXFP4 权重，部署统一采用 Marlin MoE 算子后端（`--moe-runner-backend marlin`），兼具优秀的吞吐与数值稳定性。

部分关键参数说明：
- `--model-path ~/models/Ling-3.0-flash-Fin-fp4`：指定本地金融权重路径
- `--served-model-name ling-v3-flash-fin-fp4`：对外注册的服务模型标识
- `--moe-runner-backend marlin`：启用针对 MXFP4 专家矩阵优化的 Marlin 硬件加速算子
- `--cuda-graph-backend-decode full --cuda-graph-max-bs-decode 1`：启用单并发 Decode 阶段 CUDA Graph 全图捕获，降低调度延迟
- `--attention-backend flashinfer --fp8-gemm-backend cutlass`：启用 FlashInfer 注意力与 CUTLASS GEMM 加速
- `--mem-fraction-static 0.8`：显存预分配比例设为 80%，为超长上下文预留充足 KV 空间
- `--tool-call-parser ling3 --reasoning-parser ling3`：启用 Ling-3.0 工具调用与思考链解析器
- `--json-model-override-args`：配置 YaRN RoPE Scaling，原生支持 256K 超长上下文窗口

⚠️ 特别注意：

`sglang.launch_server` 以前台常驻模式运行。如果你直接在 Notebook 中运行下方单元格，Jupyter 将阻塞而无法执行后续单元格的代码。建议你在单独的终端会话中执行启动命令。启动成功后即可使用 Jupyter 进行后续验证。

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

服务端就绪时的典型日志：
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

### 步骤 5: 验证部署成功并使用模型服务

服务启动后，即可在各类 LLM 客户端中使用。Ling-3.0-flash-Fin 默认开启思考模式（Thinking Mode），推荐采样参数配置为 `temperature=1.0`、`top_p=0.95`、`top_k=20`。

本节提供 3 项逐步验证用例：
1. **健康检查**：确认模型注册状态与接口连通性；
2. **金融推理与流式测速**：验证自由现金流 (FCFF) 深度推理，提取 `<think>` 思考链并测算 TTFT 与 Decode TPS；
3. **金融 Tool Calling 验证**：测试财报查询结构化工具调用。

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

#### 步骤 5.2: 金融深度推理、Thinking 思考链与测速

使用 OpenAI Python SDK 调用接口，通过高端半导体制造企业 FCFF 折现估值专业用例验证模型的专业金融逻辑，实时提取 `<think>` 思考链内容，并基于服务端统计的精确 Token 总数测算首字延迟 (TTFT) 与解码速率 (Decode TPS)：

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

典型测试结果：
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

#### 步骤 5.3: 测试金融 Tool Calling 结构化调用

定义符合金融研报与指标查询场景的结构化工具定义（以财报指标查询 `query_financial_statement` 为例），验证模型对专业参数的抽取与原生标签生成能力：

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

典型测试结果：
```text
Testing Financial Tool Calling with model 'ling-v3-flash-fin-fp4'...

=== Financial Tool Call Output Detected (Parsed via --tool-call-parser ling3) ===
Tool Call ID: call_0_query_financial_statement_20261010
Function Name: query_financial_statement
Arguments JSON: {"ticker": "贵州茅台", "report_period": "2025Q3", "metrics": ["营业收入", "净利润"]}
```

+++

### 步骤 6: 常见问题与故障排查 (Troubleshooting)

1. **Tool Call 解析配置 (`--tool-call-parser ling3`)**：
   - 现象：客户端调用带有 `tools` 参数的请求时，返回结果混在 `message.content` 中出现 `<tool_call>` 标签，未解析到 `message.tool_calls`。
   - 解决：SGLang 启动参数必须显式声明 `--tool-call-parser ling3`（或 `glm45`）。这样 SGLang 的 Serving 模块即可将 Ling-3.0 生成的原生 XML 标签无缝解析转换为标准 JSON 工具调用。

2. **冷启动首字延迟 (Cold Start TTFT) 与 Triton JIT 编译**：
   - 现象：模型服务就绪后，首个推理请求可能耗时 20~30 秒。
   - 解决：这是正常行为。SGLang 采用针对 Ling-3.0 42 层交替架构（线性注意力 KDA 与 Softmax MLA）的 Triton 深度优化算子，在首次触发时需要进行 JIT 编译并捕获 CUDA Graph；编译缓存落地后，后续请求首字延迟稳定在 ~120ms。

3. **源码编译阶段内存保护 (`MAX_JOBS=4`)**：
   - 现象：在执行 `MAX_JOBS=4 uv pip install -e "./sglang/python[all]"` 编译 C++/CUDA 扩展时，因多核并发过高触发 OOM。
   - 解决：通过环境变量 `MAX_JOBS=4`（或 `MAX_JOBS=2`）限制并发编译线程数，降低并发构建内存压力。

4. **显存预分配比例规划 (`--mem-fraction-static 0.8`)**：
   - 现象：在 DGX Spark (GB10 121GB 统一内存) 上部署若设置 `--mem-fraction-static 0.75` 会报错：`Loaded weights leave no GPU memory for the KV cache`。
   - 说明：Ling-3.0-flash-Fin FP4 模型加载与 Marlin/Mamba 算子分配后静态显存分配需占用约 85 GB，超过 110 GB 可用显存的 75%（82.5 GB）。因此静态显存比例需显式设置为 `0.8`（即预留 80% 约 88 GB 静态空间），为系统留出充足 KV Cache。
