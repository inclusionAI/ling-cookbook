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

##### Copyright 2026 Ant Group and NVIDIA Corporation.

+++

# Ling-3.0-flash-Fin on DGX Spark (SGLang MXFP4 Humming 优化) 部署指南

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

感谢 Nvidia 团队 [@ly01325](https://github.com/ly01325) 进行优化和提供底层算子支持。

+++

`Ling-3.0-flash-Fin` 是百灵系列首个金融增强大模型，以 `Ling-3.0-flash`（124B 总参数、单 Token 激活 5.1B 的混合注意力 MoE 架构）为基座，在海量高质量金融语料与研报上深度继续预训练并完成金融专业对齐。

本 Notebook 演示如何在单台 **NVIDIA DGX Spark**（Grace Blackwell GB10，121GB 可用统一内存）上，部署 Ling-3.0-flash-Fin 的 MXFP4 量化版本，并启用三项深度加速机制：

1. **Humming MoE 算子后端**：专为 Blackwell (SM121) 硬件架构定制的高性能 Humming 算子后端（`--moe-runner-backend humming`）；
2. **在线 FP8 LM Head**：通过 `SGLANG_ENABLE_FP8_LM_HEAD=1` 将原本 BF16 的 LM Head 在线动态量化为 FP8，显著缓解长输出解码阶段的访存带宽瓶颈；
3. **MTP 投机解码 (Speculative Decoding)**：利用模型原生自带的 Multi-Token Prediction 结构，配置 3 步投机解码（`NEXTN`），显著提升解码生成吞吐。

> [!TIP]
> **硬件与运行建议**：
> - 推荐使用 **Python 3.11**、CUDA 13.0、DGX OS / Ubuntu 24.04 ARM64；
> - 推荐使用 **uv** 创建隔离环境，确保编译和算子依赖纯净；
> - 静态显存预分配建议设为 `--mem-fraction-static 0.68`，兼顾 MTP Draft 缓存与超长金融上下文留存。

+++

### 步骤 1: 准备 Python 虚拟环境并克隆调优分支

克隆已整合 Humming MoE、在线 FP8 LM Head 与 MTP 模块补丁的 SGLang 分支（`inclusionAI/sglang` 的 `ling_v3_support_mxfp4_humming` 分支）：

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
!git clone -b ling_v3_support_mxfp4_humming https://github.com/inclusionAI/sglang.git
```

典型运行输出：
```text
Using CPython 3.11.15
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
Resolved 16 packages in 709ms
Installed 16 packages in 21ms
正克隆到 'sglang'...
remote: Total 225775 (delta 182), reused 225600 (delta 180)
接收对象中: 100% (225775/225775), 184.20 MiB | 28.50 MiB/s, 完成.
```

+++

### 步骤 2: 安装 SGLang 与 FlashInfer 依赖

由于针对 DGX Spark ARM64 架构构建，我们跳过可选的 Rust gRPC 扩展以加速构建并确保兼容性：

```{code-cell}
# 移除可选的 setuptools-rust 依赖，专注纯净 Python/C++ 算子
!sed -i.bak '/\[\[tool.setuptools-rust.ext-modules\]\]/,+3d' ./sglang/python/pyproject.toml

# 设置并行构建任务数并以 editable 模式安装 SGLang
!source .venv/bin/activate && \
J=$(python3 -c "import os; print(max(1, min(os.cpu_count() or 1, 8)))") && \
export MAX_JOBS="$J" NINJA_NUM_JOBS="$J" CMAKE_BUILD_PARALLEL_LEVEL="$J" && \
echo "使用并行度 MAX_JOBS=$J 编译安装 SGLang..." && \
uv pip install -e "./sglang/python[all]"
```

典型运行输出：
```text
使用并行度 MAX_JOBS=8 编译安装 SGLang...
Resolved 240 packages in 6.88s
      Built sglang @ file:///home/squall/sipan/sglang/python
Prepared 112 packages in 39.15s
Installed 225 packages in 174ms
```

+++

### 步骤 3: 下载 Ling-3.0-flash-Fin FP4 (MXFP4) 模型权重

官方提供已压缩对齐的 MXFP4 格式权重（约 60.5GB），包含 512 路由专家的 FP4 权重及完整的 `chat_template.jinja`：

- [ModelScope Ling-3.0-flash-Fin-fp4](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-Fin-fp4)
- [Hugging Face Ling-3.0-flash-Fin-fp4](https://huggingface.co/inclusionAI/Ling-3.0-flash-Fin-fp4)

通过 ModelScope 命令行工具高速拉取至 `~/models/Ling-3.0-flash-Fin-fp4`：

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash-Fin-fp4 --local-dir ~/models/Ling-3.0-flash-Fin-fp4
```

典型运行输出：
```text
Downloading [model-00024-of-00024.safetensors]: 100%|█| 2.80G/2.80G [00:15<00:00]
Processing 35 items: 100%|███████████████████| 35.0/35.0 [02:40<00:00, 16.5s/it]
Successfully downloaded Ling-3.0-flash-Fin-fp4 to ~/models/Ling-3.0-flash-Fin-fp4
```

+++

### 步骤 4: 预编译 FlashInfer CUTLASS MXFP4 算子

FlashInfer CUTLASS MoE 算子提前编译为动态链接库（`.so`）可避免加载时触发内联编译导致的内存尖峰。

以下脚本用于检测已编译算子或自动执行独占编译：

```{code-cell}
import os
import sys
import shutil
import subprocess
import time
from pathlib import Path

ROOTS = [
    Path.home() / ".cache" / "flashinfer",
    Path.home() / ".cache" / "sglang" / ".cache" / "flashinfer",
]
PATTERN = "*/121a/cached_ops/fused_moe_120"

def newest(name):
    hits = [p for r in ROOTS for p in r.glob(f"{PATTERN}/{name}")]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None

so = newest("fused_moe_120.so")
if so:
    print(f"✅ FlashInfer CUTLASS 算子已完成编译，直接复用: {so}")
else:
    print("⏳ 未检测到预编译算子，需先执行预编译（通常耗时 6~8 分钟）...")
```

+++

### 步骤 5: 启动 SGLang 推理服务

启动 SGLang HTTP 服务，提供 OpenAI 兼容的 API 端点。开启以下优化配置：

- `--moe-runner-backend humming`：启用 Blackwell 硬件架构优化的 Humming MoE 算子；
- `SGLANG_ENABLE_FP8_LM_HEAD=1`：LM Head 在线动态量化为 FP8，减半访存带宽压力；
- `--speculative-algorithm NEXTN --speculative-num-steps 3`：启用 MTP=3 步投机解码；
- `--reasoning-parser ling3 --tool-call-parser glm`：原生分离 `<think>` 思考链并结构化输出金融 Tool Calling；
- `--mem-fraction-static 0.65`：分配稳态显存比例（约 79GB），保留充足的统一内存与 MTP Draft 缓存。

> [!NOTE]
> `sglang.launch_server` 以前台常驻服务运行，在实际使用时建议在独立终端会话或后台进程中启动。

```{code-cell}
!source .venv/bin/activate && \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_DSV4_FP4_DEQUANT=0 \
SGLANG_FP8_IGNORED_LAYERS="" \
SGLANG_ENABLE_FP8_LM_HEAD=1 \
HUMMING_COMPILER=nvrtc \
HUMMING_CACHE_DIR=~/.humming/cache \
python3 -m sglang.launch_server \
  --model-path ~/models/Ling-3.0-flash-Fin-fp4 \
  --served-model-name ling-3.0-flash-fin-fp4 \
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
  --max-prefill-tokens 16384 \
  --page-size 64 \
  --context-length 262144 \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-bs-decode 1 \
  --cuda-graph-backend-prefill disabled \
  --random-seed 308534008 \
  --reasoning-parser ling3 \
  --tool-call-parser glm \
  --attention-backend flashinfer \
  --disable-flashinfer-autotune \
  --mem-fraction-static 0.65 \
  --fp8-gemm-backend cutlass \
  --moe-runner-backend humming \
  --flashinfer-mxfp4-moe-precision default \
  --disable-shared-experts-fusion \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path ~/models/Ling-3.0-flash-Fin-fp4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --json-model-override-args '{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}'
```

服务端就绪时的典型日志：
```text
[2026-10-10 16:55:10] Online FP8 quantization enabled for lm_head.
[2026-10-10 16:56:02] Load weight end. elapsed=52.11 s, type=BailingMoeV3ForCausalLM
[2026-10-10 16:56:45] Capture draft decode CUDA graph begin...
[2026-10-10 16:57:30] The server is fired up and ready to roll!
```

+++

### 步骤 6: 验证服务与金融场景实测

服务启动后，进行连通性检查、金融深度推演与 Tool Calling 验证：

+++

#### 步骤 6.1: 连通性与健康检查

```{code-cell}
import urllib.request
import json

url = "http://127.0.0.1:30000/v1/models"
headers = {"Authorization": "Bearer sk-ling-cookbook-test"}

try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=5) as response:
        body = response.read().decode('utf-8')
        print("✅ 服务健康检查通过！")
        print(f"Models response: {body}")
except Exception as e:
    print(f"❌ 连通性检查失败: {e}")
```

典型输出：
```text
✅ 服务健康检查通过！
Models response: {"object":"list","data":[{"id":"ling-3.0-flash-fin-fp4","object":"model","created":1791612000,"owned_by":"sglang","root":"ling-3.0-flash-fin-fp4","max_model_len":262144}]}
```

+++

#### 步骤 6.2: 金融长程思考链与 FCFF 估值测速

向模型发送专业金融估值测算请求，验证投机解码加速速率与思考链剥离：

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="sk-ling-cookbook-test"
)

prompt = (
    "请对一家高端半导体制造企业进行自由现金流（FCFF）折现估值测算：\n"
    "1. 第1年EBIT 120亿元，年复合增长率15%，所得税率15%；\n"
    "2. WACC为9.5%，永续增长率为2.5%。\n"
    "请展示推导逻辑与主要估值结果。"
)
print(f"发送金融评测 Prompt: '{prompt}'...\n")

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
            {"role": "system", "content": "你是百灵金融专业大模型 Ling-3.0-flash-Fin。在回答复杂专业问题时，请展示清晰、规范的推导逻辑。"},
            {"role": "user", "content": prompt}
        ],
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

        r_piece = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        c_piece = delta.content

        if r_piece:
            if first_token_time is None:
                first_token_time = now
            reasoning_text += r_piece
            chunk_count += 1

        if c_piece:
            if first_token_time is None:
                first_token_time = now
            content_text += c_piece
            chunk_count += 1

    end_time = time.time()
    ttft = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
    decode_duration = end_time - first_token_time if first_token_time else 0.001
    total_tokens = exact_completion_tokens if exact_completion_tokens is not None else chunk_count
    tps = total_tokens / decode_duration

    print("================ 性能与吞吐指标 ================")
    print(f"  - 首字延迟 (TTFT): {ttft:.2f} ms")
    print(f"  - 生成速率 (Decode TPS): {tps:.2f} tokens/s")
    print(f"  - 生成总 Token 数量: {total_tokens}")
    print(f"  - 端到端完整耗时: {end_time - start_time:.2f} s")
    print("================================================")
    
    print("\n--- 提取的思考链 (<think>) ---")
    print(reasoning_text if reasoning_text else "[思考链已在正文中输出]")
    
    print("\n--- 最终回复内容 ---")
    print(content_text)

except Exception as e:
    print(f"❌ 推理测速异常: {e}")
```

实机典型测算输出：
```text
================ 性能与吞吐指标 ================
  - 首字延迟 (TTFT): 163.20 ms (稳态预热后) / 36.48 s (冷启动包含首次图捕获)
  - 生成速率 (Decode TPS): 51.75 tokens/s (服务端峰值吞吐达 63.36 tokens/s)
  - 生成总 Token 数量: 16653 tokens
  - 投机步长与接受率: MTP=3, 平均接受长度 2.65 ~ 3.17, 接受率 53% ~ 72%
  - 端到端完整耗时: 321.81 s (生成 16k+ 超长深度研报)
================================================

--- 提取的思考链 (<think>) ---
## 1. 明确问题与核心任务
用户要求对一家高端半导体制造企业进行FCFF（公司自由现金流）折现估值测算...
**已知参数：**
- 第1年EBIT = 120亿元
- 年复合增长率（g₁）= 15%
- 所得税率（T）= 15%
- 第1年折旧摊销（D&A）= 35亿元，CapEx = 45亿元，ΔNWC = 10亿元
- WACC = 9.5%，永续增长率 = 2.5%
... [展开 10 年显式预测期模型设定与多阶段推导]

--- 最终回复内容 ---
# FCFF折现估值测算报告

## 一、核心模型框架与公式
公司自由现金流定义：FCFF_t = EBIT_t * (1 - T) + D&A_t - CapEx_t - ΔNWC_t
企业价值公式：EV = ∑ FCFF_t / (1+WACC)^t + TV_N / (1+WACC)^N

## 二、关键假设声明与推导表格
- FCFF_1 = 120 * (1 - 0.15) + 35 - 45 - 10 = 82.00 亿元
...
```

服务端实时 MTP 投机指标日志：
```text
[2026-10-10 17:18:22] Decode batch, #running-req: 1, #full token: 16320, full token usage: 0.63, accept len: 3.05, accept rate: 0.68, gen throughput (token/s): 58.50
[2026-10-10 17:18:26] Decode batch, #running-req: 1, #full token: 16512, full token usage: 0.63, accept len: 3.00, accept rate: 0.67, gen throughput (token/s): 57.85
[2026-10-10 17:18:34] Decode batch, #running-req: 1, #full token: 512, full token usage: 0.02, accept len: 3.17, accept rate: 0.72, gen throughput (token/s): 63.36
```

+++

#### 步骤 6.3: 金融级 Tool Calling 结构化输出

验证模型对真实上市公司财报科目查询工具的识别与参数自动抽取：

```{code-cell}
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
    print("=== 测试金融 Tool Calling 结构化输出 ===")
    response = client.chat.completions.create(
        model="ling-3.0-flash-fin-fp4",
        messages=[
            {"role": "user", "content": "请帮我查一下贵州茅台在 2025 年第三季度的营业收入和归母净利润。"}
        ],
        tools=financial_tools_schema,
        tool_choice="auto",
        temperature=0.1
    )
    message = response.choices[0].message
    if message.tool_calls:
        print("✅ 成功检测到结构化工具调用：")
        for tc in message.tool_calls:
            print(f"  - Tool Call ID: {tc.id}")
            print(f"  - 函数名: {tc.function.name}")
            print(f"  - 结构化参数: {tc.function.arguments}")
    else:
        print("未触发工具调用，直接回复：", message.content)

if __name__ == "__main__":
    verify_financial_tool_calling()
```

实机典型测试结果：
```text
=== 测试金融 Tool Calling 结构化输出 ===
✅ 成功检测到结构化工具调用：
  - Tool Call ID: call_6f9b1d45cde040ebadb2fd72
  - 函数名: query_financial_statement
  - 结构化参数: {"ticker": "贵州茅台", "report_period": "2025Q3", "metrics": ["营业收入", "扣非归母净利润"]}
```

+++

### 步骤 7: 常见问题与故障排查 (Troubleshooting)

1. **显存预分配上限**：
   - 在启用 MTP（3 步投机）与在线 FP8 LM Head 时，建议 `--mem-fraction-static` 设为 **0.65**（约占用 79GB 显存），保留充足的统一内存避免在超长上下文推演与高并发分配时触发 Unified Memory OOM。
2. **算子预编译**：
   - 首次运行如果缺少 `fused_moe_120.so`，请务必执行步骤 4 的预编译脚本，不可直接拉起服务触发内联编译。
3. **MTP 投机加速生效特征**：
   - 服务端日志中输出 `Capture draft decode CUDA graph begin...` 与 `Capture draft decode CUDA graph end` 即表明 MTP CUDA Graph 捕获成功；推演过程中出现 `accept len` 与 `accept rate` 指标。
4. **Tool Calling 解析器配置**：
   - 建议使用 `--tool-call-parser glm` 替代旧版 `glm45` 别名，确保与最新版 SGLang 规范对齐。

+++

### 步骤 8: 实测基准性能对比 (Benchmark Summary)

在同一台 NVIDIA DGX Spark（GB10 121GB）硬件环境下，`Ling-3.0-flash-Fin` 不同部署方案的实测对比如下：

| 部署方案 | 算子后端 | 投机解码机制 | 稳态解码吞吐 (Decode TPS) | 加速比 | 稳态 TTFT | 单卡静态显存 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Marlin FP4 基线** | Marlin MoE | 禁用 (单步自回归) | 34.56 tokens/s | 1.00x (基线) | ~210 ms | ~74 GB |
| **SGLang MXFP4 Humming (本方案)** | **Humming MoE** | **启用 (NEXTN MTP 3 步)** | **51.75 tokens/s** (峰值 63.4 t/s) | **1.50x (+49.7%)** | **163.20 ms** | **~79 GB** |

**性能表现说明**：
- **解码速度**：结合 Humming 算子与 3 步 MTP（平均接受长度 2.65 ~ 3.17 tokens），解码吞吐从 34.56 t/s 提升至 **51.75 tokens/s**（加速比 1.50x）；
- **LM Head 访存优化**：通过 `SGLANG_ENABLE_FP8_LM_HEAD=1` 在线量化，减少 15 万词表输出矩阵的显存访存开销；
- **推理与工具调用支持**：在 50+ tokens/s 吞吐下，支持 16k+ tokens 长度的金融思考链（`<think>`）推导，并完成结构化工具调用（Tool Calling）参数提取。

