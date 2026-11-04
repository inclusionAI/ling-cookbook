## About this fork / 本 fork 说明

This fork of [`inclusionAI/ling-cookbook`](https://github.com/inclusionAI/ling-cookbook) hosts the deliverables for **[Issue #11 — Ling-3.0-tiny Fine-tuning Recipe (2026 Ant Open Source Camp · University Tour)](https://github.com/inclusionAI/ling-cookbook/issues/11)**, maintained by [@lostlight530](https://github.com/lostlight530) (Jiang, Xuanyi · [ORCID 0009-0001-3617-0832](https://orcid.org/0009-0001-3617-0832)).

本 fork 在上游基础上承载一项交付资产，其余内容与上游 `main` 保持同步：

| 资产 | 位置 | 说明 |
|---|---|---|
| **LoRA Adapter 物理权重** | [`adapter.tar.gz`](adapter.tar.gz)（本分支根目录） | Ling-3.0-tiny 端侧 Agent 工具调用 QLoRA 产物；`adapter_model.safetensors` 8,276,480 B；SHA-256 `0d17c57b149c935abad90fc76238fe8cce672d3b67894d5eccba293898d6a5fe` |
| **Recipe（PR 源）** | 分支 `edge-agent-tool-calling` → `guide/finetuning/ling-3.0-tiny/edge-agent-tool-calling/` | 已通过 PR 提交至上游；含说明文档、训练脚本、全量语料、评测证据与可重算脚本 |

### 分支约定 / Branch policy

- `main`：跟随上游 `inclusionAI:main` 同步，仅额外挂载 `adapter.tar.gz` 与本说明节——**不承载任何实验分支内容**
- `edge-agent-tool-calling`：交付分支，PR diff 仅含 recipe 目录纯文本（无二进制）

### 资产引用 / Citing the adapter

```bibtex
@misc{jiang2026lingedge,
  author       = {Jiang, Xuanyi},
  title        = {Ling-3.0-tiny Edge Agent Tool Calling QLoRA Adapter},
  year         = {2026},
  howpublished = {\url{https://github.com/lostlight530/ling-cookbook/blob/main/adapter.tar.gz}},
  note         = {LoRA r=8 $\alpha$=16, 4-bit NF4, seed-42 recipe; ORCID 0009-0001-3617-0832}
}
```

> 上游说明（未改动）：以下为 `inclusionAI/ling-cookbook` 原始 README 内容。

---

<p align="right">
  <b>English</b> | <a href="README-zh.md">简体中文</a>
</p>

# Ling Cookbook

> Ling Cookbook provides deployment recipes and practical engineering examples for the Ling model family. Work in progress.

---

## Resources & Community

- **Official Website**: [ant-ling.com](https://www.ant-ling.com/)
- **Hugging Face**: [inclusionAI](https://huggingface.co/inclusionAI)
- **ModelScope**: [inclusionAI](https://www.modelscope.cn/organization/inclusionAI)
- **GitHub**: [inclusionAI/ling-cookbook](https://github.com/inclusionAI/ling-cookbook)
- **Discord**: [Ling Community](https://discord.com/invite/GNaQc8WC5T)
- **X (Twitter)**: [@AntLingAGI](https://x.com/AntLingAGI)

---

## Table of Contents

### Local Deployment (`guide/local-deploy/`)

Recipes for deploying Ling models for local inference across consumer and workstation hardware.

#### Ling-3.0-flash

<p align="right">
  <b>English</b> | <a href="README-zh.md#ling-30-flash">简体中文</a>
</p>

| Model | Device | Framework | Quant | Info |
| :--- | :--- | :--- | :--- | :--- |
| Ling-3.0-flash | Single NVIDIA DGX Spark | SGLang | MXFP4 (Speed Optimized) | [Recipe: SGLang MXFP4 Humming Deployment Guide](guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-mxfp4-humming.ipynb) |
| Ling-3.0-flash | Single NVIDIA DGX Spark | SGLang | MXFP4 | [Recipe: SGLang MXFP4 Deployment Guide](guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-mxfp4.ipynb) |
| Ling-3.0-flash | Single NVIDIA DGX Spark | SGLang | INT4 | [Recipe: SGLang INT4 Deployment Guide](guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-int4.ipynb) |
| Ling-3.0-flash | Single NVIDIA DGX Spark | vLLM | FP4 | [Recipe: vLLM FP4 Deployment Guide](guide/local-deploy/ling-3.0-flash/dgx-spark-vllm-ling-3-flash-fp4.ipynb) |
| Ling-3.0-flash | Single NVIDIA DGX Spark | vLLM | INT4 | [Recipe: vLLM INT4 Deployment Guide](guide/local-deploy/ling-3.0-flash/dgx-spark-vllm-ling-3-flash-int4.ipynb) |
| Ling-3.0-flash | Single NVIDIA DGX Spark | llama.cpp | GGUF (Q6_K / Q4_K) | [Recipe: llama.cpp GGUF Deployment Guide](guide/local-deploy/ling-3.0-flash/dgx-spark-llamacpp-ling-3-flash-q4-gguf.ipynb) |

#### Ling-3.0-flash-VL

<p align="right">
  <b>English</b> | <a href="README-zh.md#ling-30-flash-vl">简体中文</a>
</p>

| Model | Device | Framework | Quant | Info |
| :--- | :--- | :--- | :--- | :--- |
| Ling-3.0-flash-VL | Single NVIDIA DGX Spark | SGLang | INT4 | [Recipe: SGLang INT4 Deployment Guide](guide/local-deploy/ling-3.0-flash-vl/dgx-spark-sglang-ling-3-flash-vl-int4.ipynb) |
| Ling-3.0-flash-VL | Single NVIDIA DGX Spark | vLLM | INT4 | [Recipe: vLLM INT4 Deployment Guide](guide/local-deploy/ling-3.0-flash-vl/dgx-spark-vllm-ling-3-flash-vl-int4.ipynb) |
| Ling-3.0-flash-VL | Single NVIDIA DGX Spark | llama.cpp | GGUF (Q4_K_M) | [Recipe: llama.cpp GGUF Deployment Guide](guide/local-deploy/ling-3.0-flash-vl/dgx-spark-llamacpp-ling-3-flash-vl-q4-gguf.ipynb) |

#### Ling-3.0-flash-Fin

<p align="right">
  <b>English</b> | <a href="README-zh.md#ling-30-flash-fin">简体中文</a>
</p>

| Model | Device | Framework | Quant | Info |
| :--- | :--- | :--- | :--- | :--- |
| Ling-3.0-flash-Fin | Single NVIDIA DGX Spark | SGLang | MXFP4 (Speed Optimized) | [Recipe: SGLang MXFP4 Humming Optimization Deployment Guide](guide/local-deploy/ling-3.0-flash-fin/dgx-spark-sglang-ling-3-flash-fin-mxfp4-humming.ipynb) |
| Ling-3.0-flash-Fin | Single NVIDIA DGX Spark | SGLang | MXFP4 | [Recipe: SGLang MXFP4 Deployment Guide](guide/local-deploy/ling-3.0-flash-fin/dgx-spark-sglang-ling-3-flash-fin-mxfp4.ipynb) |
| Ling-3.0-flash-Fin | Single NVIDIA DGX Spark | vLLM | FP4 | [Recipe: vLLM FP4 Deployment Guide](guide/local-deploy/ling-3.0-flash-fin/dgx-spark-vllm-ling-3-flash-fin-fp4.ipynb) |
| Ling-3.0-flash-Fin | Single NVIDIA DGX Spark | llama.cpp | GGUF (Q4_K_M) | [Recipe: llama.cpp GGUF Deployment Guide](guide/local-deploy/ling-3.0-flash-fin/dgx-spark-llamacpp-ling-3-flash-fin-q4-gguf.ipynb) |

#### Ling-3.0-tiny

<p align="right">
  <b>English</b> | <a href="README-zh.md#ling-30-tiny">简体中文</a>
</p>

| Model | Device | Framework | Quant | Info |
| :--- | :--- | :--- | :--- | :--- |
| Ling-3.0-tiny | Single NVIDIA DGX Spark | llama.cpp | BF16 (GGUF) | [Recipe: llama.cpp BF16 Deployment Guide](guide/local-deploy/ling-3.0-tiny/dgx-spark-llamacpp-ling-3-tiny-bf16-gguf.ipynb) |
| Ling-3.0-tiny | Single NVIDIA DGX Spark | SGLang | BF16 | [Recipe: SGLang BF16 Deployment Guide](guide/local-deploy/ling-3.0-tiny/dgx-spark-sglang-ling-3-tiny-bf16.ipynb) |
| Ling-3.0-tiny | Single NVIDIA DGX Spark | vLLM | BF16 | [Recipe: vLLM BF16 Deployment Guide](guide/local-deploy/ling-3.0-tiny/dgx-spark-vllm-ling-3-tiny-bf16.ipynb) |
| Ling-3.0-tiny | Apple Silicon Mac | Ollama | INT4 / FP8 / BF16 | [Recipe: Ollama On-Device Deployment Guide](guide/local-deploy/ling-3.0-tiny/mac-ollama-ling-3-tiny.ipynb) |
| Ling-3.0-tiny | Apple Silicon Mac | llama.cpp | BF16 / Q8_0 / Q4_K_M (GGUF) | [Recipe: llama.cpp Metal On-Device Deployment Guide](guide/local-deploy/ling-3.0-tiny/mac-llamacpp-ling-3-tiny.ipynb) |
| Ling-3.0-tiny | Apple Silicon Mac | MLX | BF16 / MXFP8 / 4-bit | [Recipe: MLX-LM On-Device Deployment Guide](guide/local-deploy/ling-3.0-tiny/mac-mlx-ling-3-tiny.ipynb) |

<a id="recommended-skills"></a>
### Agent Skills for Specialized Capabilities

Some of our models provide distinctive multimodal and agentic capabilities. To help users explore these features, we provide pre-built Agent Skills. You can download and use them locally, or clone the repository to let your agent inspect the skills, audit their safety, and assist you in experiencing the model capabilities.

These agent skills and their associated models include:

<p align="right">
  <b>English</b> | <a href="README-zh.md#recommended-skills">简体中文</a>
</p>

| Name | Demo Model | Description | Link |
| :--- | :--- | :--- | :--- |
| `image-to-editable-ppt` | Ming-Image-0.1-Design-Layer [Hugging Face](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer) [ModelScope](https://www.modelscope.cn/models/inclusionAI/Ming-Image-0.1-Design-Layer) | Reconstructs a slide image/screenshot into an editable native `.pptx` | [README](resources/recommended-skills/image-to-editable-ppt/README.md) |
| `ling-gui-agent-skill` | Ling-3.0-flash-VL [OpenRouter](https://openrouter.ai/models/inclusionai/ling-3.0-flash-vl) [Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL) [ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-3.0-flash-VL) | Vision-grounded GUI automation across desktop (macOS/Win/Linux) and Android | [README](resources/recommended-skills/ling-gui-agent-skill/README.md) |
| `ling-ui-design` | Ming-Image-0.1-Design [Hugging Face](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design) [ModelScope](https://www.modelscope.cn/models/inclusionAI/Ming-Image-0.1-Design)<br>Ming-Image-0.1-Design-Layer [Hugging Face](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer) [ModelScope](https://www.modelscope.cn/models/inclusionAI/Ming-Image-0.1-Design-Layer) | Visual-first web UI generation, layer decomposition, asset extraction, and browser validation | [README](resources/recommended-skills/ling-ui-design/README.md) |

---

## Building Documentation

This project uses Jupytext to maintain all Jupyter Notebooks. All deployment guides and examples are authored in `src/` (MyST Markdown) and compiled to their delivery destinations via `scripts/compile_notebooks.py` (`*-zh.md` to `*-zh.ipynb`, `*.md` to `*.ipynb`).

After editing files under `src/`, run `python3 scripts/compile_notebooks.py` to compile the notebooks. Direct manual edits to `.ipynb` files are prohibited.

---

## Acknowledgements

- Special thanks to the NVIDIA team [@ly01325](https://github.com/ly01325) for optimizing and contributing the [SGLang Humming Deployment Guide](guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-mxfp4-humming.ipynb), delivering Humming MoE operator acceleration, online FP8 LM Head, and MTP speculative decoding synergy on Grace Blackwell (GB10 / SM121) hardware.