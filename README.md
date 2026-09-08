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

### Agent setup (`resources/setup/`)

Standalone OpenRouter installers for Claude Code, Codex, and Hermes Agent:
[English guide](resources/setup/README.md) · [简体中文](resources/setup/README.zh-CN.md).

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

---

## Building Documentation

This project uses Jupytext to maintain all Jupyter Notebooks. All deployment guides and examples are authored in `src/` (MyST Markdown) and compiled to their delivery destinations via `scripts/compile_notebooks.py` (`*-zh.md` to `*-zh.ipynb`, `*.md` to `*.ipynb`).

After editing files under `src/`, run `python3 scripts/compile_notebooks.py` to compile the notebooks. Direct manual edits to `.ipynb` files are prohibited.

---

## Acknowledgements

- Special thanks to the NVIDIA team [@ly01325](https://github.com/ly01325) for optimizing and contributing the [SGLang Humming Deployment Guide](guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-mxfp4-humming.ipynb), delivering Humming MoE operator acceleration, online FP8 LM Head, and MTP speculative decoding synergy on Grace Blackwell (GB10 / SM121) hardware.
