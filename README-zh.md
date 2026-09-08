<p align="right">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

# Ling Cookbook

> `inclusionAI/ling-cookbook` 是 Ling 系列大语言模型的部署与开发指南仓库，提供已验证的硬件部署方案和模型的工程特性与使用技巧。当前正在开发中。

---

## 相关资源与社群

- **官方网站**：[ant-ling.com](https://www.ant-ling.com/)
- **Hugging Face**：[inclusionAI](https://huggingface.co/inclusionAI)
- **ModelScope**：[inclusionAI](https://www.modelscope.cn/organization/inclusionAI)
- **GitHub**：[inclusionAI/ling-cookbook](https://github.com/inclusionAI/ling-cookbook)
- **Discord**：[Ling Community](https://discord.com/invite/GNaQc8WC5T)
- **X (Twitter)**：[@AntLingAGI](https://x.com/AntLingAGI)

---

## 内容目录

### Agent 安装包（`resources/setup/`）

面向 Claude Code、Codex 和 Hermes Agent 的独立 OpenRouter 安装包：
[English guide](resources/setup/README.md) · [简体中文](resources/setup/README.zh-CN.md)。

### 本地部署 (`guide/local-deploy/`)

包含在消费级硬件和设备上部署 Ling 系列模型提供推理服务的示例。

#### Ling-3.0-flash

<p align="right">
  <a href="README.md#ling-30-flash">English</a> | <b>简体中文</b>
</p>

| 模型 | 设备 | 框架 | 量化 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| Ling-3.0-flash | 单台 NVIDIA DGX Spark | SGLang | MXFP4 (速度优化) | [文档: SGLang MXFP4 Humming 优化部署指南](guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-mxfp4-humming-zh.ipynb) |
| Ling-3.0-flash | 单台 NVIDIA DGX Spark | SGLang | MXFP4 | [文档: SGLang MXFP4 部署指南](guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-mxfp4-zh.ipynb) |
| Ling-3.0-flash | 单台 NVIDIA DGX Spark | SGLang | INT4 | [文档: SGLang INT4 部署指南](guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-int4-zh.ipynb) |
| Ling-3.0-flash | 单台 NVIDIA DGX Spark | vLLM | FP4 | [文档: vLLM FP4 部署指南](guide/local-deploy/ling-3.0-flash/dgx-spark-vllm-ling-3-flash-fp4-zh.ipynb) |
| Ling-3.0-flash | 单台 NVIDIA DGX Spark | vLLM | INT4 | [文档: vLLM INT4 部署指南](guide/local-deploy/ling-3.0-flash/dgx-spark-vllm-ling-3-flash-int4-zh.ipynb) |
| Ling-3.0-flash | 单台 NVIDIA DGX Spark | llama.cpp | GGUF (Q6_K / Q4_K) | [文档: llama.cpp GGUF 部署指南](guide/local-deploy/ling-3.0-flash/dgx-spark-llamacpp-ling-3-flash-q4-gguf-zh.ipynb) |

#### Ling-3.0-tiny

<p align="right">
  <a href="README.md#ling-30-tiny">English</a> | <b>简体中文</b>
</p>

| 模型 | 设备 | 框架 | 量化 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| Ling-3.0-tiny | 单台 NVIDIA DGX Spark | llama.cpp | BF16 (GGUF) | [文档: llama.cpp BF16 部署指南](guide/local-deploy/ling-3.0-tiny/dgx-spark-llamacpp-ling-3-tiny-bf16-gguf-zh.ipynb) |
| Ling-3.0-tiny | 单台 NVIDIA DGX Spark | SGLang | BF16 | [文档: SGLang BF16 部署指南](guide/local-deploy/ling-3.0-tiny/dgx-spark-sglang-ling-3-tiny-bf16-zh.ipynb) |
| Ling-3.0-tiny | 单台 NVIDIA DGX Spark | vLLM | BF16 | [文档: vLLM BF16 部署指南](guide/local-deploy/ling-3.0-tiny/dgx-spark-vllm-ling-3-tiny-bf16-zh.ipynb) |
| Ling-3.0-tiny | Apple Silicon Mac | Ollama | INT4 / FP8 / BF16 | [文档: Ollama 端侧部署指南](guide/local-deploy/ling-3.0-tiny/mac-ollama-ling-3-tiny-zh.ipynb) |

---

## 构建文档

本项目用 Jupytext 维护所有 Jupyter Notebook。所有部署指南及用例源码维护在 `src/` 中；由 `scripts/compile_notebooks.py` 构建到根目录下的各交付目录（`*-zh.md` 对应构建至 `*-zh.ipynb`，`*.md` 对应构建至 `*.ipynb`）。 

修改 `src/` 下的内容后，运行 `python3 scripts/compile_notebooks.py` 即可完成文档构建。严禁直接编辑 Jupyter Notebook 文件。

---

## 致谢 (Acknowledgements)

- 感谢 Nvidia 团队 [@ly01325](https://github.com/ly01325) 优化和提供 [SGLang Humming 优化部署指南](guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-mxfp4-humming-zh.ipynb)，针对 Grace Blackwell (GB10 / SM121) 硬件实现了 Humming MoE 算子加速、在线 FP8 LM Head 与 MTP 投机解码协同优化。
