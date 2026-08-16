# Ling Cookbook

> Ling Cookbook provides deployment recipes and practical examples for the Ling model family. Work in progress.

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

### 本地部署 (`guide/local-deploy/`)

包含在消费级硬件和设备上部署 Ling 系列模型提供推理服务的示例。

#### Ling-3.0-flash

| 模型 | 设备                  | 框架 | 量化 | 链接 |
| :--- |:----------------------| :--- | :--- | :--- |
| Ling-3.0-flash | 单台 NVIDIA DGX Spark | SGLang | MXFP4 (W4A16 / W4A8) | [文档: SGLang MXFP4 部署指南](guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-mxfp4.ipynb) |
| Ling-3.0-flash | 单台 NVIDIA DGX Spark | SGLang | INT4 (W4A16) | [文档: SGLang INT4 部署指南](guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-int4.ipynb) |
| Ling-3.0-flash | 单台 NVIDIA DGX Spark | vLLM | FP4 (MXFP4) | [文档: vLLM FP4 部署指南](guide/local-deploy/ling-3.0-flash/dgx-spark-vllm-ling-3-flash-fp4.ipynb) |
| Ling-3.0-flash | 单台 NVIDIA DGX Spark | vLLM | INT4 (W4A16) | [文档: vLLM INT4 部署指南](guide/local-deploy/ling-3.0-flash/dgx-spark-vllm-ling-3-flash-int4.ipynb) |
| Ling-3.0-flash | 单台 NVIDIA DGX Spark | llama.cpp | GGUF (Q6_K / Q4_K) | [文档: llama.cpp GGUF 部署指南](guide/local-deploy/ling-3.0-flash/dgx-spark-llamacpp-ling-3-flash-q4-gguf.ipynb) |

#### Ling-3.0-tiny

| 模型 | 设备                  | 框架 | 量化 | 链接 |
| :--- |:----------------------| :--- | :--- | :--- |
| Ling-3.0-tiny | 单台 NVIDIA DGX Spark | llama.cpp | BF16 (GGUF) | [文档: llama.cpp BF16 部署指南](guide/local-deploy/ling-3.0-tiny/dgx-spark-llamacpp-ling-3-tiny-bf16-gguf.ipynb) |
| Ling-3.0-tiny | 单台 NVIDIA DGX Spark | SGLang | BF16 | [文档: SGLang BF16 部署指南](guide/local-deploy/ling-3.0-tiny/dgx-spark-sglang-ling-3-tiny-bf16.ipynb) |
| Ling-3.0-tiny | 单台 NVIDIA DGX Spark | vLLM | BF16 | [文档: vLLM BF16 部署指南](guide/local-deploy/ling-3.0-tiny/dgx-spark-vllm-ling-3-tiny-bf16.ipynb) |
