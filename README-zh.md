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
| Ling-3.0-tiny | Apple Silicon Mac | llama.cpp | BF16 / Q8_0 / Q4_K_M (GGUF) | [文档: llama.cpp Metal 端侧部署指南](guide/local-deploy/ling-3.0-tiny/mac-llamacpp-ling-3-tiny-zh.ipynb) |
| Ling-3.0-tiny | Apple Silicon Mac | MLX | BF16 / MXFP8 / 4-bit | [文档: MLX-LM 原生端侧部署指南](guide/local-deploy/ling-3.0-tiny/mac-mlx-ling-3-tiny-zh.ipynb) |

<a id="recommended-skills"></a>
### 部分能力 Agent Skill

我们的部分模型提供了相对独特的多模态 / Agentic 能力。为了方便大家体验这些能力，我们提供了一些预制 Agent Skill 。你可以下载到本地使用，也可以 clone 仓库，让你的 agent 阅读这些 skill，审计其安全性，并协助你体验模型能力。

这些 agent skill 和对应的模型包括：

<p align="right">
  <a href="README.md#recommended-skills">English</a> | <b>简体中文</b>
</p>

| 名字 | 演示模型 | 简单介绍 | 链接 |
| :--- | :--- | :--- | :--- |
| `image-to-editable-ppt` | Ming-Image-0.1-Design-Layer [Hugging Face](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer) [ModelScope](https://www.modelscope.cn/models/inclusionAI/Ming-Image-0.1-Design-Layer) | 将单张幻灯片图片/截图逆向重建为原生可编辑 `.pptx` | [README](resources/recommended-skills/image-to-editable-ppt/README.md) |
| `ling-gui-agent-skill` | Ling-3.0-flash-VL [OpenRouter](https://openrouter.ai/models/inclusionai/ling-3.0-flash-vl) [Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL) [ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-3.0-flash-VL) | 基于视觉定位与键鼠/触控操作的跨平台桌面与 Android GUI 自动化 | [README](resources/recommended-skills/ling-gui-agent-skill/README.md) |
| `ling-ui-design` | Ming-Image-0.1-Design [Hugging Face](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design) [ModelScope](https://www.modelscope.cn/models/inclusionAI/Ming-Image-0.1-Design)<br>Ming-Image-0.1-Design-Layer [Hugging Face](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer) [ModelScope](https://www.modelscope.cn/models/inclusionAI/Ming-Image-0.1-Design-Layer) | 视觉优先的前端界面生成、图层解耦、切图提取与浏览器校验闭环 | [README](resources/recommended-skills/ling-ui-design/README.md) |

---

## 构建文档

本项目用 Jupytext 维护所有 Jupyter Notebook。所有部署指南及用例源码维护在 `src/` 中；由 `scripts/compile_notebooks.py` 构建到根目录下的各交付目录（`*-zh.md` 对应构建至 `*-zh.ipynb`，`*.md` 对应构建至 `*.ipynb`）。 

修改 `src/` 下的内容后，运行 `python3 scripts/compile_notebooks.py` 即可完成文档构建。严禁直接编辑 Jupyter Notebook 文件。

---

## 致谢

- 感谢 Nvidia 团队 [@ly01325](https://github.com/ly01325) 优化和提供 [SGLang Humming 优化部署指南](guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-mxfp4-humming-zh.ipynb)，针对 Grace Blackwell (GB10 / SM121) 硬件实现了 Humming MoE 算子加速、在线 FP8 LM Head 与 MTP 投机解码协同优化。
