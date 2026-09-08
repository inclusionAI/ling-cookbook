[English](README.md) | [简体中文](README.zh-CN.md)

# 通过 OpenRouter 使用 Ling

构建版本：`0.4.0` · 默认模型：`inclusionai/ling-3.0-flash-vl`

准备好 OpenRouter API Key，并安装 Bash、Python 3 和要配置的 Agent CLI。在终端执行以下命令，即可将脚本下载到当前目录并打开菜单。需要安装 `curl`。

## 打开配置菜单

按使用的 Agent，复制**一条**命令查看全部菜单选项：1 安装或更新、2 查看状态、3 自检、9 卸载并恢复、0 退出。选择安装后按提示输入 API Key，输入内容不会显示。安装完成后重启 Agent。

**Claude Code**

```bash
curl -fL --retry 3 -o claude-code-ling-3-flash-vl-openrouter-setup.sh \
  https://raw.githubusercontent.com/inclusionAI/ling-cookbook/refs/heads/main/resources/setup/claude-code-ling-3-flash-vl-openrouter-setup.sh && \
bash claude-code-ling-3-flash-vl-openrouter-setup.sh
```

**Codex**

```bash
curl -fL --retry 3 -o codex-ling-3-flash-vl-openrouter-setup.sh \
  https://raw.githubusercontent.com/inclusionAI/ling-cookbook/refs/heads/main/resources/setup/codex-ling-3-flash-vl-openrouter-setup.sh && \
bash codex-ling-3-flash-vl-openrouter-setup.sh
```

**Hermes Agent**

```bash
curl -fL --retry 3 -o hermes-ling-3-flash-vl-openrouter-setup.sh \
  https://raw.githubusercontent.com/inclusionAI/ling-cookbook/refs/heads/main/resources/setup/hermes-ling-3-flash-vl-openrouter-setup.sh && \
bash hermes-ling-3-flash-vl-openrouter-setup.sh
```

安装器通过备份和恢复流程保留原配置。可选的 GUI Skill 需要 `npx`；如需跳过，在命令前加 `LING_INSTALL_GUI_SKILL=n`。自动化安装可通过环境或密钥管理系统注入 `OPENROUTER_API_KEY`。

## 切换模型

默认使用 **Ling-3.0-flash-VL**。输入 `/model` 可改选 **Ling-3.0-flash**；Codex 桌面端使用输入框旁的模型选择器。安装后需重启以加载选项。

## 检查或移除配置

**不带参数**运行对应脚本，即可在菜单中选择状态检查、自检或卸载。自检会发起 API 请求。

```bash
curl -fL --retry 3 -o codex-ling-3-flash-vl-openrouter-setup.sh \
  https://raw.githubusercontent.com/inclusionAI/ling-cookbook/refs/heads/main/resources/setup/codex-ling-3-flash-vl-openrouter-setup.sh && \
bash codex-ling-3-flash-vl-openrouter-setup.sh
```

Claude Code 和 Hermes 使用上方对应命令。卸载会恢复安装器管理的配置。

如需直接操作，可在最后一行 `bash` 命令后添加 `--install`、`--status`、`--self-test`、`--uninstall`、`--version` 或 `--help`。`--install` 会在询问 API Key 前说明将进行的修改和恢复方法。

以上默认使用英文脚本。如需中文交互，可替换为 [claude-code-ling-3-flash-vl-openrouter-setup.zh-CN.sh](https://raw.githubusercontent.com/inclusionAI/ling-cookbook/refs/heads/main/resources/setup/claude-code-ling-3-flash-vl-openrouter-setup.zh-CN.sh)、[codex-ling-3-flash-vl-openrouter-setup.zh-CN.sh](https://raw.githubusercontent.com/inclusionAI/ling-cookbook/refs/heads/main/resources/setup/codex-ling-3-flash-vl-openrouter-setup.zh-CN.sh) 或 [hermes-ling-3-flash-vl-openrouter-setup.zh-CN.sh](https://raw.githubusercontent.com/inclusionAI/ling-cookbook/refs/heads/main/resources/setup/hermes-ling-3-flash-vl-openrouter-setup.zh-CN.sh)。
