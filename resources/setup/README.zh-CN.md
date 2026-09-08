[English](README.md) | [简体中文](README.zh-CN.md)

# OpenRouter Ling 安装脚本包

构建版本：`0.4.0`

默认模型：`inclusionai/ling-3.0-flash-vl`

本目录包含 Claude Code CLI、Codex 和 Hermes Agent 的独立安装脚本。无语言后缀的 `.sh` 文件为英文默认版，`.zh-CN.sh` 文件为简体中文版。请在本目录运行命令，以下相对路径可直接使用。

| 目标 | 英文版 | 简体中文版 |
| --- | --- | --- |
| Claude Code CLI | `./claude-code-ling-3-flash-vl-setup.sh` | `./claude-code-ling-3-flash-vl-setup.zh-CN.sh` |
| Codex | `./codex-ling-3-flash-vl-setup.sh` | `./codex-ling-3-flash-vl-setup.zh-CN.sh` |
| Hermes Agent | `./hermes-ling-3-flash-vl-setup.sh` | `./hermes-ling-3-flash-vl-setup.zh-CN.sh` |

## 使用安装脚本

以下示例使用 Codex；如需 Claude Code 或 Hermes，请替换为表格中的对应文件名：

```bash
./codex-ling-3-flash-vl-setup.zh-CN.sh --install
./codex-ling-3-flash-vl-setup.zh-CN.sh --status
./codex-ling-3-flash-vl-setup.zh-CN.sh --self-test
./codex-ling-3-flash-vl-setup.zh-CN.sh --version
./codex-ling-3-flash-vl-setup.zh-CN.sh --uninstall --yes
```

不带参数运行会打开交互菜单。安装时会隐藏输入 API Key。非交互环境请通过可信环境或密钥管理系统注入 `OPENROUTER_API_KEY`，不要把 Key 作为命令行参数传入。

```bash
OPENROUTER_API_KEY='由密钥系统注入' ./codex-ling-3-flash-vl-setup.zh-CN.sh --install
```

Codex 提供 `none`（不思考）和 `high`（思考），默认 `high`。Theta Flash-VL 与 OpenRouter Flash 已完成 Responses 实测；OpenRouter Flash-VL 按用户确认的同模式契约配置，待注册后验证。Theta 即使返回非空 reasoning 内容，推理 token 统计仍为 0。

Codex 的提供方名称显示为 `Ling (openrouter)`，由构建选择；模型选择器名称单独配置。重新安装构建产物并重启 Codex 后生效。

## 运行时覆盖

本产物默认使用 `inclusionai/ling-3.0-flash-vl`。执行安装脚本时可通过 `LING_MODEL`、`LING_BASE_URL`、`LING_MESSAGES_BASE_URL` 覆盖配置。`LING_CODEX_MODEL`、`LING_CLAUDE_MODEL`、`LING_HERMES_MODEL` 等目标专用变量优先；`LING_GUI_BASE_URL` 控制可选 GUI Skill 的服务地址。

思考默认统一开启。Hermes 在 `agent.reasoning_overrides` 按 Ling 模型写入 `high`，使用 `/reasoning none` 或 `/reasoning high` 切换；保留其他模型和全局偏好，卸载恢复未被用户改动的管理项。Claude 设置 `alwaysThinkingEnabled=true`，仅声明 `thinking` 能力，移除 settings 中的 `MAX_THINKING_TOKENS`／`CLAUDE_CODE_DISABLE_THINKING` 锁定项。使用 `/config` 或 macOS Option+T／其他系统 Alt+T 切换，外部配置仍可能覆盖。不代表支持分级 effort 或 adaptive thinking。

两家网关的 Messages 开关实测通过。本机 Hermes 源码 `6327930` 将自定义提供方映射为 Chat Completions 顶层 `reasoning_effort=high/none`，已针对两家网关验证传输层。Theta 会忽略嵌套 `reasoning.enabled=false`。本机 Claude 2.1.187 连 `--version` 都被 SIGKILL 终止，配置测试不代表原生请求及网关能力识别已通过。OpenRouter Flash-VL 仍待注册后实测。

运行要求：Bash、Python 3 和所选目标的 CLI。安装可选 GUI Skill 还需要 `npx`；设置 `LING_INSTALL_GUI_SKILL=n` 可跳过。每个安装器会按照自身安装和卸载流程保留原有目标配置。

GUI Skill 来源地址在构建时写入产物：`https://github.com/inclusionAI/ling-cookbook/tree/main/resources/recommended-skills/ling-gui-agent-skill`，三个安装器共用。执行时可通过 `LING_GUI_AGENT_SKILL_SOURCE` 再次覆盖；空值使用内嵌默认地址。

## 切换 OpenRouter 模型

默认模型为 `inclusionai/ling-3.0-flash-vl`（`Ling-3.0-flash-VL`），另一个选项为 `inclusionai/ling-3.0-flash`（`Ling-3.0-flash`）。安装一次本产物并重启目标后，即可在 harness 内切换，无需重新构建或安装：

| Harness | 原生切换方式 |
| --- | --- |
| Codex | 打开 `/model`，从安装的 `model_catalog_json` 目录选择上述任一展示名称。 |
| Claude Code | 打开 `/model`。v2.1.242+ 使用带展示名称的 `modelPicker`；旧版可用 `/model opus` 切到 Flash-VL、`/model sonnet` 切到 Flash，别名名称的展示取决于版本。 |
| Hermes | 打开 `/model` 后选择 `antchat-ling3` provider，或直接使用 `/model Ling-3.0-flash-VL`、`/model Ling-3.0-flash`。已配置两个模型 ID 及命名别名；当前 Hermes 的选择列表仍显示原始 ID，当前模型展示会解析别名（已有更短别名可能优先）。 |

此配置不再通过 Claude 的 `ANTHROPIC_MODEL` 或强制子代理模型锁定选择。外部环境变量、项目设置或管理策略仍可能优先；如果阻止切换，需要明确调整这些设置。仅重新构建不会更新已有安装。

两个模型均配置为 262,144 tokens（262K），由 OpenRouter 构建脚本指定。Codex 为每个模型写入 `context_window` 和 `max_context_window`，保留 90% 可用窗口比例（约 236K 可用）；Claude Code 为此网关配置写入 `CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`；Hermes 为每个模型写入 `context_length=262144`，切换模型时也会读取。这是客户端配置，不代表已完成长上下文 API 验证。

2026-09-08 的 Responses 实测确认 Flash 的 `high`（29 个 reasoning tokens）与 `none`（0 个）均正常完成且答案正确，因此 Flash 与 Flash-VL 均配置这两个选项，默认仍为 `high`。Flash-VL 按用户确认的同模式契约对齐；注册前 API 返回 HTTP 400（无效模型 ID），仍待注册后实测。这些探测不代表视觉及工具能力验证；Codex 仍保留保守的纯文本能力声明。

参考：[Claude 模型选择器](https://code.claude.com/docs/en/settings-reference#modelpicker)、[Codex 配置](https://developers.openai.com/codex/config-reference)、[Hermes provider](https://hermes-agent.nousresearch.com/docs/integrations/providers/)。
