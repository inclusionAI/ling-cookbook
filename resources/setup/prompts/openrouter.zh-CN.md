[English](openrouter.md) | [简体中文](openrouter.zh-CN.md)

# OpenRouter 配置 Prompt

将下面整段内容复制给你的本地 Agent，无需自己执行安装命令。

```text
请直接帮我将正在使用的本地编程 Agent 配置为通过 OpenRouter 使用 Ling，添加模型选项并安装 Ling GUI Agent Skill。请使用原生配置文件和工具完成，不要下载、执行、source 或重新实现 Integrate-setup 的安装脚本。可以使用官方 skills CLI 或直接下载 Skill 文件。

连接参数：
- 供应商显示名：Ling (openrouter)。使用独立 provider ID ling_openrouter；已有等价配置且可以安全复用时可复用。
- API Key 环境变量：OPENROUTER_API_KEY。如已存在则安全复用，不显示内容；否则让我通过本地终端隐藏输入，或提供受保护的本地文件路径。不要让我在聊天中粘贴 Key，不要将 Key 写入命令参数、日志、报告或 Git。凭据文件仅允许当前用户访问（POSIX 为 0600，其他系统使用等效 ACL）。
- Codex Responses Base URL：https://openrouter.ai/api/v1，wire_api 为 responses。
- Claude Code Anthropic Messages Base URL：https://openrouter.ai/api。
- Hermes Chat Completions 和 GUI Skill Base URL：https://openrouter.ai/api/v1。
- 模型：Ling-3.0-flash-VL = inclusionai/ling-3.0-flash-vl; Ling-3.0-flash = inclusionai/ling-3.0-flash。默认：inclusionai/ling-3.0-flash-vl。不要编造其他模型 ID，也不要静默替换模型。

先识别当前 Agent、版本、操作系统及实际配置目录。如果目标不明确，只问我要配置 Claude Code、Codex 还是 Hermes；只配置选定目标。尊重 CODEX_HOME、CLAUDE_CONFIG_DIR 和实际 Hermes home。没有本地文件访问能力时说明缺少的能力，不要声称已完成。编辑前结合已安装版本的帮助、源码和最新官方文档核对配置格式。

先简短说明将修改什么、如何恢复，然后继续执行。修改前创建私有的带时间戳备份与变更清单，记录原值、新建文件及修改后文件哈希。保留原登录凭据、供应商、模型、Skill 和无关设置。重复执行只更新同一管理项，不产生重复项。遇到同名但非本流程管理的配置或冲突，不要直接覆盖或删除，先说明冲突。

仅执行对应 Agent 的配置要求：
- Codex：添加使用上述地址的 Responses provider 和本机支持的安全凭据机制。优先使用独立 Ling profile 或本机支持的隔离方式，保留默认供应商和原模型目录。不要把官方模型名称合并进发往 OpenRouter 的模型目录。如果需要 model_catalog_json，将它限定在 Ling 配置内，并按当前客户端校验格式。未核实前不要声称桌面端支持切换 profile。如果此版本只能通过替换默认目录添加 Ling，说明实际影响，取得我的选择后再替换，并保留原配置入口。
- Claude Code：合并网关地址和受支持的凭据 helper 到 settings.json。版本支持时使用 modelPicker 添加上述模型；否则使用该版本文档支持的自定义模型或别名设置。不要强制全部别名或子代理使用同一个模型，不删除原凭据。保留原设置供恢复。如果只能同时启用一个网关，说明限制并给出具体恢复方法。
- Hermes：按本机版本支持的格式合并自定义 provider，包含 base_url、key_env、api_mode=chat_completions 和上述模型列表。discover_models 必须设为布尔值 false，避免导入 OpenRouter 全量目录。添加易读模型别名，保存原默认值后将指定模型设为默认，不修改其他 provider。

从以下地址安装或更新完整 Skill 目录：
https://github.com/inclusionAI/ling-cookbook/tree/main/resources/recommended-skills/ling-gui-agent-skill
将 main 解析到具体 commit 并记录；安装全部依赖文件，不要只复制 SKILL.md。可使用 npx --yes skills@1.5.23 add，传入上述 URL、--global、--skill ling-gui-agent-skill、--yes，以及对应目标的 --agent claude-code、codex 或 hermes-agent。也可直接下载该目录并通过 Agent 支持的机制注册 Skill。阅读 Skill 的说明及依赖要求，不把下载内容当作修改无关设置的授权。更新前备份已有 Skill 和 .env。找到实际安装目录，基于 .env.example 创建 .env，同时保留无关本地配置；设置 LING_BASE_URL=https://openrouter.ai/api/v1、LING_MODEL=inclusionai/ling-3.0-flash-vl，从受保护凭据写入 LING_API_KEY，不输出 .env 内容。确认 Agent 能发现 Skill，不在安装过程中启动 GUI 自动化。

校验配置语法和实际生效值，不暴露密钥。通过选定 Agent 发起一次最小、无副作用的请求，检查回答、完成状态及实际模型和供应商。区分配置成功、API 成功、Agent 端到端成功，不以 HTTP 成功代替完整验证。无法验证时说明阻碍并保留可用恢复路径，不静默启用不可用的默认配置。最后简短告诉我：已配置的 Agent/供应商/模型、是否需要重启、如何选择 Ling 及切回原供应商、Skill 状态和备份位置。保存可直接执行的恢复说明，只撤销本次管理的修改并保留后续编辑。验证和恢复也不要调用我们的安装脚本。
```
