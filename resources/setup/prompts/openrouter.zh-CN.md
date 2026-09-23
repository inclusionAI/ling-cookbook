[English](openrouter.md) | [简体中文](openrouter.zh-CN.md)

# OpenRouter 配置 Prompt

将下面整段内容复制给你的本地 Agent。

```text
请直接帮我将正在使用的本地编程 Agent 配置为通过 OpenRouter 使用 Ling，添加模型选项并安装 Ling GUI Agent Skill。请使用原生配置文件和工具完成。可以使用官方 skills CLI 或直接下载 Skill 文件。

连接参数：
- 供应商显示名：Ling (openrouter)。使用独立 provider ID ling_openrouter；已有等价配置且可以安全复用时可复用。
- API Key 环境变量：OPENROUTER_API_KEY。
- OpenAI Responses Base URL：https://openrouter.ai/api/v1。
- Anthropic Messages Base URL：https://openrouter.ai/api。
- OpenAI Chat Completions 和 GUI Skill Base URL：https://openrouter.ai/api/v1。
- 模型：Ling-3.0-flash-VL = inclusionai/ling-3.0-flash-vl; Ling-3.0-flash = inclusionai/ling-3.0-flash。默认：inclusionai/ling-3.0-flash-vl。不要编造其他模型 ID，也不要静默替换模型。

先识别当前 Agent、版本、操作系统及实际配置目录。根据当前运行环境识别你自身所属的 Agent，并配置当前 Agent，不要求我从固定列表选择。其他 Agent 也应查阅自身原生的供应商、模型、凭据和 Skill 配置机制，应用上述连接参数。只有无法确定运行环境时，才询问缺失的环境信息，并在确定前不修改文件。使用当前环境实际生效的配置位置。没有本地文件访问能力时说明缺少的能力，不要声称已完成。编辑前结合已安装版本的帮助、源码和最新官方文档核对配置格式。

先简短说明将修改什么、如何恢复，然后继续执行。修改前创建私有的带时间戳备份与变更清单，记录原值、新建文件及修改后文件哈希。保留原登录凭据、供应商、模型、Skill 和无关设置。重复执行只更新同一管理项，不产生重复项。遇到同名但非本流程管理的配置或冲突，不要直接覆盖或删除，先说明冲突。

从以下地址安装或更新完整 Skill 目录：
https://github.com/inclusionAI/ling-cookbook/tree/main/resources/recommended-skills/ling-gui-agent-skill
配置 Skill 的 .env：设置 LING_BASE_URL=https://openrouter.ai/api/v1、LING_MODEL=inclusionai/ling-3.0-flash-vl，从受保护凭据写入 LING_API_KEY，不输出 .env 内容。确认 Agent 能发现 Skill，不在安装过程中启动 GUI 自动化。

校验配置语法和实际生效值，不暴露密钥。通过选定 Agent 发起一次最小、无副作用的请求，检查回答、完成状态及实际模型和供应商。区分配置成功、API 成功、Agent 端到端成功，不以 HTTP 成功代替完整验证。无法验证时说明阻碍并保留可用恢复路径，不静默启用不可用的默认配置。最后简短告诉我：已配置的 Agent/供应商/模型、是否需要重启、如何选择 Ling 及切回原供应商、Skill 状态和备份位置。保存可直接执行的恢复说明，只撤销本次管理的修改并保留后续编辑。
```
