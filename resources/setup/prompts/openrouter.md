[English](openrouter.md) | [简体中文](openrouter.zh-CN.md)

# OpenRouter setup prompt

Copy the entire block below into your local agent.

```text
Please configure the local coding agent I am using to access Ling through OpenRouter, add its model choices, and install the Ling GUI Agent Skill. Do the work directly using native configuration files and tools. You may use the official skills CLI or download the Skill files directly.

Connection contract:
- Provider label: Ling (openrouter); use a distinct provider ID ling_openrouter unless an existing equivalent entry can be safely reused.
- API key environment variable: OPENROUTER_API_KEY.
- OpenAI Responses base URL: https://openrouter.ai/api/v1.
- Anthropic Messages base URL: https://openrouter.ai/api.
- OpenAI Chat Completions and GUI Skill base URL: https://openrouter.ai/api/v1.
- Models: Ling-3.0-flash-VL = inclusionai/ling-3.0-flash-vl; Ling-3.0-flash = inclusionai/ling-3.0-flash. Default: inclusionai/ling-3.0-flash-vl. Do not invent additional model IDs or silently substitute another model.

First identify the current agent, its version, operating system, and effective configuration directory. Identify yourself from the current runtime and configure that agent; do not ask me to choose from a fixed list. For other agents, inspect their native provider, model, credential and Skill mechanisms and apply the connection contract above. If runtime identity cannot be established, ask only for the missing environment information before modifying files. Use the effective configuration locations in the current environment. If you cannot access my local files, explain the missing local access rather than claiming completion. Check installed help/source and current official documentation for supported settings before editing.

Briefly explain what will change and how I can restore it, then proceed. Save a private timestamped backup and a change manifest before editing, recording original values, newly created files and the post-change hashes. Preserve existing login credentials, providers, models, skills, and unrelated settings. Repeated runs should update the same managed entry, not duplicate it. Never overwrite someone else's same-named configuration or delete conflicting settings without explaining the conflict.

Install or update the complete Skill directory from:
https://github.com/inclusionAI/ling-cookbook/tree/main/resources/recommended-skills/ling-gui-agent-skill
Configure the Skill .env: set LING_BASE_URL=https://openrouter.ai/api/v1, LING_MODEL=inclusionai/ling-3.0-flash-vl and LING_API_KEY from the protected credential. Do not print the resulting .env. Verify that the agent discovers the Skill. Do not launch GUI automation as part of installation.

Validate syntax and effective configuration without exposing secrets. Run a minimal harmless request through the selected agent, then check response content, completion status and the actual model/provider used. Separate configuration success, API success, and agent-level success; a successful HTTP response alone is insufficient. If unavailable, report the blocker and leave a usable restore path. Do not activate a broken default silently. Give me a short result: configured agent/provider/model, whether restart is needed, how to select Ling and return to the original provider, Skill status, and the backup location. Save exact restore instructions that revert only your changes and preserve later edits.
```
