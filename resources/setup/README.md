[English](README.md) | [Simplified Chinese](README.zh-CN.md)

# OpenRouter Ling installer bundle

Bundle version: `0.4.0`

Default model: `inclusionai/ling-3.0-flash-vl`

This directory contains standalone installers for Claude Code CLI, Codex, and Hermes Agent. Unsuffixed `.sh` entrypoints are English; `.zh-CN.sh` entrypoints are Simplified Chinese. Run all commands from this directory so the relative paths below work unchanged.

| Target | English | Simplified Chinese |
| --- | --- | --- |
| Claude Code CLI | `./claude-code-ling-3-flash-vl-setup.sh` | `./claude-code-ling-3-flash-vl-setup.zh-CN.sh` |
| Codex | `./codex-ling-3-flash-vl-setup.sh` | `./codex-ling-3-flash-vl-setup.zh-CN.sh` |
| Hermes Agent | `./hermes-ling-3-flash-vl-setup.sh` | `./hermes-ling-3-flash-vl-setup.zh-CN.sh` |

## Use an installer

Replace the Codex filename below with the Claude Code or Hermes filename from the table when needed:

```bash
./codex-ling-3-flash-vl-setup.sh --install
./codex-ling-3-flash-vl-setup.sh --status
./codex-ling-3-flash-vl-setup.sh --self-test
./codex-ling-3-flash-vl-setup.sh --version
./codex-ling-3-flash-vl-setup.sh --uninstall --yes
```

Without an argument, an installer opens its interactive menu. Installation prompts for the API key without displaying it. For non-interactive use, inject `OPENROUTER_API_KEY` through a trusted environment or secret manager; never pass a key as a command-line argument.

```bash
OPENROUTER_API_KEY='injected-by-your-secret-manager' ./codex-ling-3-flash-vl-setup.sh --install
```

Codex offers `none` (no thinking) and `high` (thinking), defaulting to `high`. Theta Flash-VL and OpenRouter Flash were verified with live Responses requests; OpenRouter Flash-VL follows the user-confirmed same-mode contract pending registration. Theta reports zero reasoning tokens even with nonempty reasoning content.

Codex displays `Ling (openrouter)` as its provider name. The build selects this label; model picker labels remain separate. Reinstall the rebuilt bundle and restart Codex to apply it.

## Runtime overrides

The bundle defaults to `inclusionai/ling-3.0-flash-vl`. You may override `LING_MODEL`, `LING_BASE_URL`, and `LING_MESSAGES_BASE_URL` when running an installer. Target-specific variables such as `LING_CODEX_MODEL`, `LING_CLAUDE_MODEL`, and `LING_HERMES_MODEL` take precedence. `LING_GUI_BASE_URL` controls the optional GUI Skill endpoint.

Thinking defaults are aligned to on. Hermes stores `high` per Ling model in `agent.reasoning_overrides`; use `/reasoning none` or `/reasoning high`. Other models and the global preference are preserved; uninstall restores unchanged managed entries while preserving later edits. Claude sets `alwaysThinkingEnabled=true`, declares only `thinking` capabilities, and removes stored `MAX_THINKING_TOKENS` / `CLAUDE_CODE_DISABLE_THINKING` locks. Use `/config` or Option+T (macOS) / Alt+T to toggle; external settings may override it. No graded effort or adaptive-thinking support is implied.

Both gateways' Messages enable/disable probes passed. Installed Hermes source revision `6327930` maps custom providers to top-level Chat Completions `reasoning_effort=high/none`; its transport was tested against both gateways. Theta ignores nested `reasoning.enabled=false`. Claude's local 2.1.187 executable was SIGKILL-terminated even for `--version`, so its native request behavior and gateway capability detection remain unverified despite settings tests. OpenRouter Flash-VL still awaits registration and live validation.

Requirements: Bash, Python 3, and the selected target CLI. The optional GUI Skill installation also requires `npx`. Use `LING_INSTALL_GUI_SKILL=n` to skip it. Each installer preserves the existing target configuration according to its install and uninstall flow.

The GUI Skill source is embedded at build time: `https://github.com/inclusionAI/ling-cookbook/tree/main/resources/recommended-skills/ling-gui-agent-skill`. Override it at runtime with `LING_GUI_AGENT_SKILL_SOURCE`; an empty value uses the embedded default. This source is shared by all three installers in this bundle.

## Switch OpenRouter models

The default is `inclusionai/ling-3.0-flash-vl` (`Ling-3.0-flash-VL`). The other option is `inclusionai/ling-3.0-flash` (`Ling-3.0-flash`). Install this bundle once, restart the target, then switch inside the harness without rebuilding or reinstalling:

| Harness | Native switching |
| --- | --- |
| Codex | Open `/model` and select either display name from the installed `model_catalog_json` catalog. |
| Claude Code | Open `/model`. Version 2.1.242+ uses labeled `modelPicker` rows. Older versions can use `/model opus` for Flash-VL and `/model sonnet` for Flash; alias label rendering depends on the version. |
| Hermes | Open `/model`, then the `antchat-ling3` provider, or use `/model Ling-3.0-flash-VL` and `/model Ling-3.0-flash`. The provider has both model IDs and named aliases; the current Hermes picker lists raw IDs, while the current-model display resolves aliases (a shorter existing alias may take precedence). |

Claude's installer no longer pins `ANTHROPIC_MODEL` or forces the subagent model for this profile. External environment variables, project settings, or managed model restrictions can still take precedence. Preserve or adjust those explicitly if switching is blocked. Rebuilding alone does not update an existing installation.

Both models declare a 262,144-token context window (262K), embedded by the OpenRouter build. Codex writes `context_window` and `max_context_window` per model, retaining its 90% effective-window reserve (about 236K available). Claude Code writes `CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144` for this gateway profile. Hermes writes per-model `context_length=262144`, which also applies when switching models. This is a client configuration, not a long-context API validation.

Live Responses probes on 2026-09-08 verified Flash with `high` (29 reasoning tokens) and `none` (0), both completed with the correct answer. Both Flash and Flash-VL offer `none` and `high`, defaulting to `high`. Flash-VL follows the user-confirmed same-mode contract; its API probe returned HTTP 400 before model registration, so live verification is pending. These probes do not certify vision or tool capabilities. Codex retains its conservative text-only capability declaration.

References: [Claude model picker](https://code.claude.com/docs/en/settings-reference#modelpicker), [Codex configuration](https://developers.openai.com/codex/config-reference), [Hermes providers](https://hermes-agent.nousresearch.com/docs/integrations/providers/).
