[English](README.md) | [Simplified Chinese](README.zh-CN.md)

# Use Ling with OpenRouter

Bundle version: `0.4.0` · Default model: `inclusionai/ling-3.0-flash-vl`

Have your OpenRouter API key ready. You need Bash, Python 3, and the CLI you want to configure. Download this folder and open a terminal in it.

## Install

Copy **one** command for your agent. Enter your API key when prompted; input is hidden. Restart the agent after installation.

**Claude Code**

```bash
bash ./claude-code-ling-3-flash-vl-setup.sh --install
```

**Codex**

```bash
bash ./codex-ling-3-flash-vl-setup.sh --install
```

**Hermes Agent**

```bash
bash ./hermes-ling-3-flash-vl-setup.sh --install
```

The installer preserves existing configuration through its backup/restore flow. The optional GUI Skill requires `npx`; to skip it, prefix your command with `LING_INSTALL_GUI_SKILL=n`. For automated installation, inject `OPENROUTER_API_KEY` through your environment or secret manager.

## Switch models

Default: **Ling-3.0-flash-VL**. Use `/model` to select **Ling-3.0-flash** instead; in Codex Desktop, use the model selector beside the input box. Restart after installing to load the choices.

## Check or remove configuration

Run your agent's script **without arguments** to open its menu for status, self-test, or uninstall. Self-test makes an API request.

```bash
bash ./codex-ling-3-flash-vl-setup.sh
```

For Claude Code or Hermes, use its installation command above with `--install` removed. Uninstall follows the installer's backup/restore flow.

Chinese-language entrypoints: `./claude-code-ling-3-flash-vl-setup.zh-CN.sh`, `./codex-ling-3-flash-vl-setup.zh-CN.sh`, `./hermes-ling-3-flash-vl-setup.zh-CN.sh`.
