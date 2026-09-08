[English](README.md) | [Simplified Chinese](README.zh-CN.md)

# Use Ling with OpenRouter

Bundle version: `0.4.0` · Default model: `inclusionai/ling-3.0-flash-vl`

Have your OpenRouter API key ready. You need Bash, Python 3, and the CLI you want to configure. Download this folder and open a terminal in it.

## Open the setup menu

Copy **one** command for your agent to view all menu options. Choose 1 to install/update, 2 for status, 3 for self-test, 9 to uninstall/restore, or 0 to exit. Installation prompts for your API key with hidden input. Restart the agent after installation.

**Claude Code**

```bash
bash ./claude-code-ling-3-flash-vl-openrouter-setup.sh
```

**Codex**

```bash
bash ./codex-ling-3-flash-vl-openrouter-setup.sh
```

**Hermes Agent**

```bash
bash ./hermes-ling-3-flash-vl-openrouter-setup.sh
```

The installer preserves existing configuration through its backup/restore flow. The optional GUI Skill requires `npx`; to skip it, prefix your command with `LING_INSTALL_GUI_SKILL=n`. For automated installation, inject `OPENROUTER_API_KEY` through your environment or secret manager.

## Switch models

Default: **Ling-3.0-flash-VL**. Use `/model` to select **Ling-3.0-flash** instead; in Codex Desktop, use the model selector beside the input box. Restart after installing to load the choices.

## Check or remove configuration

Run your agent's script **without arguments** to open its menu for status, self-test, or uninstall. Self-test makes an API request.

```bash
bash ./codex-ling-3-flash-vl-openrouter-setup.sh
```

Use the corresponding command above for Claude Code or Hermes. Uninstall restores the configuration managed by the installer; the optional GUI Skill and its `.env` remain installed.

For direct operation, append `--install`, `--status`, `--self-test`, `--uninstall`, `--version`, or `--help` to your command. `--install` prints the planned changes and restore instructions before requesting your API key.

Chinese-language entrypoints: `./claude-code-ling-3-flash-vl-openrouter-setup.zh-CN.sh`, `./codex-ling-3-flash-vl-openrouter-setup.zh-CN.sh`, `./hermes-ling-3-flash-vl-openrouter-setup.zh-CN.sh`.
