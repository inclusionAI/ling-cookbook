[English](README.md) | [Simplified Chinese](README.zh-CN.md)

# Use Ling with OpenRouter

Bundle version: `0.4.0` · Default model: `inclusionai/ling-3.0-flash-vl`

Have your OpenRouter API key ready. You need Bash, Python 3, and the CLI you want to configure. Run the commands below in a terminal; they download the script to the current directory and open its menu. `curl` is required.

## Configure through your agent

Copy a prompt into your local agent to configure Ling and install the GUI Skill.

- OpenRouter: [English](prompts/openrouter.md) · [简体中文](prompts/openrouter.zh-CN.md)
- Theta: [English](prompts/theta.md) · [简体中文](prompts/theta.zh-CN.md)


## Open the setup menu

Copy **one** command for your agent to view all menu options. Choose 1 to install/update, 2 for status, 3 for self-test, 9 to uninstall/restore, or 0 to exit. Installation prompts for your API key with hidden input. Restart the agent after installation.

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

The installer preserves existing configuration through its backup/restore flow. The optional GUI Skill requires `npx`; to skip it, prefix your command with `LING_INSTALL_GUI_SKILL=n`. For automated installation, inject `OPENROUTER_API_KEY` through your environment or secret manager.

## Switch models

Default: **Ling-3.0-flash-VL**. Use `/model` to select **Ling-3.0-flash** instead; in Codex Desktop, use the model selector beside the input box. Restart after installing to load the choices.

## Check or remove configuration

Run your agent's script **without arguments** to open its menu for status, self-test, or uninstall. Self-test makes an API request.

```bash
curl -fL --retry 3 -o codex-ling-3-flash-vl-openrouter-setup.sh \
  https://raw.githubusercontent.com/inclusionAI/ling-cookbook/refs/heads/main/resources/setup/codex-ling-3-flash-vl-openrouter-setup.sh && \
bash codex-ling-3-flash-vl-openrouter-setup.sh
```

Use the corresponding command above for Claude Code or Hermes. Uninstall restores the configuration managed by the installer.

For direct operation, append `--install`, `--status`, `--self-test`, `--uninstall`, `--version`, or `--help` to the final `bash` command. `--install` prints the planned changes and restore instructions before requesting your API key.

Chinese-language entrypoints: [claude-code-ling-3-flash-vl-openrouter-setup.zh-CN.sh](https://raw.githubusercontent.com/inclusionAI/ling-cookbook/refs/heads/main/resources/setup/claude-code-ling-3-flash-vl-openrouter-setup.zh-CN.sh), [codex-ling-3-flash-vl-openrouter-setup.zh-CN.sh](https://raw.githubusercontent.com/inclusionAI/ling-cookbook/refs/heads/main/resources/setup/codex-ling-3-flash-vl-openrouter-setup.zh-CN.sh), [hermes-ling-3-flash-vl-openrouter-setup.zh-CN.sh](https://raw.githubusercontent.com/inclusionAI/ling-cookbook/refs/heads/main/resources/setup/hermes-ling-3-flash-vl-openrouter-setup.zh-CN.sh).
