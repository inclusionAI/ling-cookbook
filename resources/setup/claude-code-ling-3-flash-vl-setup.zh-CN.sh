#!/usr/bin/env bash
# Bundle version: 0.4.0
# Installer version: 0.4.0
# Provider: openrouter
# Entrypoint: claude-code-ling-3-flash-vl-setup.zh-CN.sh
# Model: inclusionai/ling-3.0-flash-vl
# Payload SHA256: cf18b2f405dd2b82b1098887df8135ca98c96f93f1199f2c387e4ee5243477d5
set -euo pipefail
if [ "${1:-}" = "--version" ]; then
  [ "$#" -eq 1 ] || { printf '%s\n' '--version accepts no extra arguments' >&2; exit 2; }
  cat <<'LING_BUNDLE_VERSION'
Bundle version: 0.4.0
Installer version: 0.4.0
Provider: openrouter
Entrypoint: claude-code-ling-3-flash-vl-setup.zh-CN.sh
Model: inclusionai/ling-3.0-flash-vl
Payload SHA256: cf18b2f405dd2b82b1098887df8135ca98c96f93f1199f2c387e4ee5243477d5
LING_BUNDLE_VERSION
  exit 0
fi
# BEGIN INSTALLER
export LING_SETUP_LANG=zh-CN
umask 077
set -euo pipefail

# BEGIN BUILD DEFAULTS
export LING_CODEX_PROVIDER_NAME='Ling (openrouter)'
if [ -z "${LING_MODEL:-}" ]; then
  export LING_MODEL=inclusionai/ling-3.0-flash-vl
else
  export LING_MODEL
fi
if [ -z "${LING_BASE_URL:-}" ]; then
  export LING_BASE_URL=https://openrouter.ai/api/v1
else
  export LING_BASE_URL
fi
if [ -z "${LING_MESSAGES_BASE_URL:-}" ]; then
  export LING_MESSAGES_BASE_URL=https://openrouter.ai/api
else
  export LING_MESSAGES_BASE_URL
fi
if [ -z "${LING_GUI_AGENT_SKILL_SOURCE:-}" ]; then
  export LING_GUI_AGENT_SKILL_SOURCE=https://github.com/inclusionAI/ling-cookbook/tree/main/resources/recommended-skills/ling-gui-agent-skill
else
  export LING_GUI_AGENT_SKILL_SOURCE
fi
if [ -z "${LING_MODEL_CHOICES:-}" ]; then
  export LING_MODEL_CHOICES='[{"model":"inclusionai/ling-3.0-flash-vl","label":"Ling-3.0-flash-VL","context_length":262144,"reasoning_levels":["none","high"]},{"model":"inclusionai/ling-3.0-flash","label":"Ling-3.0-flash","context_length":262144,"reasoning_levels":["none","high"]}]'
else
  export LING_MODEL_CHOICES
fi
# END BUILD DEFAULTS

readonly SCRIPT_VERSION="0.4.0"
readonly INSTALLER_ID="claude-code-ling-3-flash-vl"
readonly DEFAULT_MODEL="$LING_MODEL"
readonly DEFAULT_BASE_URL="$LING_MESSAGES_BASE_URL"
readonly GUI_SKILL_NAME="ling-gui-agent-skill"
readonly GUI_SKILL_AGENT="claude-code"
readonly GUI_SKILL_SOURCE="$LING_GUI_AGENT_SKILL_SOURCE"
readonly GUI_SKILL_BASE_URL="${LING_GUI_BASE_URL:-$LING_BASE_URL}"
readonly SKILLS_CLI_PACKAGE="skills@1.5.23"

model="${LING_CLAUDE_MODEL:-$DEFAULT_MODEL}"
case "$model" in
  ''|*[!a-zA-Z0-9_./:@+~-]*) printf '%s\n' 'Invalid model ID' >&2; exit 1 ;;
esac
base_url="${LING_CLAUDE_BASE_URL:-$DEFAULT_BASE_URL}"
theta_api_key=""
language="${LING_SETUP_LANG:-en}"
stage_dir=""
self_test_dir=""
backup_created="0"
skill_install_log=""
gui_skill_env_stage=""

# English source text first; Chinese is selected before writing to stdout/tty.
message() {
  if is_english; then printf '%s' "$1"; else printf '%s' "$2"; fi
}

info() {
  printf '%s\n' "$*"
}

is_english() {
  [ "$language" = "en" ]
}

die() {
  printf "$(message 'Error: %s\n' '错误：%s\n')" "$*" >&2
  exit 1
}

cleanup() {
  unset theta_api_key OPENROUTER_API_KEY

  if [ -n "$skill_install_log" ] && [ -f "$skill_install_log" ]; then
    rm -f -- "$skill_install_log"
  fi
  if [ -n "$gui_skill_env_stage" ] && [ -f "$gui_skill_env_stage" ]; then
    rm -f -- "$gui_skill_env_stage"
  fi

  if [ -n "$self_test_dir" ] && [ -d "$self_test_dir" ]; then
    rm -f -- "$self_test_dir/output.txt" "$self_test_dir/run.log"
    rmdir -- "$self_test_dir" 2>/dev/null || true
  fi

  if [ -n "$stage_dir" ] && [ -d "$stage_dir" ]; then
    rm -f -- \
      "$stage_dir/api-key" \
      "$stage_dir/manifest.txt" \
      "$stage_dir/read-api-key.sh" \
      "$stage_dir/settings.json"
    rmdir -- "$stage_dir" 2>/dev/null || true
  fi

  if [ "$backup_created" = "1" ] && [ -n "${backup_dir:-}" ] && [ -d "$backup_dir" ]; then
    rm -f -- "$backup_settings" "$manifest_file"
    rmdir -- "$backup_dir" 2>/dev/null || true
  fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

usage() {
  if is_english; then
  cat <<'EOF'
Ling-3.0-flash-VL persistent setup for Claude Code CLI (English)

Usage:
  bash claude-code-ling-3-flash-vl-setup.sh
  bash claude-code-ling-3-flash-vl-setup.sh --install
  bash claude-code-ling-3-flash-vl-setup.sh --status
  bash claude-code-ling-3-flash-vl-setup.sh --self-test
  bash claude-code-ling-3-flash-vl-setup.sh --uninstall
  bash claude-code-ling-3-flash-vl-setup.sh --uninstall --yes

After installation, running `claude` directly defaults to:
  Base URL: https://openrouter.ai/api
  Model:    inclusionai/ling-3.0-flash-vl
  Protocol: Anthropic Messages API

The installer prompts for a OpenRouter personal-token API key and stores it with
mode 0600. It never writes the token to settings.json.

The installer recommends the Ling GUI Agent Skill and installs it through
npx skills unless you answer N/n. Set LING_INSTALL_GUI_SKILL=n to skip it in
non-interactive automation. Override its Git source with
LING_GUI_AGENT_SKILL_SOURCE. After installation, the setup regenerates the
Skill-local .env from .env.example with the current token, model, and Chat
Completions Base URL, and stores it with mode 0600.

Uninstall restores the settings.json snapshot from before the first install.
It does not modify ~/.claude.json, .credentials.json, macOS Keychain, or the
saved Claude subscription login. The shared GUI Skill, its .env, and the GUI
Agent token remain installed for other Agent integrations. Claude Desktop uses
separate gateway settings.
EOF
  else
  cat <<'EOF'
Ling-3.0-flash-VL Claude Code CLI 持久配置工具

用法：
  bash claude-code-ling-3-flash-vl-setup.zh-CN.sh
  bash claude-code-ling-3-flash-vl-setup.zh-CN.sh --install
  bash claude-code-ling-3-flash-vl-setup.zh-CN.sh --status
  bash claude-code-ling-3-flash-vl-setup.zh-CN.sh --self-test
  bash claude-code-ling-3-flash-vl-setup.zh-CN.sh --uninstall
  bash claude-code-ling-3-flash-vl-setup.zh-CN.sh --uninstall --yes

安装后，直接运行 claude 会默认使用：
  Base URL: https://openrouter.ai/api
  Model:    inclusionai/ling-3.0-flash-vl
  Protocol: Anthropic Messages API

安装器将提示输入 OpenRouter 个人令牌 APIKey，并将其保存在 Claude 配置目录下、
权限为 0600。令牌不会写入 settings.json。

安装器还会推荐通过 npx skills 安装 Ling GUI Agent Skill。询问 [Y/n] 时
只有 N/n 会跳过，其他输入默认安装；自动化可设置 LING_INSTALL_GUI_SKILL=n
跳过。可使用 LING_GUI_AGENT_SKILL_SOURCE 覆盖默认 Git 仓库地址。
安装成功后会根据 Skill 的 .env.example 生成 .env，写入当前令牌、Ling 模型和
Chat Completions Base URL，权限为 0600；重复安装会重新生成这个文件。

卸载会恢复首次安装前的 settings.json，并删除 Claude 配置目录中的令牌。
不会修改 ~/.claude.json、.credentials.json、macOS Keychain，也不会退出订阅账户。
共享的 Ling GUI Agent Skill 及其 .env 会保留，.env 中仍包含 GUI Agent 令牌，
避免卸载一个模型集成影响其他 Agent。

注意：本工具仅配置 Claude Code CLI。Claude Desktop APP 使用独立配置。
Claude Code CLI 在自定义 Base URL 后会透传自定义模型 ID；Anthropic 不为非 Claude
模型路由提供官方支持，本工具属于由网关和模型提供方负责验证的兼容性方案。
EOF
  fi
}

resolve_paths() {
  local requested_dir="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"

  [ -n "$requested_dir" ] || die "$(message "CLAUDE_CONFIG_DIR must not be empty" "CLAUDE_CONFIG_DIR 不能为空")"
  [ "$requested_dir" != "/" ] || die "$(message "CLAUDE_CONFIG_DIR must not be the root directory" "CLAUDE_CONFIG_DIR 不能是根目录")"

  if [ ! -d "$requested_dir" ]; then
    mkdir -p -- "$requested_dir"
    chmod 700 "$requested_dir"
  fi
  claude_dir="$(cd -- "$requested_dir" >/dev/null 2>&1 && pwd -P)"

  settings_file="$claude_dir/settings.json"
  install_dir="$claude_dir/ling-3-flash-vl"
  backup_dir="$claude_dir/backup-ling-3-flash-vl"
  backup_settings="$backup_dir/settings.json"
  manifest_file="$backup_dir/manifest.txt"
  token_file="$install_dir/api-key"
  token_reader="$install_dir/read-api-key.sh"
  gui_skill_dir="$claude_dir/skills/$GUI_SKILL_NAME"

  [ ! -L "$settings_file" ] || die "$(message "Symlinked settings.json is not supported: $settings_file" "暂不支持符号链接形式的 settings.json：$settings_file")"
  [ ! -L "$install_dir" ] || die "$(message "The installation directory must not be a symlink: $install_dir" "安装目录不能是符号链接：$install_dir")"
  [ ! -L "$backup_dir" ] || die "$(message "The backup directory must not be a symlink: $backup_dir" "备份目录不能是符号链接：$backup_dir")"
}

resolve_json_runtime() {
  if command -v python3 >/dev/null 2>&1; then
    json_runtime="python3"
  elif command -v node >/dev/null 2>&1; then
    json_runtime="node"
  else
    die "$(message "python3 or node is required to safely merge settings.json" "需要 python3 或 node 来安全合并 settings.json")"
  fi
}

resolve_claude() {
  claude_executable="$(command -v claude 2>/dev/null || true)"
  [ -n "$claude_executable" ] \
    || die "$(message "Claude Code CLI not found. Install Claude Code first" "未找到 Claude Code CLI。请先安装 Claude Code")"
}

write_gui_skill_env() {
  local template="$gui_skill_dir/.env.example"
  local destination="$gui_skill_dir/.env"
  local line=""
  local found_base_url="0"
  local found_api_key="0"
  local found_model="0"

  [ -r "$template" ] || {
    if is_english; then
      die "Ling GUI Agent Skill is missing its configuration template: $template"
    else
      die "$(message "Ling GUI Agent Skill configuration template is missing: $template" "Ling GUI Agent Skill 缺少配置模板：$template")"
    fi
  }

  gui_skill_env_stage="$(mktemp "$gui_skill_dir/.env.ling-setup.XXXXXX")"
  umask 077
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      LING_BASE_URL=*)
        printf 'LING_BASE_URL=%s\n' "$GUI_SKILL_BASE_URL"
        found_base_url="1"
        ;;
      LING_API_KEY=*)
        printf 'LING_API_KEY=%s\n' "$theta_api_key"
        found_api_key="1"
        ;;
      LING_MODEL=*)
        printf 'LING_MODEL=%s\n' "$model"
        found_model="1"
        ;;
      *) printf '%s\n' "$line" ;;
    esac
  done < "$template" > "$gui_skill_env_stage"
  [ "$found_base_url" = "1" ] || printf 'LING_BASE_URL=%s\n' "$GUI_SKILL_BASE_URL" >> "$gui_skill_env_stage"
  [ "$found_api_key" = "1" ] || printf 'LING_API_KEY=%s\n' "$theta_api_key" >> "$gui_skill_env_stage"
  [ "$found_model" = "1" ] || printf 'LING_MODEL=%s\n' "$model" >> "$gui_skill_env_stage"
  chmod 600 "$gui_skill_env_stage"
  mv -f -- "$gui_skill_env_stage" "$destination"
  gui_skill_env_stage=""

  if is_english; then
    info "Ling GUI Agent Skill configuration: $destination (mode 0600)"
  else
    info "$(message "Ling GUI Agent Skill configuration: ${destination} (mode 0600)" "Ling GUI Agent Skill 配置：${destination}（权限 0600）")"
  fi
}

install_gui_skill() {
  local npx_executable=""

  npx_executable="$(command -v npx 2>/dev/null || true)"
  if [ -z "$npx_executable" ]; then
    if is_english; then
      die "npx was not found. Install Node.js/npm to install the Ling GUI Agent Skill"
    else
      die "$(message "npx not found. Install Node.js/npm to install the Ling GUI Agent Skill" "未找到 npx。请先安装 Node.js/npm，以便安装 Ling GUI Agent Skill")"
    fi
  fi

  skill_install_log="$(mktemp "${TMPDIR:-/tmp}/ling-gui-skill-install.XXXXXX")"
  if is_english; then
    info "Installing the Ling GUI Agent Skill through npx skills..."
  else
    info "$(message "Installing the Ling GUI Agent Skill through npx skills..." "正在通过 npx skills 安装 Ling GUI Agent Skill……")"
  fi
  if ! env -u OPENROUTER_API_KEY CLAUDE_CONFIG_DIR="$claude_dir" \
    "$npx_executable" --yes "$SKILLS_CLI_PACKAGE" add "$GUI_SKILL_SOURCE" \
      --global --agent "$GUI_SKILL_AGENT" --skill "$GUI_SKILL_NAME" --yes \
      >"$skill_install_log" 2>&1
  then
    tail -n 40 "$skill_install_log" >&2
    if is_english; then
      die "Ling GUI Agent Skill installation failed"
    else
      die "$(message "Ling GUI Agent Skill installation failed" "Ling GUI Agent Skill 安装失败")"
    fi
  fi
  if [ ! -r "$gui_skill_dir/SKILL.md" ]; then
    if is_english; then
      die "npx skills succeeded but the Skill was not found at $gui_skill_dir/SKILL.md"
    else
      die "$(message "npx skills succeeded but the Skill was not found at $gui_skill_dir/SKILL.md" "npx skills 返回成功，但未找到 Skill：$gui_skill_dir/SKILL.md")"
    fi
  fi
  write_gui_skill_env
  rm -f -- "$skill_install_log"
  skill_install_log=""
  show_gui_skill_status
}

should_install_gui_skill() {
  local answer="${LING_INSTALL_GUI_SKILL:-}"

  if [ -n "$answer" ]; then
    case "$answer" in
      n|N) return 1 ;;
      *) return 0 ;;
    esac
  fi
  if ! { [ -t 0 ] || [ -t 1 ] || [ -t 2 ]; }; then
    return 0
  fi
  [ -r /dev/tty ] || return 0
  if is_english; then
    printf 'Recommended: install the Ling GUI Agent Skill for better GUI automation. Install it? [Y/n] ' > /dev/tty
  else
    printf "$(message 'Recommended: install the Ling GUI Agent Skill for better GUI automation. Install it? [Y/n] ' '推荐安装 Ling GUI Agent Skill 以获得更好的 GUI 自动化效果。是否安装？[Y/n] ')" > /dev/tty
  fi
  IFS= read -r answer < /dev/tty || answer=""
  case "$answer" in
    n|N) return 1 ;;
    *) return 0 ;;
  esac
}

skip_gui_skill() {
  if is_english; then
    info "Ling GUI Agent Skill installation skipped by user choice."
  else
    info "$(message "Ling GUI Agent Skill installation skipped by user choice." "已按用户选择跳过 Ling GUI Agent Skill 安装。")"
  fi
}

show_gui_skill_status() {
  if [ -r "$gui_skill_dir/SKILL.md" ]; then
    if is_english; then
      info "Ling GUI Agent Skill: installed ($gui_skill_dir)"
    else
      info "$(message "Ling GUI Agent Skill: installed (${gui_skill_dir})" "Ling GUI Agent Skill：已安装（${gui_skill_dir}）")"
    fi
    if [ -r "$gui_skill_dir/.env" ]; then
      if is_english; then
        info "Ling GUI Agent Skill configuration: present (value hidden)"
      else
        info "$(message "Ling GUI Agent Skill configuration: saved (value hidden)" "Ling GUI Agent Skill 配置：已保存（不显示内容）")"
      fi
    else
      if is_english; then
        info "Ling GUI Agent Skill configuration: missing"
      else
        info "$(message "Ling GUI Agent Skill configuration: missing" "Ling GUI Agent Skill 配置：缺失")"
      fi
    fi
  else
    if is_english; then
      info "Ling GUI Agent Skill: not installed (expected: $gui_skill_dir)"
    else
      info "$(message "Ling GUI Agent Skill: not installed (expected: ${gui_skill_dir})" "Ling GUI Agent Skill：未安装（预期：${gui_skill_dir}）")"
    fi
  fi
}

sha256_file() {
  local path="$1"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  else
    die "$(message "shasum or sha256sum is required to verify configuration snapshots" "缺少 shasum 或 sha256sum，无法校验配置快照")"
  fi
}

file_mode() {
  local path="$1"
  if stat -f '%Lp' "$path" >/dev/null 2>&1; then
    stat -f '%Lp' "$path"
  else
    stat -c '%a' "$path"
  fi
}

manifest_value() {
  local key="$1"
  [ -f "$manifest_file" ] || return 1
  awk -F= -v wanted="$key" '$1 == wanted { sub(/^[^=]*=/, ""); print; exit }' "$manifest_file"
}

is_managed_install() {
  [ -f "$manifest_file" ] \
    && [ "$(manifest_value installer_id 2>/dev/null || true)" = "$INSTALLER_ID" ]
}

read_api_key() {
  theta_api_key="${OPENROUTER_API_KEY:-}"
  if [ -n "$theta_api_key" ]; then
    info "$(message "Using the OpenRouter personal token from OPENROUTER_API_KEY." "使用 OPENROUTER_API_KEY 环境变量中的 OpenRouter 个人令牌。")"
    return
  fi

  [ -r /dev/tty ] \
    || die "$(message "No interactive terminal. Set OPENROUTER_API_KEY for non-interactive installation" "当前没有交互终端。请通过 OPENROUTER_API_KEY 环境变量提供 OpenRouter 个人令牌")"

  printf "$(message 'Enter your OpenRouter personal-token API key (stored locally with mode 0600): ' '请输入 OpenRouter 个人令牌 APIKey（将以 0600 权限保存在本机）: ')" > /dev/tty
  if ! IFS= read -r -s theta_api_key < /dev/tty; then
    printf '\n' > /dev/tty
    die "$(message "Could not read the OpenRouter personal token" "读取 OpenRouter 个人令牌失败")"
  fi
  printf '\n' > /dev/tty
  [ -n "$theta_api_key" ] || die "$(message "The OpenRouter personal token cannot be empty" "OpenRouter 个人令牌不能为空")"
}

write_token_reader() {
  local destination="$1"
  cat > "$destination" <<'SH'
#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" >/dev/null 2>&1 && pwd -P)
token_file="$script_dir/api-key"

[ -r "$token_file" ] || {
  printf 'OpenRouter APIKey file is missing or unreadable: %s\n' "$token_file" >&2
  exit 1
}

IFS= read -r token < "$token_file" || [ -n "${token:-}" ]
[ -n "${token:-}" ] || {
  printf 'OpenRouter APIKey file is empty: %s\n' "$token_file" >&2
  exit 1
}

printf '%s\n' "$token"
SH
}

write_settings_python() {
  local source="$1"
  local destination="$2"

  python3 - "$source" "$destination" "$model" "$base_url" "$token_reader" <<'PY'
import json
import pathlib
import sys

source, destination, model, base_url, token_reader = sys.argv[1:]
import os
choices = json.loads(os.environ.get('LING_MODEL_CHOICES', '[]'))
source_path = pathlib.Path(source)

if source_path.is_file():
    with source_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
else:
    data = {}

if not isinstance(data, dict):
    raise SystemExit("settings.json must contain a JSON object")

env = data.get("env", {})
if not isinstance(env, dict):
    raise SystemExit("settings.json env must be a JSON object")

for key in (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_DISABLE_THINKING",
    "MAX_THINKING_TOKENS",
):
    env.pop(key, None)

env.update(
    {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_MODEL": model,
        "ANTHROPIC_DEFAULT_FABLE_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL_FORCE": "1",
        "ANTHROPIC_CUSTOM_MODEL_OPTION": model,
        "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Ling-3.0-flash-VL",
        "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": "OpenRouter Anthropic-compatible model",
    }
)

for prefix in ("ANTHROPIC_DEFAULT_FABLE_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_CUSTOM_MODEL_OPTION"):
    env[prefix + "_SUPPORTED_CAPABILITIES"] = "thinking"
data["alwaysThinkingEnabled"] = True
data["model"] = model
if choices:
    env['CLAUDE_CODE_MAX_CONTEXT_TOKENS'] = str(min(row['context_length'] for row in choices))
    for key in ('ANTHROPIC_MODEL', 'CLAUDE_CODE_SUBAGENT_MODEL', 'CLAUDE_CODE_SUBAGENT_MODEL_FORCE'):
        env.pop(key, None)
    labels = {row['model']: row['label'] for row in choices}
    env['ANTHROPIC_CUSTOM_MODEL_OPTION_NAME'] = labels.get(model, model)
    for alias, row in zip(('OPUS', 'SONNET'), choices):
        env[f'ANTHROPIC_DEFAULT_{alias}_MODEL'] = row['model']
        env[f'ANTHROPIC_DEFAULT_{alias}_MODEL_NAME'] = row['label']
    picker = data.setdefault('modelPicker', {'replaceBuiltInOptions': True})
    picker['options'] = [{'model': row['model'], 'label': row['label']} for row in choices] + [row for row in picker.get('options', []) if row['model'] not in labels]
data["apiKeyHelper"] = token_reader
data["env"] = env

with pathlib.Path(destination).open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(data, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
}

write_settings_node() {
  local source="$1"
  local destination="$2"

  node - "$source" "$destination" "$model" "$base_url" "$token_reader" <<'JS'
const fs = require("fs");
const [source, destination, model, baseUrl, tokenReader] = process.argv.slice(2);
let data = fs.existsSync(source) ? JSON.parse(fs.readFileSync(source, "utf8")) : {};
if (data === null || Array.isArray(data) || typeof data !== "object") {
  throw new Error("settings.json must contain a JSON object");
}
let env = data.env === undefined ? {} : data.env;
if (env === null || Array.isArray(env) || typeof env !== "object") {
  throw new Error("settings.json env must be a JSON object");
}
for (const key of [
  "ANTHROPIC_API_KEY",
  "ANTHROPIC_AUTH_TOKEN",
  "CLAUDE_CODE_USE_BEDROCK",
  "CLAUDE_CODE_USE_VERTEX",
  "CLAUDE_CODE_USE_FOUNDRY",
  "CLAUDE_CODE_DISABLE_THINKING",
  "MAX_THINKING_TOKENS",
]) delete env[key];
Object.assign(env, {
  ANTHROPIC_BASE_URL: baseUrl,
  ANTHROPIC_MODEL: model,
  ANTHROPIC_DEFAULT_MODEL: model,
  ANTHROPIC_DEFAULT_FABLE_MODEL: model,
  ANTHROPIC_DEFAULT_OPUS_MODEL: model,
  ANTHROPIC_DEFAULT_SONNET_MODEL: model,
  ANTHROPIC_DEFAULT_HAIKU_MODEL: model,
  CLAUDE_CODE_SUBAGENT_MODEL: model,
  CLAUDE_CODE_SUBAGENT_MODEL_FORCE: "1",
  ANTHROPIC_CUSTOM_MODEL_OPTION: model,
  ANTHROPIC_CUSTOM_MODEL_OPTION_NAME: "Ling-3.0-flash-VL",
  ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION: "OpenRouter Anthropic-compatible model",
});
for (const prefix of ["ANTHROPIC_DEFAULT_FABLE_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_CUSTOM_MODEL_OPTION"]) env[prefix + "_SUPPORTED_CAPABILITIES"] = "thinking";
data.alwaysThinkingEnabled = true;
data.model = model;
const choices = JSON.parse(process.env.LING_MODEL_CHOICES || '[]');
if (choices.length) {
  env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(Math.min(...choices.map(row => row.context_length)));
  for (const key of ['ANTHROPIC_MODEL', 'CLAUDE_CODE_SUBAGENT_MODEL', 'CLAUDE_CODE_SUBAGENT_MODEL_FORCE']) delete env[key];
  const labels = Object.fromEntries(choices.map(row => [row.model, row.label]));
  env.ANTHROPIC_CUSTOM_MODEL_OPTION_NAME = labels[model] || model;
  ['OPUS', 'SONNET'].forEach((alias, i) => {
    env[`ANTHROPIC_DEFAULT_${alias}_MODEL`] = choices[i].model;
    env[`ANTHROPIC_DEFAULT_${alias}_MODEL_NAME`] = choices[i].label;
  });
  data.modelPicker ??= {replaceBuiltInOptions: true};
  data.modelPicker.options = choices.map(({model, label}) => ({model, label})).concat((data.modelPicker.options || []).filter(row => !(row.model in labels)));
}
data.apiKeyHelper = tokenReader;
data.env = env;
fs.writeFileSync(destination, JSON.stringify(data, null, 2) + "\n", "utf8");
JS
}

write_settings() {
  local source="$1"
  local destination="$2"

  case "$json_runtime" in
    python3) write_settings_python "$source" "$destination" ;;
    node) write_settings_node "$source" "$destination" ;;
    *) die "$(message "Unknown JSON runtime: $json_runtime" "未知 JSON 运行时：$json_runtime")" ;;
  esac
}

validate_json() {
  local path="$1"
  case "$json_runtime" in
    python3)
      python3 -c 'import json, pathlib, sys; data=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")); assert isinstance(data, dict)' "$path"
      ;;
    node)
      node -e 'const fs=require("fs"); const value=JSON.parse(fs.readFileSync(process.argv[1], "utf8")); if (!value || Array.isArray(value) || typeof value !== "object") process.exit(1)' "$path"
      ;;
  esac
}

warn_shell_conflicts() {
  local found="0"
  local name
  for name in \
    ANTHROPIC_API_KEY \
    ANTHROPIC_AUTH_TOKEN \
    CLAUDE_CODE_USE_BEDROCK \
    CLAUDE_CODE_USE_VERTEX \
    CLAUDE_CODE_USE_FOUNDRY
  do
    if [ -n "${!name:-}" ]; then
      if [ "$found" = "0" ]; then
        info "$(message "Warning: the current shell contains environment variables that may change authentication routing:" "警告：当前 shell 中存在可能改变认证路由的环境变量：")"
      fi
      info "  - $name"
      found="1"
    fi
  done
  if [ "$found" = "1" ]; then
    info "$(message "Unset these shell variables while verifying. The installer does not modify shell profiles." "请在验证时取消这些 shell 变量；安装器不会修改 shell profile。")"
  fi
}

install_ling() {
  local original_settings_existed="0"
  local original_settings_mode=""
  local installed_sha=""
  local previous_installed_sha=""
  local current_sha=""
  local safety_copy=""

  resolve_json_runtime
  resolve_claude
  read_api_key

  if [ -e "$backup_dir" ] && [ ! -d "$backup_dir" ]; then
    die "$(message "Backup path exists but is not a directory: $backup_dir" "备份路径已存在但不是目录：$backup_dir")"
  fi
  if [ -d "$backup_dir" ] && ! is_managed_install; then
    die "$(message "Refusing to overwrite a backup directory not created by this installer: $backup_dir" "备份目录不是本安装器创建的，拒绝覆盖：$backup_dir")"
  fi
  if [ -e "$install_dir" ] && [ ! -d "$install_dir" ]; then
    die "$(message "Installation path exists but is not a directory: $install_dir" "安装路径已存在但不是目录：$install_dir")"
  fi
  if [ -d "$install_dir" ] && [ ! -d "$backup_dir" ]; then
    die "$(message "Refusing to overwrite an installation directory without a backup manifest: $install_dir" "发现无备份清单的安装目录，拒绝覆盖：$install_dir")"
  fi

  if should_install_gui_skill; then
    install_gui_skill
  else
    skip_gui_skill
  fi

  if is_managed_install && [ -f "$settings_file" ]; then
    previous_installed_sha="$(manifest_value installed_settings_sha256 || true)"
    current_sha="$(sha256_file "$settings_file")"
    if [ -z "$previous_installed_sha" ] || [ "$current_sha" != "$previous_installed_sha" ]; then
      safety_copy="$claude_dir/settings.before-ling3-update.$(date '+%Y%m%d-%H%M%S').json"
      cp -p -- "$settings_file" "$safety_copy"
      chmod 600 "$safety_copy"
      info "$(message "Detected settings changes since the previous install; saved a safety copy: $safety_copy" "检测到上次安装后的配置改动，已先保存：$safety_copy")"
    fi
  fi

  stage_dir="$(mktemp -d "$claude_dir/.ling3-claude-stage.XXXXXX")"
  write_token_reader "$stage_dir/read-api-key.sh"
  printf '%s' "$theta_api_key" > "$stage_dir/api-key"
  unset theta_api_key OPENROUTER_API_KEY
  write_settings "$settings_file" "$stage_dir/settings.json"
  chmod 600 "$stage_dir/api-key" "$stage_dir/settings.json"
  chmod 700 "$stage_dir/read-api-key.sh"
  validate_json "$stage_dir/settings.json" || die "$(message "Generated settings.json is invalid" "生成的 settings.json 无效")"
  installed_sha="$(sha256_file "$stage_dir/settings.json")"

  if [ ! -d "$backup_dir" ]; then
    mkdir -- "$backup_dir"
    backup_created="1"
    chmod 700 "$backup_dir"
    if [ -f "$settings_file" ]; then
      cp -p -- "$settings_file" "$backup_settings"
      original_settings_existed="1"
      original_settings_mode="$(file_mode "$settings_file")"
    fi
  else
    original_settings_existed="$(manifest_value original_settings_existed || true)"
    original_settings_mode="$(manifest_value original_settings_mode || true)"
    case "$original_settings_existed" in
      0) ;;
      1) [ -f "$backup_settings" ] || die "$(message "Original settings backup is missing: $backup_settings" "原始配置备份缺失：$backup_settings")" ;;
      *) die "$(message "Invalid original_settings_existed in the backup manifest" "备份清单中的 original_settings_existed 无效")" ;;
    esac
  fi

  {
    printf 'installer_id=%s\n' "$INSTALLER_ID"
    printf 'script_version=%s\n' "$SCRIPT_VERSION"
    printf 'installed_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'original_settings_existed=%s\n' "$original_settings_existed"
    printf 'original_settings_mode=%s\n' "$original_settings_mode"
    printf 'installed_settings_sha256=%s\n' "$installed_sha"
    printf 'model=%s\n' "$model"
    printf 'base_url=%s\n' "$base_url"
  } > "$stage_dir/manifest.txt"
  chmod 600 "$stage_dir/manifest.txt"
  mv -f -- "$stage_dir/manifest.txt" "$manifest_file"
  backup_created="0"

  mkdir -p -- "$install_dir"
  chmod 700 "$install_dir"
  mv -f -- "$stage_dir/api-key" "$token_file"
  mv -f -- "$stage_dir/read-api-key.sh" "$token_reader"
  chmod 600 "$token_file"
  chmod 700 "$token_reader"

  mv -f -- "$stage_dir/settings.json" "$settings_file"
  chmod 600 "$settings_file"

  info ""
  info "$(message "Installation complete. Running claude now defaults to ${model}." "安装完成。以后直接运行 claude，默认使用 ${model}。")"
  info "$(message "Claude settings: ${settings_file}" "Claude 配置：${settings_file}")"
  info "$(message "OpenRouter token: ${token_file} (mode 0600; not stored in settings.json)" "OpenRouter 令牌：${token_file}（权限 0600，不写入 settings.json）")"
  info "$(message "Pre-install settings backup: ${backup_dir}" "首次安装前的配置备份：${backup_dir}")"
  info "$(message "The subscription login, ~/.claude.json, .credentials.json, and macOS Keychain were not modified." "订阅登录、~/.claude.json、.credentials.json 和 macOS Keychain 均未修改。")"
  info "$(message "Claude Desktop does not use this CLI gateway configuration." "Claude Desktop APP 不读取这项 CLI 网关配置。")"
  warn_shell_conflicts
  info "$(message "Run this script with --self-test to verify the end-to-end path." "可运行本脚本 --self-test 验证完整调用链。")"
}

status_ling() {
  local recorded_model=""
  local recorded_base_url=""

  if ! is_managed_install; then
    info "$(message "Status: not installed" "状态：未安装")"
    info "$(message "Claude config directory: $claude_dir" "Claude 配置目录：$claude_dir")"
    return 1
  fi

  recorded_model="$(manifest_value model 2>/dev/null || true)"
  recorded_base_url="$(manifest_value base_url 2>/dev/null || true)"
  info "$(message "Status: installed" "状态：已安装")"
  info "$(message "Model: ${recorded_model:-unknown}" "模型：${recorded_model:-未知}")"
  info "$(message "Base URL: ${recorded_base_url:-unknown}" "Base URL：${recorded_base_url:-未知}")"
  info "$(message "Claude settings: $settings_file" "Claude 配置：$settings_file")"
  if [ -f "$token_file" ]; then
    info "$(message "OpenRouter token: saved (value hidden)" "OpenRouter 令牌：已保存（不显示内容）")"
  else
    info "$(message "OpenRouter token: missing" "OpenRouter 令牌：缺失")"
  fi
  info "$(message "Installer version: $(manifest_value script_version 2>/dev/null || printf 'unknown')" "安装版本：$(manifest_value script_version 2>/dev/null || printf '未知')")"
  show_gui_skill_status
  warn_shell_conflicts
}

confirm_uninstall() {
  local answer=""
  printf "$(message 'Restore the Claude Code CLI settings from before the first install? [y/N] ' '卸载后将恢复首次安装前的 Claude Code CLI 配置。继续？[y/N] ')" > /dev/tty
  IFS= read -r answer < /dev/tty || return 1
  case "$answer" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

uninstall_ling() {
  local assume_yes="$1"
  local original_settings_existed=""
  local original_settings_mode=""
  local installed_sha=""
  local current_sha=""
  local safety_copy=""

  is_managed_install || die "$(message "No managed Ling installation was found" "未发现由本安装器创建的 Ling 配置")"

  if [ "$assume_yes" != "1" ]; then
    [ -r /dev/tty ] || die "$(message "No interactive terminal; append --yes for automated uninstall" "当前没有交互终端；自动化卸载请追加 --yes")"
    if ! confirm_uninstall; then
      info "$(message "Uninstall cancelled." "已取消卸载。")"
      return
    fi
  fi

  original_settings_existed="$(manifest_value original_settings_existed || true)"
  original_settings_mode="$(manifest_value original_settings_mode || true)"
  installed_sha="$(manifest_value installed_settings_sha256 || true)"

  if [ -f "$settings_file" ]; then
    current_sha="$(sha256_file "$settings_file")"
    if [ -z "$installed_sha" ] || [ "$current_sha" != "$installed_sha" ]; then
      safety_copy="$claude_dir/settings.before-ling3-uninstall.$(date '+%Y%m%d-%H%M%S').json"
      cp -p -- "$settings_file" "$safety_copy"
      chmod 600 "$safety_copy"
      info "$(message "Detected post-install settings changes; saved a safety copy: $safety_copy" "检测到安装后的配置改动，已额外保存：$safety_copy")"
    fi
  fi

  case "$original_settings_existed" in
    1)
      [ -f "$backup_settings" ] || die "$(message "Original settings backup is missing: $backup_settings" "原始配置备份缺失：$backup_settings")"
      cp -p -- "$backup_settings" "$claude_dir/.settings.json.ling3-restore"
      mv -f -- "$claude_dir/.settings.json.ling3-restore" "$settings_file"
      [ -n "$original_settings_mode" ] && chmod "$original_settings_mode" "$settings_file"
      ;;
    0)
      rm -f -- "$settings_file"
      ;;
    *)
      die "$(message "Invalid original_settings_existed in the backup manifest" "备份清单中的 original_settings_existed 无效")"
      ;;
  esac

  rm -f -- "$token_file" "$token_reader"
  rmdir -- "$install_dir" 2>/dev/null \
    || die "$(message "Settings restored; installation directory contains unknown files and was preserved: $install_dir" "安装目录包含未知文件，已恢复配置但未删除目录：$install_dir")"
  rm -f -- "$backup_settings" "$manifest_file"
  rmdir -- "$backup_dir" 2>/dev/null \
    || die "$(message "Settings restored; backup directory contains unknown files and was preserved: $backup_dir" "备份目录包含未知文件，已恢复配置但未删除目录：$backup_dir")"

  info "$(message "Uninstall complete: restored the Claude Code CLI settings from before the first install." "卸载完成：已恢复首次安装前的 Claude Code CLI 配置。")"
  info "$(message "Deleted the OpenRouter token from the Claude model configuration directory." "已删除 Claude 模型配置目录中的 OpenRouter 令牌。")"
  info "$(message "The subscription login was not modified; running claude now uses the restored account configuration." "订阅登录凭证未修改；现在运行 claude 将恢复原账户配置。")"
  if [ -r "$gui_skill_dir/SKILL.md" ]; then
    if is_english; then
      info "Ling GUI Agent Skill remains installed at: $gui_skill_dir"
    else
      info "$(message "Ling GUI Agent Skill remains installed at: $gui_skill_dir" "Ling GUI Agent Skill 保留在：$gui_skill_dir")"
    fi
    if [ -r "$gui_skill_dir/.env" ]; then
      if is_english; then
        info "Its .env and GUI Agent token were retained for other Agent integrations."
      else
        info "$(message "Its .env and GUI Agent token were preserved for other integrations." "其中的 .env 和 GUI Agent 令牌已为其他 Agent 集成保留。")"
      fi
    fi
  fi
}

self_test_ling() {
  local recorded_model=""
  local self_test_result=""

  resolve_claude
  is_managed_install || die "$(message "Run --install first" "请先执行 --install")"
  [ -r "$token_file" ] || die "$(message "OpenRouter token file is missing or unreadable: $token_file" "OpenRouter 令牌文件缺失或不可读：$token_file")"
  [ -x "$token_reader" ] || die "$(message "Token reader is missing or not executable: $token_reader" "令牌读取器缺失或不可执行：$token_reader")"

  recorded_model="$(manifest_value model 2>/dev/null || true)"
  [ -n "$recorded_model" ] || die "$(message "The backup manifest is missing the model" "备份清单缺少模型信息")"
  self_test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ling-claude-installed-test.XXXXXX")"
  info "$(message "Running the Claude Code CLI end-to-end self-test with the persistent configuration..." "正在使用持久配置执行 Claude Code CLI 端到端自检……")"

  if ! CLAUDE_CONFIG_DIR="$claude_dir" "$claude_executable" \
    -p \
    --model "$recorded_model" \
    --output-format text \
    --no-session-persistence \
    --max-turns 1 \
    "Calculate 19 + 23. Output only the number; do not use tools." \
    > "$self_test_dir/output.txt" 2> "$self_test_dir/run.log"
  then
    tail -n 40 "$self_test_dir/run.log" >&2
    die "$(message "Claude Code CLI end-to-end self-test failed" "Claude Code CLI 端到端自检失败")"
  fi

  self_test_result="$(tr -d '[:space:]' < "$self_test_dir/output.txt")"
  [ "$self_test_result" = "42" ] \
    || die "$(message "Claude Code CLI returned an answer other than the expected 42 (actual: ${self_test_result})" "Claude Code CLI 已返回响应，但自检结果不是预期的 42（实际：${self_test_result}）")"

  info "$(message "Self-test passed: persistent settings -> OpenRouter Anthropic Messages API -> $recorded_model returned 42." "自检通过：持久配置 → OpenRouter Anthropic Messages API → $recorded_model 返回 42。")"
}

interactive_menu() {
  if is_english; then
  cat <<EOF
Ling-3.0-flash-VL Claude Code CLI setup v$SCRIPT_VERSION (English)

1) Install or update the Ling default model
2) Show installation status
3) Run the end-to-end self-test
9) Uninstall and restore the subscription configuration
0) Exit
EOF
  else
  cat <<EOF
Ling-3.0-flash-VL Claude Code CLI 配置工具 v$SCRIPT_VERSION

1) 安装或更新 Ling 默认模型
2) 查看安装状态
3) 执行端到端自检
9) 卸载并恢复 Claude Code 订阅账户配置
0) 退出
EOF
  fi
  printf "$(message 'Select [0/1/2/3/9]: ' '请选择 [0/1/2/3/9]: ')" > /dev/tty
  IFS= read -r choice < /dev/tty || die "$(message "Could not read the menu selection" "读取菜单选择失败")"

  case "$choice" in
    1) install_ling ;;
    2) status_ling || true ;;
    3) self_test_ling ;;
    9) uninstall_ling 0 ;;
    0) info "$(message "Exited." "已退出。")" ;;
    *) die "$(message "Invalid selection: $choice" "无效选择：$choice")" ;;
  esac
}

main() {
  umask 077

  case "${1:-}" in
    -h|--help)
      usage
      return
      ;;
  esac

  resolve_paths

  case "${1:-}" in
    "")
      [ -r /dev/tty ] || die "$(message "No interactive terminal; use --install, --status, or --uninstall --yes" "当前没有交互终端，请使用 --install、--status 或 --uninstall --yes")"
      interactive_menu
      ;;
    --install)
      [ "$#" -eq 1 ] || die "$(message "--install does not accept additional arguments" "--install 不接受额外参数")"
      install_ling
      ;;
    --status)
      [ "$#" -eq 1 ] || die "$(message "--status does not accept additional arguments" "--status 不接受额外参数")"
      status_ling
      ;;
    --self-test)
      [ "$#" -eq 1 ] || die "$(message "--self-test does not accept additional arguments" "--self-test 不接受额外参数")"
      self_test_ling
      ;;
    --uninstall)
      if [ "$#" -eq 1 ]; then
        uninstall_ling 0
      elif [ "$#" -eq 2 ] && [ "$2" = "--yes" ]; then
        uninstall_ling 1
      else
        die "$(message "Uninstall usage: --uninstall [--yes]" "卸载用法：--uninstall [--yes]")"
      fi
      ;;
    *)
      usage >&2
      die "$(message "Unknown argument: $1" "未知参数：$1")"
      ;;
  esac
}

main "$@"
