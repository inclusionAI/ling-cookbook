#!/usr/bin/env bash
# Bundle version: 0.4.0
# Installer version: 0.4.0
# Provider: openrouter
# Entrypoint: codex-ling-3-flash-vl-openrouter-setup.zh-CN.sh
# Model: inclusionai/ling-3.0-flash-vl
# Payload SHA256: be3398365fc444797089d35b1f9ac246b9d0b395060d9e6f8eeddd6ba549998c
set -euo pipefail
if [ "${1:-}" = "--version" ]; then
  [ "$#" -eq 1 ] || { printf '%s\n' '--version accepts no extra arguments' >&2; exit 2; }
  cat <<'LING_BUNDLE_VERSION'
Bundle version: 0.4.0
Installer version: 0.4.0
Provider: openrouter
Entrypoint: codex-ling-3-flash-vl-openrouter-setup.zh-CN.sh
Model: inclusionai/ling-3.0-flash-vl
Payload SHA256: be3398365fc444797089d35b1f9ac246b9d0b395060d9e6f8eeddd6ba549998c
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
readonly INSTALLER_ID="codex-ling-3-flash-vl"
readonly PROVIDER_ID="antchat_ling3"
readonly MODEL="${LING_CODEX_MODEL:-$LING_MODEL}"
case "$MODEL" in
  ''|*[!a-zA-Z0-9_./:@+~-]*) printf '%s\n' 'Invalid model ID' >&2; exit 1 ;;
esac
readonly BASE_URL="${LING_CODEX_BASE_URL:-$LING_BASE_URL}"
readonly GUI_SKILL_NAME="ling-gui-agent-skill"
readonly GUI_SKILL_AGENT="codex"
readonly GUI_SKILL_SOURCE="$LING_GUI_AGENT_SKILL_SOURCE"
readonly GUI_SKILL_BASE_URL="${LING_GUI_BASE_URL:-$LING_BASE_URL}"
readonly SKILLS_CLI_PACKAGE="skills@1.5.23"

theta_api_key=""
language="${LING_SETUP_LANG:-en}"
self_test_dir=""
stage_dir=""
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
    rm -f -- "$self_test_dir/final.txt" "$self_test_dir/run.log"
    rmdir -- "$self_test_dir" 2>/dev/null || true
  fi

  if [ -n "$stage_dir" ] && [ -d "$stage_dir" ]; then
    rm -f -- \
      "$stage_dir/api-key" \
      "$stage_dir/config.toml" \
      "$stage_dir/manifest.txt" \
      "$stage_dir/models.json" \
      "$stage_dir/read-api-key.sh"
    rmdir -- "$stage_dir" 2>/dev/null || true
  fi

  if [ "$backup_created" = "1" ] && [ -n "${backup_dir:-}" ] && [ -d "$backup_dir" ]; then
    rm -f -- "$backup_config" "$manifest_file"
    rmdir -- "$backup_dir" 2>/dev/null || true
  fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

usage() {
  if is_english; then
  cat <<'EOF'
Ling-3.0-flash-VL persistent setup for Codex (English)

Usage:
  bash codex-ling-3-flash-vl-setup.sh
  bash codex-ling-3-flash-vl-setup.sh --install
  bash codex-ling-3-flash-vl-setup.sh --status
  bash codex-ling-3-flash-vl-setup.sh --self-test
  bash codex-ling-3-flash-vl-setup.sh --uninstall
  bash codex-ling-3-flash-vl-setup.sh --uninstall --yes

After installation, running `codex` directly defaults to:
  Base URL: https://openrouter.ai/api/v1
  Model:    inclusionai/ling-3.0-flash-vl
  Protocol: OpenAI Responses API

The installer prompts for a OpenRouter personal-token API key and stores it with
mode 0600. It also recommends the Ling GUI Agent Skill and installs it through
npx skills unless you answer N/n. Set LING_INSTALL_GUI_SKILL=n to skip it in
non-interactive automation, and override its Git source with
LING_GUI_AGENT_SKILL_SOURCE. After installation, the setup regenerates the
Skill-local .env from .env.example with the current token, model, and Chat
Completions Base URL, and stores it with mode 0600.

Uninstall restores config.toml from before the first install and does not
delete auth.json or sign out of the Codex subscription account. The shared
Ling GUI Agent Skill, its .env, and the GUI Agent token remain installed for
other Agent integrations.
EOF
  else
  cat <<'EOF'
Ling-3.0-flash-VL Codex 持久配置工具

用法：
  bash codex-ling-3-flash-vl-setup.zh-CN.sh
  bash codex-ling-3-flash-vl-setup.zh-CN.sh --install
  bash codex-ling-3-flash-vl-setup.zh-CN.sh --status
  bash codex-ling-3-flash-vl-setup.zh-CN.sh --self-test
  bash codex-ling-3-flash-vl-setup.zh-CN.sh --uninstall
  bash codex-ling-3-flash-vl-setup.zh-CN.sh --uninstall --yes

安装后，直接运行 codex 会默认使用：
  Base URL: https://openrouter.ai/api/v1
  Model:    inclusionai/ling-3.0-flash-vl
  Protocol: OpenAI Responses API

卸载会恢复首次安装前的 config.toml，并删除 Codex 模型配置目录中的 OpenRouter 令牌。
不会删除 auth.json，也不会主动退出 Codex 订阅账户。

安装器还会推荐通过 npx skills 安装 Ling GUI Agent Skill。询问 [Y/n] 时
只有 N/n 会跳过，其他输入默认安装；自动化可设置 LING_INSTALL_GUI_SKILL=n
跳过。可使用 LING_GUI_AGENT_SKILL_SOURCE 覆盖默认 Git 仓库地址。
安装成功后会根据 Skill 的 .env.example 生成 .env，写入当前令牌、Ling 模型和
Chat Completions Base URL，权限为 0600；重复安装会重新生成这个文件。
卸载时保留这个共享 Skill 及其 .env；.env 中仍包含 GUI Agent 令牌。

注意：用户级 config.toml 可能被 Codex CLI、Desktop 和 IDE 共享。
EOF
  fi
}

resolve_paths() {
  local requested_home="${CODEX_HOME:-$HOME/.codex}"

  [ -n "${HOME:-}" ] || die "$(message "HOME must not be empty" "HOME 不能为空")"
  [ "$HOME" != "/" ] || die "$(message "HOME must not be the root directory" "HOME 不能是根目录")"
  [ -n "$requested_home" ] || die "$(message "CODEX_HOME must not be empty" "CODEX_HOME 不能为空")"
  [ "$requested_home" != "/" ] || die "$(message "CODEX_HOME must not be the root directory" "CODEX_HOME 不能是根目录")"

  if [ ! -d "$requested_home" ]; then
    mkdir -p -- "$requested_home"
    chmod 700 "$requested_home"
  fi
  codex_home="$(cd -- "$requested_home" >/dev/null 2>&1 && pwd -P)"

  config_file="$codex_home/config.toml"
  install_dir="$codex_home/ling-3-flash-vl"
  backup_dir="$codex_home/backup-ling-3-flash-vl"
  backup_config="$backup_dir/config.toml"
  manifest_file="$backup_dir/manifest.txt"
  catalog_file="$install_dir/models.json"
  token_file="$install_dir/api-key"
  token_reader="$install_dir/read-api-key.sh"
  gui_skill_dir="$HOME/.agents/skills/$GUI_SKILL_NAME"

  [ ! -L "$config_file" ] || die "$(message "Symlinked config.toml is not supported: $config_file" "暂不支持符号链接形式的 config.toml：$config_file")"
  [ ! -L "$install_dir" ] || die "$(message "The installation directory must not be a symlink: $install_dir" "安装目录不能是符号链接：$install_dir")"
  [ ! -L "$backup_dir" ] || die "$(message "The backup directory must not be a symlink: $backup_dir" "备份目录不能是符号链接：$backup_dir")"
}

resolve_codex() {
  codex_executable="$(command -v codex 2>/dev/null || true)"
  [ -n "$codex_executable" ] \
    || die "$(message "Codex CLI not found. Install @openai/codex first" "未找到 Codex CLI。请先安装 @openai/codex")"
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
        printf 'LING_MODEL=%s\n' "$MODEL"
        found_model="1"
        ;;
      *) printf '%s\n' "$line" ;;
    esac
  done < "$template" > "$gui_skill_env_stage"
  [ "$found_base_url" = "1" ] || printf 'LING_BASE_URL=%s\n' "$GUI_SKILL_BASE_URL" >> "$gui_skill_env_stage"
  [ "$found_api_key" = "1" ] || printf 'LING_API_KEY=%s\n' "$theta_api_key" >> "$gui_skill_env_stage"
  [ "$found_model" = "1" ] || printf 'LING_MODEL=%s\n' "$MODEL" >> "$gui_skill_env_stage"
  chmod 600 "$gui_skill_env_stage"
  mv -f -- "$gui_skill_env_stage" "$destination"
  gui_skill_env_stage=""

  if is_english; then
    info "Ling GUI Agent Skill configuration: $destination"
  else
    info "$(message "Ling GUI Agent Skill configuration: ${destination}" "Ling GUI Agent Skill 配置：${destination}")"
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
  if ! env -u OPENROUTER_API_KEY CODEX_HOME="$codex_home" \
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

toml_quote() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  printf '"%s"' "$value"
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

  printf "$(message 'Enter your OpenRouter personal-token API key (stored locally): ' '请输入 OpenRouter 个人令牌 APIKey（将保存在本机）: ')" > /dev/tty
  if ! IFS= read -r -s theta_api_key < /dev/tty; then
    printf '\n' > /dev/tty
    die "$(message "Could not read the OpenRouter personal token" "读取 OpenRouter 个人令牌失败")"
  fi
  printf '\n' > /dev/tty
  [ -n "$theta_api_key" ] || die "$(message "The OpenRouter personal token cannot be empty" "OpenRouter 个人令牌不能为空")"
}

write_catalog() {
  local destination="$1"
  cat > "$destination" <<JSON
{
  "models": [
    {
      "slug": "$MODEL",
      "display_name": "Ling-3.0-flash-VL",
      "description": "OpenRouter-hosted Ling-3.0-flash-VL model for Codex integration testing.",
      "base_instructions": "You are Codex, a coding agent working with the user in the current repository. Follow the user's request, all developer instructions, and applicable AGENTS.md files. Use tools when needed, preserve unrelated user changes, never expose credentials, verify completed work, and report results concisely.",
      "prefer_websockets": false,
      "support_verbosity": false,
      "apply_patch_tool_type": "freeform",
      "web_search_tool_type": "text",
      "input_modalities": ["text"],
      "supports_image_detail_original": false,
      "truncation_policy": {"mode": "tokens", "limit": 10000},
      "supports_parallel_tool_calls": false,
      "tool_mode": null,
      "use_responses_lite": false,
      "include_skills_usage_instructions": false,
      "auto_review_model_override": null,
      "context_window": 131072,
      "max_context_window": 131072,
      "effective_context_window_percent": 90,
      "auto_compact_token_limit": null,
      "reasoning_summary_format": "experimental",
      "default_reasoning_summary": "none",
      "default_reasoning_level": "high",
      "supported_reasoning_levels": [
        {"effort": "none", "description": "No thinking"},
        {"effort": "high", "description": "Validated default for the initial OpenRouter integration"}
      ],
      "shell_type": "shell_command",
      "visibility": "list",
      "minimal_client_version": "0.148.0",
      "supported_in_api": true,
      "availability_nux": null,
      "upgrade": null,
      "priority": 1,
      "experimental_supported_tools": [],
      "supports_search_tool": false,
      "default_service_tier": null,
      "supports_reasoning_summaries": false
    }
  ]
}
JSON
  if [ "${LING_MODEL_CHOICES:-[]}" != '[]' ]; then
    python3 - "$destination" "$MODEL" <<'PY'
import copy, json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
template = data['models'][0]
choices = json.loads(os.environ['LING_MODEL_CHOICES'])
labels = {row['model']: row['label'] for row in choices}
models = list(dict.fromkeys([sys.argv[2], *labels]))
data['models'] = []
for priority, model in enumerate(models, 1):
    row = copy.deepcopy(template)
    row.update(slug=model, display_name=labels.get(model, model),
               description=labels.get(model, model), priority=priority)
    context = next((choice['context_length'] for choice in choices if choice['model'] == model), None)
    if context is not None:
        row.update(context_window=context, max_context_window=context)
    levels = next((choice.get('reasoning_levels') for choice in choices if choice['model'] == model), None)
    if levels is not None:
        row['supported_reasoning_levels'] = [
            {'effort': level, 'description': 'No thinking' if level == 'none' else 'Thinking enabled'}
            for level in levels]
    data['models'].append(row)
path.write_text(json.dumps(data, indent=2) + '\n')
PY
  fi
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

write_config() {
  local source="$1"
  local destination="$2"

  {
    printf 'model = %s\n' "$(toml_quote "$MODEL")"
    printf 'model_provider = %s\n' "$(toml_quote "$PROVIDER_ID")"
    printf 'model_reasoning_effort = "high"\n'
    printf 'model_reasoning_summary = "none"\n'
    printf 'model_catalog_json = %s\n\n' "$(toml_quote "$catalog_file")"

    if [ -f "$source" ]; then
      awk '
        BEGIN { top_level = 1; skip_provider = 0 }

        /^[[:space:]]*\[\[?/ {
          header = $0
          sub(/[[:space:]]*#.*/, "", header)
          gsub(/[[:space:]]/, "", header)
          top_level = 0
          if (header == "[model_providers.antchat_ling3]" ||
              index(header, "[model_providers.antchat_ling3.") == 1) {
            skip_provider = 1
            next
          }
          skip_provider = 0
        }

        skip_provider { next }

        top_level && /^[[:space:]]*(model|model_provider|model_reasoning_effort|model_reasoning_summary|model_catalog_json)[[:space:]]*=/ {
          next
        }

        { print }
      ' "$source"
    fi

    printf '\n[model_providers.%s]\n' "$PROVIDER_ID"
    printf 'name = %s\n' "$(toml_quote "$LING_CODEX_PROVIDER_NAME")"
    printf 'base_url = %s\n' "$(toml_quote "$BASE_URL")"
    printf 'wire_api = "responses"\n'
    printf 'supports_websockets = false\n\n'
    printf '[model_providers.%s.auth]\n' "$PROVIDER_ID"
    printf 'command = %s\n' "$(toml_quote "$token_reader")"
    printf 'timeout_ms = 5000\n'
    printf 'refresh_interval_ms = 0\n'
  } > "$destination"
}

validate_staged_files() {
  if command -v python3 >/dev/null 2>&1 \
    && python3 -c 'import tomllib' >/dev/null 2>&1; then
    python3 -c 'import pathlib, sys, tomllib; tomllib.loads(pathlib.Path(sys.argv[1]).read_text())' \
      "$stage_dir/config.toml" \
      || die "$(message "Generated config.toml is not valid TOML" "生成的 config.toml 不是有效 TOML")"
  fi

  if command -v jq >/dev/null 2>&1; then
    jq -e --arg model "$MODEL" \
      '.models | type == "array" and any(.slug == $model)' \
      "$stage_dir/models.json" >/dev/null \
      || die "$(message "Generated models.json is invalid" "生成的 models.json 无效")"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import json, pathlib, sys; data=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert any(item.get("slug") == sys.argv[2] for item in data["models"])' \
      "$stage_dir/models.json" "$MODEL" \
      || die "$(message "Generated models.json is invalid" "生成的 models.json 无效")"
  else
    grep -Fq "\"slug\": \"$MODEL\"" "$stage_dir/models.json" \
      || die "$(message "Generated models.json does not contain the target model" "生成的 models.json 缺少目标模型")"
  fi
}

install_ling() {
  local original_config_existed="0"
  local installed_sha=""
  local previous_installed_sha=""
  local current_sha=""
  local safety_copy=""

  resolve_codex
  if is_english; then
    info "This will configure Codex to use Ling ($MODEL) through $BASE_URL and store your API key locally."
    info "Existing managed configuration will be backed up for restoration. Backup: $backup_dir"
    info "To restore: run this same script without arguments and choose 9, or run it with --uninstall."
    info "Press Ctrl+C now to cancel."
  else
    info "即将配置 Codex 使用 Ling（${MODEL}），服务地址为 ${BASE_URL}，并将 API Key 保存在本地。"
    info "将备份所管理的原配置以便恢复。备份位置：$backup_dir"
    info "恢复方法：不带参数运行同一个脚本并选择 9，或使用 --uninstall。"
    info "现在可按 Ctrl+C 取消。"
  fi
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

  if is_managed_install && [ -f "$config_file" ]; then
    previous_installed_sha="$(manifest_value installed_config_sha256 || true)"
    current_sha="$(sha256_file "$config_file")"
    if [ -z "$previous_installed_sha" ] || [ "$current_sha" != "$previous_installed_sha" ]; then
      safety_copy="$codex_home/config.before-ling3-update.$(date '+%Y%m%d-%H%M%S').toml"
      cp -p -- "$config_file" "$safety_copy"
      chmod 600 "$safety_copy"
      info "$(message "Detected settings changes since the previous install; saved a safety copy: $safety_copy" "检测到上次安装后的配置改动，已先保存：$safety_copy")"
    fi
  fi

  stage_dir="$(mktemp -d "$codex_home/.ling3-stage.XXXXXX")"
  write_catalog "$stage_dir/models.json"
  write_token_reader "$stage_dir/read-api-key.sh"
  printf '%s' "$theta_api_key" > "$stage_dir/api-key"
  unset theta_api_key OPENROUTER_API_KEY
  write_config "$config_file" "$stage_dir/config.toml"
  chmod 600 "$stage_dir/api-key" "$stage_dir/config.toml" "$stage_dir/models.json"
  chmod 700 "$stage_dir/read-api-key.sh"
  validate_staged_files
  installed_sha="$(sha256_file "$stage_dir/config.toml")"

  if [ ! -d "$backup_dir" ]; then
    mkdir -- "$backup_dir"
    backup_created="1"
    chmod 700 "$backup_dir"
    if [ -f "$config_file" ]; then
      cp -p -- "$config_file" "$backup_config"
      original_config_existed="1"
    fi
  else
    original_config_existed="$(manifest_value original_config_existed || true)"
    case "$original_config_existed" in
      0) ;;
      1) [ -f "$backup_config" ] || die "$(message "Original settings backup is missing: $backup_config" "原始配置备份缺失：$backup_config")" ;;
      *) die "$(message "Invalid original_config_existed in the backup manifest" "备份清单中的 original_config_existed 无效")" ;;
    esac
  fi

  {
    printf 'installer_id=%s\n' "$INSTALLER_ID"
    printf 'script_version=%s\n' "$SCRIPT_VERSION"
    printf 'installed_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'original_config_existed=%s\n' "$original_config_existed"
    printf 'installed_config_sha256=%s\n' "$installed_sha"
    printf 'model=%s\n' "$MODEL"
    printf 'base_url=%s\n' "$BASE_URL"
  } > "$stage_dir/manifest.txt"
  chmod 600 "$stage_dir/manifest.txt"
  mv -f -- "$stage_dir/manifest.txt" "$manifest_file"
  backup_created="0"

  mkdir -p -- "$install_dir"
  chmod 700 "$install_dir"
  mv -f -- "$stage_dir/api-key" "$token_file"
  mv -f -- "$stage_dir/models.json" "$catalog_file"
  mv -f -- "$stage_dir/read-api-key.sh" "$token_reader"
  chmod 600 "$token_file" "$catalog_file"
  chmod 700 "$token_reader"

  mv -f -- "$stage_dir/config.toml" "$config_file"
  chmod 600 "$config_file"

  info ""
  info "$(message "Installation complete. Running codex now defaults to ${MODEL}." "安装完成。以后直接运行 codex，默认使用 ${MODEL}。")"
  info "$(message "OpenRouter token: ${token_file}" "OpenRouter 令牌：${token_file}")"
  info "$(message "Original settings backup: $backup_dir" "原始配置：$backup_dir")"
  info "$(message "The Codex subscription credential in auth.json was not modified." "Codex 订阅登录凭证 auth.json 未被修改。")"
  info "$(message "Note: clients sharing $codex_home may all read this configuration, including Codex CLI, Desktop, and IDE integrations." "注意：共享 $codex_home 的 Codex CLI、Desktop 和 IDE 都可能读取这份配置。")"
  info "$(message "Run this script with --self-test to verify the end-to-end path." "可运行本脚本 --self-test 验证完整调用链。")"
}

status_ling() {
  if ! is_managed_install; then
    info "$(message "Status: not installed" "状态：未安装")"
    info "$(message "Codex config directory: $codex_home" "Codex 配置目录：$codex_home")"
    return 1
  fi

  info "$(message "Status: installed" "状态：已安装")"
  info "$(message "Model: $MODEL" "模型：$MODEL")"
  info "$(message "Base URL: $BASE_URL" "Base URL：$BASE_URL")"
  info "$(message "Codex settings: $config_file" "Codex 配置：$config_file")"
  info "$(message "Model catalog: $catalog_file" "模型目录：$catalog_file")"
  if [ -f "$token_file" ]; then
    info "$(message "OpenRouter token: saved (value hidden)" "OpenRouter 令牌：已保存（不显示内容）")"
  else
    info "$(message "OpenRouter token: missing" "OpenRouter 令牌：缺失")"
  fi
  info "$(message "Installer version: $(manifest_value script_version 2>/dev/null || printf 'unknown')" "安装版本：$(manifest_value script_version 2>/dev/null || printf '未知')")"
  show_gui_skill_status
}

confirm_uninstall() {
  local answer=""
  printf "$(message 'Restore the Codex settings from before the first install? [y/N] ' '卸载后将恢复首次安装前的 Codex 配置。继续？[y/N] ')" > /dev/tty
  IFS= read -r answer < /dev/tty || return 1
  case "$answer" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

uninstall_ling() {
  local assume_yes="$1"
  local original_config_existed=""
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

  original_config_existed="$(manifest_value original_config_existed || true)"
  installed_sha="$(manifest_value installed_config_sha256 || true)"

  if [ -f "$config_file" ]; then
    current_sha="$(sha256_file "$config_file")"
    if [ -z "$installed_sha" ] || [ "$current_sha" != "$installed_sha" ]; then
      safety_copy="$codex_home/config.before-ling3-uninstall.$(date '+%Y%m%d-%H%M%S').toml"
      cp -p -- "$config_file" "$safety_copy"
      chmod 600 "$safety_copy"
      info "$(message "Detected post-install settings changes; saved a safety copy: $safety_copy" "检测到安装后的配置改动，已额外保存：$safety_copy")"
    fi
  fi

  case "$original_config_existed" in
    1)
      [ -f "$backup_config" ] || die "$(message "Original settings backup is missing: $backup_config" "原始配置备份缺失：$backup_config")"
      cp -p -- "$backup_config" "$codex_home/.config.toml.ling3-restore"
      mv -f -- "$codex_home/.config.toml.ling3-restore" "$config_file"
      ;;
    0)
      rm -f -- "$config_file"
      ;;
    *)
      die "$(message "Invalid original_config_existed in the backup manifest" "备份清单中的 original_config_existed 无效")"
      ;;
  esac

  rm -f -- "$token_file" "$catalog_file" "$token_reader"
  rmdir -- "$install_dir" 2>/dev/null \
    || die "$(message "Settings restored; installation directory contains unknown files and was preserved: $install_dir" "安装目录包含未知文件，已恢复配置但未删除目录：$install_dir")"
  rm -f -- "$backup_config" "$manifest_file"
  rmdir -- "$backup_dir" 2>/dev/null \
    || die "$(message "Settings restored; backup directory contains unknown files and was preserved: $backup_dir" "备份目录包含未知文件，已恢复配置但未删除目录：$backup_dir")"

  info "$(message "Uninstall complete: restored the Codex settings from before the first install." "卸载完成：已恢复首次安装前的 Codex 配置。")"
  info "$(message "Deleted the OpenRouter token from the Codex model configuration directory." "已删除 Codex 模型配置目录中的 OpenRouter 令牌。")"
  info "$(message "The Codex subscription credential in auth.json was not modified; running codex now uses the restored account configuration." "Codex 订阅登录凭证 auth.json 未被修改；现在运行 codex 将恢复原账户配置。")"
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
  resolve_codex
  is_managed_install || die "$(message "Run --install first" "请先执行 --install")"
  [ -r "$token_file" ] || die "$(message "OpenRouter token file is missing or unreadable: $token_file" "OpenRouter 令牌文件缺失或不可读：$token_file")"
  [ -x "$token_reader" ] || die "$(message "Token reader is missing or not executable: $token_reader" "令牌读取器缺失或不可执行：$token_reader")"
  [ -r "$catalog_file" ] || die "$(message "Model catalog is missing or unreadable: $catalog_file" "模型目录缺失或不可读：$catalog_file")"

  self_test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ling-codex-installed-test.XXXXXX")"
  info "$(message "Running the Codex end-to-end self-test with the persistent configuration..." "正在使用持久配置执行 Codex 端到端自检……")"

  if ! CODEX_HOME="$codex_home" "$codex_executable" \
    --ask-for-approval never \
    --sandbox read-only \
    exec \
    --ignore-rules \
    --ephemeral \
    --color never \
    --output-last-message "$self_test_dir/final.txt" \
    "Calculate 19 + 23. Output only the number; do not use tools." \
    > "$self_test_dir/run.log" 2>&1
  then
    tail -n 40 "$self_test_dir/run.log" >&2
    die "$(message "Codex end-to-end self-test failed" "Codex 端到端自检失败")"
  fi

  self_test_result="$(tr -d '[:space:]' < "$self_test_dir/final.txt")"
  [ "$self_test_result" = "42" ] \
    || die "$(message "Codex returned an answer other than the expected 42 (actual: ${self_test_result})" "Codex 已返回响应，但自检结果不是预期的 42（实际：${self_test_result}）")"

  info "$(message "Self-test passed: persistent settings -> OpenRouter Responses API -> $MODEL returned 42." "自检通过：持久配置 → OpenRouter Responses API → $MODEL 返回 42。")"
}

interactive_menu() {
  if is_english; then
  cat <<EOF
Ling-3.0-flash-VL Codex setup v$SCRIPT_VERSION (English)

1) Install or update the Ling default model
2) Show installation status
3) Run the end-to-end self-test
9) Uninstall and restore the subscription configuration
0) Exit
EOF
  else
  cat <<EOF
Ling-3.0-flash-VL Codex 配置工具 v$SCRIPT_VERSION

1) 安装或更新 Ling 默认模型
2) 查看安装状态
3) 执行端到端自检
9) 卸载并恢复 Codex 订阅账户配置
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
