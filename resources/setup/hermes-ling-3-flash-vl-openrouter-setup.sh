#!/usr/bin/env bash
# Bundle version: 0.4.0
# Installer version: 0.4.0
# Provider: openrouter
# Entrypoint: hermes-ling-3-flash-vl-openrouter-setup.sh
# Model: inclusionai/ling-3.0-flash-vl
# Payload SHA256: e4599d8a85c7f9b29a6b641547586936f32923f6509f750e3695e2ff81f6c9a7
set -euo pipefail
if [ "${1:-}" = "--version" ]; then
  [ "$#" -eq 1 ] || { printf '%s\n' '--version accepts no extra arguments' >&2; exit 2; }
  cat <<'LING_BUNDLE_VERSION'
Bundle version: 0.4.0
Installer version: 0.4.0
Provider: openrouter
Entrypoint: hermes-ling-3-flash-vl-openrouter-setup.sh
Model: inclusionai/ling-3.0-flash-vl
Payload SHA256: e4599d8a85c7f9b29a6b641547586936f32923f6509f750e3695e2ff81f6c9a7
LING_BUNDLE_VERSION
  exit 0
fi
# BEGIN INSTALLER
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
readonly INSTALLER_ID="hermes-ling-3-flash-vl"
readonly PROVIDER_ID="antchat-ling3"
readonly MODEL="${LING_HERMES_MODEL:-$LING_MODEL}"
case "$MODEL" in
  ''|*[!a-zA-Z0-9_./:@+~-]*) printf '%s\n' 'Invalid model ID' >&2; exit 1 ;;
esac
readonly BASE_URL="${LING_HERMES_BASE_URL:-$LING_BASE_URL}"
readonly TOKEN_ENV="ANTCHAT_LING3_API_KEY"
readonly GUI_SKILL_NAME="ling-gui-agent-skill"
readonly GUI_SKILL_AGENT="hermes-agent"
readonly GUI_SKILL_SOURCE="$LING_GUI_AGENT_SKILL_SOURCE"
readonly GUI_SKILL_BASE_URL="${LING_GUI_BASE_URL:-$LING_BASE_URL}"
readonly SKILLS_CLI_PACKAGE="skills@1.5.23"

language="${LING_SETUP_LANG:-en}"
theta_api_key="${OPENROUTER_API_KEY:-}"
hermes_home=""
config_file=""
env_file=""
backup_dir=""
manifest_file=""
gui_skill_dir=""
skill_install_log=""
gui_skill_env_stage=""

is_english() {
  [ "$language" = "en" ]
}

info() {
  printf '%s\n' "$*"
}

die() {
  if is_english; then
    printf 'Error: %s\n' "$*" >&2
  else
    printf '错误：%s\n' "$*" >&2
  fi
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
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

usage() {
  if is_english; then
    cat <<'EOF'
Ling-3.0-flash-VL persistent setup for Hermes Agent (English)

Usage:
  bash hermes-ling-3-flash-vl-setup.sh
  bash hermes-ling-3-flash-vl-setup.sh --install
  bash hermes-ling-3-flash-vl-setup.sh --status
  bash hermes-ling-3-flash-vl-setup.sh --self-test
  bash hermes-ling-3-flash-vl-setup.sh --uninstall [--yes]

The installer adds a named OpenRouter provider without replacing other providers,
then makes inclusionai/ling-3.0-flash-vl the default model for new Hermes
sessions. It stores the OpenRouter personal token in ~/.hermes/.env with mode 0600.
It also recommends the Ling GUI Agent Skill through npx skills. At [Y/n], only
N/n skips it; every other answer installs it. Set LING_INSTALL_GUI_SKILL=n to
skip in automation or override the Git source with LING_GUI_AGENT_SKILL_SOURCE.
After installation, setup regenerates the Skill-local .env from .env.example
with the current token, model, and Chat Completions Base URL, using mode 0600.

Uninstall removes only the managed Hermes provider/token and restores the
previous default provider/model. Other Hermes configuration, aliases, fallback
models, OAuth credentials, memories, skills, and sessions are preserved. The
shared Ling GUI Agent Skill, its .env, and the GUI Agent token remain installed.
EOF
  else
    cat <<'EOF'
Ling-3.0-flash-VL Hermes Agent 持久配置工具

用法：
  bash hermes-ling-3-flash-vl-setup.zh-CN.sh
  bash hermes-ling-3-flash-vl-setup.zh-CN.sh --install
  bash hermes-ling-3-flash-vl-setup.zh-CN.sh --status
  bash hermes-ling-3-flash-vl-setup.zh-CN.sh --self-test
  bash hermes-ling-3-flash-vl-setup.zh-CN.sh --uninstall [--yes]

安装器会新增一个 OpenRouter 命名 provider，不会替换其他 provider；随后把
inclusionai/ling-3.0-flash-vl 设为新 Hermes 会话的默认模型。OpenRouter 个人
令牌保存在 ~/.hermes/.env，权限为 0600。

安装器还会推荐通过 npx skills 安装 Ling GUI Agent Skill。询问 [Y/n] 时
只有 N/n 会跳过，其他输入默认安装；自动化可设置 LING_INSTALL_GUI_SKILL=n
跳过。需要时可使用 LING_GUI_AGENT_SKILL_SOURCE 覆盖默认 Git 仓库地址。
安装成功后会根据 Skill 的 .env.example 生成 .env，写入当前令牌、Ling 模型和
Chat Completions Base URL，权限为 0600；重复安装会重新生成这个文件。

卸载只移除安装器管理的 Hermes provider/令牌，并恢复此前的默认 provider/模型。
其他 Hermes 配置、别名、fallback、OAuth 登录、记忆、Skill 和会话均保留，
共享的 Ling GUI Agent Skill 及其 .env 也不会删除，GUI Agent 令牌仍会保留。
EOF
  fi
}

resolve_paths() {
  [ -n "${HOME:-}" ] || die "HOME is empty"
  [ "$HOME" != "/" ] || die "HOME cannot be the filesystem root"

  hermes_home="$HOME/.hermes"
  mkdir -p -- "$hermes_home"
  chmod 700 "$hermes_home"
  hermes_home="$(cd -- "$hermes_home" >/dev/null 2>&1 && pwd -P)"
  config_file="$hermes_home/config.yaml"
  env_file="$hermes_home/.env"
  backup_dir="$hermes_home/backup-ling-3-flash-vl"
  manifest_file="$backup_dir/manifest.txt"
  gui_skill_dir="$hermes_home/skills/$GUI_SKILL_NAME"

  [ ! -L "$config_file" ] || die "Symlinked config.yaml is not supported: $config_file"
  [ ! -L "$env_file" ] || die "Symlinked .env is not supported: $env_file"
  [ ! -L "$backup_dir" ] || die "The backup directory cannot be a symlink: $backup_dir"
}

resolve_hermes() {
  hermes_executable="$(command -v hermes 2>/dev/null || true)"
  [ -n "$hermes_executable" ] || {
    if is_english; then
      die "Hermes Agent was not found. Install it before running this setup"
    else
      die "未找到 Hermes Agent，请先安装 Hermes"
    fi
  }
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
      die "Ling GUI Agent Skill 缺少配置模板：$template"
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
    info "Ling GUI Agent Skill 配置：${destination}"
  fi
}

install_gui_skill() {
  local npx_executable=""

  npx_executable="$(command -v npx 2>/dev/null || true)"
  [ -n "$npx_executable" ] || {
    if is_english; then
      die "npx was not found. Install Node.js/npm to install the Ling GUI Agent Skill"
    else
      die "未找到 npx。请先安装 Node.js/npm，以便安装 Ling GUI Agent Skill"
    fi
  }

  skill_install_log="$(mktemp "${TMPDIR:-/tmp}/ling-gui-skill-install.XXXXXX")"
  if is_english; then
    info "Installing the Ling GUI Agent Skill through npx skills..."
  else
    info "正在通过 npx skills 安装 Ling GUI Agent Skill……"
  fi
  if ! env -u OPENROUTER_API_KEY HERMES_HOME="$hermes_home" \
    "$npx_executable" --yes "$SKILLS_CLI_PACKAGE" add "$GUI_SKILL_SOURCE" \
      --global --agent "$GUI_SKILL_AGENT" --skill "$GUI_SKILL_NAME" --yes \
      >"$skill_install_log" 2>&1
  then
    tail -n 40 "$skill_install_log" >&2
    if is_english; then
      die "Ling GUI Agent Skill installation failed"
    else
      die "Ling GUI Agent Skill 安装失败"
    fi
  fi
  [ -r "$gui_skill_dir/SKILL.md" ] || {
    if is_english; then
      die "npx skills succeeded but the Skill was not found at $gui_skill_dir/SKILL.md"
    else
      die "npx skills 返回成功，但未找到 Skill：$gui_skill_dir/SKILL.md"
    fi
  }
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
    printf '推荐安装 Ling GUI Agent Skill 以获得更好的 GUI 自动化效果。是否安装？[Y/n] ' > /dev/tty
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
    info "已按用户选择跳过 Ling GUI Agent Skill 安装。"
  fi
}

show_gui_skill_status() {
  if [ -r "$gui_skill_dir/SKILL.md" ]; then
    if is_english; then
      info "Ling GUI Agent Skill: installed ($gui_skill_dir)"
    else
      info "Ling GUI Agent Skill：已安装（${gui_skill_dir}）"
    fi
    if [ -r "$gui_skill_dir/.env" ]; then
      if is_english; then
        info "Ling GUI Agent Skill configuration: present (value hidden)"
      else
        info "Ling GUI Agent Skill 配置：已保存（不显示内容）"
      fi
    else
      if is_english; then
        info "Ling GUI Agent Skill configuration: missing"
      else
        info "Ling GUI Agent Skill 配置：缺失"
      fi
    fi
  else
    if is_english; then
      info "Ling GUI Agent Skill: missing (expected: $gui_skill_dir)"
    else
      info "Ling GUI Agent Skill：缺失（预期：${gui_skill_dir}）"
    fi
  fi
}

require_supported_hermes() {
  local version_output=""
  local version=""
  local major minor ignored_patch

  version_output="$("$hermes_executable" --version 2>/dev/null || true)"
  version="$(printf '%s\n' "$version_output" | sed -nE 's/.*[^0-9]([0-9]+\.[0-9]+\.[0-9]+).*/\1/p' | head -n 1)"
  if [ -z "$version" ]; then
    if is_english; then
      info "Warning: could not determine the Hermes version; continuing with the current providers: configuration contract."
    else
      info "警告：无法识别 Hermes 版本；将按当前 providers: 配置契约继续。"
    fi
    return
  fi

  IFS=. read -r major minor ignored_patch <<< "$version"
  if (( major == 0 && minor < 21 )); then
    if is_english; then
      die "Hermes $version is too old; upgrade to Hermes Agent 0.21.0 or newer"
    else
      die "Hermes $version 版本过旧；请先升级到 Hermes Agent 0.21.0 或更高版本"
    fi
  fi
}

sha256_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    printf 'absent'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  else
    die "shasum or sha256sum is required"
  fi
}

mode_of() {
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

b64_encode() {
  printf '%s' "$1" | base64 | tr -d '\n'
}

b64_decode() {
  local value="$1"
  if printf '' | base64 --decode >/dev/null 2>&1; then
    printf '%s' "$value" | base64 --decode
  else
    printf '%s' "$value" | base64 -D
  fi
}

config_get() {
  "$hermes_executable" config get "$1" 2>/dev/null
}

config_set() {
  "$hermes_executable" config set "$1" "$2" >/dev/null
}

config_unset() {
  "$hermes_executable" config unset "$1" >/dev/null 2>&1 || true
}

update_model_aliases() {
  local action="$1" current result
  current="$("$hermes_executable" config get model_aliases --json 2>/dev/null || printf '{}')"
  result="$(python3 - "$action" "$backup_dir" "$current" "$PROVIDER_ID" "$BASE_URL" "$TOKEN_ENV" <<'PY'
import json, os, pathlib, sys
action, directory, current, provider, url, key_env = sys.argv[1:]
directory = pathlib.Path(directory)
aliases = json.loads(current)
original_path = directory / 'model-aliases.original.json'
managed_path = directory / 'model-aliases.managed.json'
if action == 'install':
    if not original_path.exists():
        original_path.write_text(json.dumps(aliases))
        original_path.chmod(0o600)
    managed = {row['label']: dict(model=row['model'], provider=provider, base_url=url, key_env=key_env)
               for row in json.loads(os.environ['LING_MODEL_CHOICES'])}
    aliases.update(managed)
    managed_path.write_text(json.dumps(managed))
    managed_path.chmod(0o600)
elif managed_path.exists():
    original = json.loads(original_path.read_text())
    for name, entry in json.loads(managed_path.read_text()).items():
        if aliases.get(name) == entry:
            if name in original:
                aliases[name] = original[name]
            else:
                aliases.pop(name, None)
print(json.dumps(aliases))
PY
)"
  if [ "$result" = '{}' ]; then
    config_unset model_aliases
  else
    config_set model_aliases "$result"
  fi
}

update_reasoning_defaults() {
  local action="$1" current result
  current="$("$hermes_executable" config get agent.reasoning_overrides --json 2>/dev/null || printf '{}')"
  result="$(python3 - "$action" "$backup_dir" "$current" "$MODEL" <<'PYREASON'
import json, os, pathlib, sys
action, directory, current, model = sys.argv[1:]
directory = pathlib.Path(directory)
values = json.loads(current)
if not isinstance(values, dict):
    raise SystemExit('agent.reasoning_overrides must be an object')
original_path = directory / 'reasoning.original.json'
managed_path = directory / 'reasoning.managed.json'
if action == 'install':
    if not original_path.exists():
        original_path.write_text(json.dumps(values))
        original_path.chmod(0o600)
    managed = json.loads(managed_path.read_text()) if managed_path.exists() else {}
    for name in [model, *[row['model'] for row in json.loads(os.environ.get('LING_MODEL_CHOICES', '[]'))]]:
        managed[name] = 'high'
    values.update(managed)
    managed_path.write_text(json.dumps(managed))
    managed_path.chmod(0o600)
elif managed_path.exists():
    original = json.loads(original_path.read_text())
    for name, value in json.loads(managed_path.read_text()).items():
        if values.get(name) == value:
            if name in original:
                values[name] = original[name]
            else:
                values.pop(name, None)
print(json.dumps(values))
PYREASON
)"
  if [ "$result" = '{}' ]; then
    config_unset agent.reasoning_overrides
  else
    config_set agent.reasoning_overrides "$result"
  fi
}

read_api_key() {
  if [ -n "$theta_api_key" ]; then
    case "$theta_api_key" in
      *$'\n'*|*$'\r'*) die "The OpenRouter personal token cannot contain a newline" ;;
    esac
    if is_english; then
      info "Using the OpenRouter personal token from OPENROUTER_API_KEY."
    else
      info "使用 OPENROUTER_API_KEY 环境变量中的 OpenRouter 个人令牌。"
    fi
    return
  fi

  [ -r /dev/tty ] || {
    if is_english; then
      die "No interactive terminal. Set OPENROUTER_API_KEY for non-interactive installation"
    else
      die "当前没有交互终端；请通过 OPENROUTER_API_KEY 环境变量提供 OpenRouter 个人令牌"
    fi
  }

  if is_english; then
    printf 'Enter your OpenRouter personal-token API key (stored locally): ' > /dev/tty
  else
    printf '请输入 OpenRouter 个人令牌 APIKey（将保存在本机）: ' > /dev/tty
  fi
  if ! IFS= read -r -s theta_api_key < /dev/tty; then
    printf '\n' > /dev/tty
    die "Could not read the OpenRouter personal token"
  fi
  printf '\n' > /dev/tty
  [ -n "$theta_api_key" ] || die "The OpenRouter personal token cannot be empty"

  case "$theta_api_key" in
    *$'\n'*|*$'\r'*) die "The OpenRouter personal token cannot contain a newline" ;;
  esac
}

write_env_key() {
  local destination="$1"
  local source="$2"
  local value="$3"
  local temporary="$destination.tmp.$$"

  umask 077
  if [ -f "$source" ]; then
    awk -v key="$TOKEN_ENV" '
      $0 ~ "^[[:space:]]*(export[[:space:]]+)?" key "[[:space:]]*=" { next }
      { print }
    ' "$source" > "$temporary"
  else
    : > "$temporary"
  fi
  printf '%s=%s\n' "$TOKEN_ENV" "$value" >> "$temporary"
  chmod 600 "$temporary"
  mv -f -- "$temporary" "$destination"
}

remove_env_key() {
  local destination="$1"
  local temporary="$destination.tmp.$$"
  [ -f "$destination" ] || return
  umask 077
  awk -v key="$TOKEN_ENV" '
    $0 ~ "^[[:space:]]*(export[[:space:]]+)?" key "[[:space:]]*=" { next }
    { print }
  ' "$destination" > "$temporary"
  chmod 600 "$temporary"
  mv -f -- "$temporary" "$destination"
}

create_initial_backup() {
  local default_value=""
  local provider_value=""
  local config_existed="0"
  local env_existed="0"
  local default_existed="0"
  local provider_existed="0"
  local env_mode="600"

  if [ -e "$backup_dir" ]; then
    die "Backup path already exists and is not managed by this installer: $backup_dir"
  fi

  if config_get "providers.$PROVIDER_ID" >/dev/null; then
    if is_english; then
      die "Provider $PROVIDER_ID already exists; refusing to overwrite an unmanaged provider"
    else
      die "provider $PROVIDER_ID 已存在；为避免覆盖非本安装器配置，已停止"
    fi
  fi

  mkdir -p -- "$backup_dir"
  chmod 700 "$backup_dir"

  if [ -f "$config_file" ]; then
    config_existed="1"
    cp -p -- "$config_file" "$backup_dir/config.yaml"
  fi
  if [ -f "$env_file" ]; then
    env_existed="1"
    env_mode="$(mode_of "$env_file")"
    cp -p -- "$env_file" "$backup_dir/.env"
  fi

  if default_value="$(config_get model.default)"; then
    default_existed="1"
  fi
  if provider_value="$(config_get model.provider)"; then
    provider_existed="1"
  fi

  umask 077
  {
    printf 'installer_id=%s\n' "$INSTALLER_ID"
    printf 'installer_version=%s\n' "$SCRIPT_VERSION"
    printf 'config_existed=%s\n' "$config_existed"
    printf 'env_existed=%s\n' "$env_existed"
    printf 'env_mode=%s\n' "$env_mode"
    printf 'model_default_existed=%s\n' "$default_existed"
    printf 'model_default_b64=%s\n' "$(b64_encode "$default_value")"
    printf 'model_provider_existed=%s\n' "$provider_existed"
    printf 'model_provider_b64=%s\n' "$(b64_encode "$provider_value")"
  } > "$manifest_file"
  chmod 600 "$manifest_file"
}

record_installed_hashes() {
  local temporary="$manifest_file.tmp.$$"
  awk '!/^installed_config_sha256=/ && !/^installed_env_sha256=/' "$manifest_file" > "$temporary"
  {
    printf 'installed_config_sha256=%s\n' "$(sha256_file "$config_file")"
    printf 'installed_env_sha256=%s\n' "$(sha256_file "$env_file")"
  } >> "$temporary"
  chmod 600 "$temporary"
  mv -f -- "$temporary" "$manifest_file"
}

install_ling() {
  resolve_hermes
  require_supported_hermes
  if is_english; then
    info "This will configure Hermes to use Ling ($MODEL) through $BASE_URL and store your API key locally."
    info "Existing managed configuration will be backed up for restoration. Backup: $backup_dir"
    info "To restore: run this same script without arguments and choose 9, or run it with --uninstall."
    info "Press Ctrl+C now to cancel."
  else
    info "即将配置 Hermes 使用 Ling（${MODEL}），服务地址为 ${BASE_URL}，并将 API Key 保存在本地。"
    info "将备份所管理的原配置以便恢复。备份位置：$backup_dir"
    info "恢复方法：不带参数运行同一个脚本并选择 9，或使用 --uninstall。"
    info "现在可按 Ctrl+C 取消。"
  fi
  read_api_key

  if [ -d "$backup_dir" ] && ! is_managed_install; then
    die "Backup directory is not owned by this installer: $backup_dir"
  fi
  if [ ! -d "$backup_dir" ] && config_get "providers.$PROVIDER_ID" >/dev/null; then
    if is_english; then
      die "Provider $PROVIDER_ID already exists; refusing to overwrite an unmanaged provider"
    else
      die "provider $PROVIDER_ID 已存在；为避免覆盖非本安装器配置，已停止"
    fi
  fi

  if should_install_gui_skill; then
    install_gui_skill
  else
    skip_gui_skill
  fi

  if [ ! -d "$backup_dir" ]; then
    create_initial_backup
  fi

  if [ -f "$config_file" ]; then
    local previous_sha current_sha safety_copy
    previous_sha="$(manifest_value installed_config_sha256 2>/dev/null || true)"
    current_sha="$(sha256_file "$config_file")"
    if [ -n "$previous_sha" ] && [ "$previous_sha" != "$current_sha" ]; then
      safety_copy="$hermes_home/config.before-ling3-update.$(date +%Y%m%d-%H%M%S).yaml"
      cp -p -- "$config_file" "$safety_copy"
      if is_english; then
        info "Detected post-install configuration changes; saved: $safety_copy"
      else
        info "检测到安装后的配置改动，已先保存：$safety_copy"
      fi
    fi
  fi

  write_env_key "$env_file" "$env_file" "$theta_api_key"
  config_set "providers.$PROVIDER_ID.base_url" "$BASE_URL"
  config_set "providers.$PROVIDER_ID.key_env" "$TOKEN_ENV"
  config_set "providers.$PROVIDER_ID.api_mode" "chat_completions"
  config_set "providers.$PROVIDER_ID.model" "$MODEL"
  if [ "${LING_MODEL_CHOICES:-[]}" != '[]' ]; then
    local model_catalog
    model_catalog="$(python3 - "$MODEL" <<'PY'
import json, os, sys
models = {row['model']: {'name': row['label'], 'context_length': row['context_length']} for row in json.loads(os.environ['LING_MODEL_CHOICES'])}
models.setdefault(sys.argv[1], {'name': sys.argv[1]})
print(json.dumps(models))
PY
)"
    config_set "providers.$PROVIDER_ID.models" "$model_catalog"
    config_set "providers.$PROVIDER_ID.discover_models" "false"
    update_model_aliases install
  fi
  update_reasoning_defaults install
  config_set "model.provider" "$PROVIDER_ID"
  config_set "model.default" "$MODEL"
  record_installed_hashes

  if is_english; then
    info "Installation complete. New Hermes sessions now default to $MODEL."
    info "Added provider: $PROVIDER_ID (other providers and model aliases were preserved)"
    info "Base URL: $BASE_URL"
    info "Protocol: OpenAI Chat Completions"
    info "OpenRouter token: $env_file (value hidden)"
    info "Run this script with --self-test to verify Hermes -> OpenRouter -> Ling."
  else
    info "安装完成。新的 Hermes 会话现在默认使用 ${MODEL}。"
    info "新增 provider：${PROVIDER_ID}（其他 provider 和模型别名均已保留）"
    info "Base URL：$BASE_URL"
    info "协议：OpenAI Chat Completions"
    info "OpenRouter 令牌：${env_file}（不显示内容）"
    info "可运行本脚本 --self-test 验证 Hermes → OpenRouter → Ling。"
  fi
}

show_status() {
  local configured_provider=""
  local configured_model=""
  local provider_url=""
  local provider_mode=""
  local token_state=""

  resolve_hermes
  configured_provider="$(config_get model.provider 2>/dev/null || true)"
  configured_model="$(config_get model.default 2>/dev/null || config_get model.model 2>/dev/null || true)"
  provider_url="$(config_get "providers.$PROVIDER_ID.base_url" 2>/dev/null || true)"
  provider_mode="$(config_get "providers.$PROVIDER_ID.api_mode" 2>/dev/null || true)"
  if [ -f "$env_file" ] && grep -Eq "^[[:space:]]*(export[[:space:]]+)?${TOKEN_ENV}[[:space:]]*=" "$env_file"; then
    if is_english; then token_state="saved"; else token_state="已保存"; fi
  else
    if is_english; then token_state="missing"; else token_state="缺失"; fi
  fi

  if is_english; then
    if is_managed_install; then info "Status: installed"; else info "Status: not installed"; fi
    info "Hermes directory: $hermes_home"
    info "Default provider: ${configured_provider:-<unset>}"
    info "Default model: ${configured_model:-<unset>}"
    info "Ling provider URL: ${provider_url:-<unset>}"
    info "Ling API mode: ${provider_mode:-<unset>}"
    info "OpenRouter token: $token_state (value hidden)"
    show_gui_skill_status
  else
    if is_managed_install; then info "状态：已安装"; else info "状态：未安装"; fi
    info "Hermes 配置目录：$hermes_home"
    info "默认 provider：${configured_provider:-<未设置>}"
    info "默认模型：${configured_model:-<未设置>}"
    info "Ling provider URL：${provider_url:-<未设置>}"
    info "Ling API 模式：${provider_mode:-<未设置>}"
    info "OpenRouter 令牌：${token_state}（不显示内容）"
    show_gui_skill_status
  fi
}

restore_model_value() {
  local key="$1"
  local existed_key="$2"
  local value_key="$3"
  local existed value
  existed="$(manifest_value "$existed_key")"
  if [ "$existed" = "1" ]; then
    value="$(b64_decode "$(manifest_value "$value_key")")"
    config_set "$key" "$value"
  else
    config_unset "$key"
  fi
}

uninstall_ling() {
  local installed_config_sha current_config_sha installed_env_sha current_env_sha
  local config_changed="1"
  local env_changed="1"

  resolve_hermes
  is_managed_install || {
    if is_english; then die "No managed Hermes Ling installation was found"; else die "未找到由本安装器管理的 Hermes Ling 配置"; fi
  }

  installed_config_sha="$(manifest_value installed_config_sha256 2>/dev/null || true)"
  current_config_sha="$(sha256_file "$config_file")"
  installed_env_sha="$(manifest_value installed_env_sha256 2>/dev/null || true)"
  current_env_sha="$(sha256_file "$env_file")"
  [ -n "$installed_config_sha" ] && [ "$installed_config_sha" = "$current_config_sha" ] && config_changed="0"
  [ -n "$installed_env_sha" ] && [ "$installed_env_sha" = "$current_env_sha" ] && env_changed="0"

  if [ "$config_changed" = "0" ]; then
    if [ "$(manifest_value config_existed)" = "1" ]; then
      cp -p -- "$backup_dir/config.yaml" "$config_file"
    else
      rm -f -- "$config_file"
    fi
  else
    resolve_hermes
    config_unset "providers.$PROVIDER_ID"
    if [ -f "$backup_dir/model-aliases.managed.json" ]; then
      update_model_aliases restore
    fi
    if [ -f "$backup_dir/reasoning.managed.json" ]; then
      update_reasoning_defaults restore
    fi
    restore_model_value "model.provider" model_provider_existed model_provider_b64
    restore_model_value "model.default" model_default_existed model_default_b64
  fi

  if [ "$env_changed" = "0" ]; then
    if [ "$(manifest_value env_existed)" = "1" ]; then
      cp -p -- "$backup_dir/.env" "$env_file"
    else
      rm -f -- "$env_file"
    fi
  else
    remove_env_key "$env_file"
    if [ -f "$backup_dir/.env" ] && grep -Eq "^[[:space:]]*(export[[:space:]]+)?${TOKEN_ENV}[[:space:]]*=" "$backup_dir/.env"; then
      local original_line temporary
      original_line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${TOKEN_ENV}[[:space:]]*=" "$backup_dir/.env" | tail -n 1)"
      temporary="$env_file.tmp.$$"
      if [ -f "$env_file" ]; then cp -- "$env_file" "$temporary"; else : > "$temporary"; fi
      printf '%s\n' "$original_line" >> "$temporary"
      chmod "$(manifest_value env_mode)" "$temporary"
      mv -f -- "$temporary" "$env_file"
    elif [ "$(manifest_value env_existed)" = "1" ] && [ -f "$env_file" ]; then
      chmod "$(manifest_value env_mode)" "$env_file"
    fi
  fi

  rm -f -- "$backup_dir/config.yaml" "$backup_dir/.env" "$manifest_file" "$backup_dir/model-aliases.original.json" "$backup_dir/model-aliases.managed.json" "$backup_dir/reasoning.original.json" "$backup_dir/reasoning.managed.json"
  rmdir -- "$backup_dir" 2>/dev/null || true

  if is_english; then
    info "Uninstall complete. Removed the managed Ling provider and token."
    info "Restored the previous default provider/model; unrelated Hermes data was preserved."
    if [ -r "$gui_skill_dir/SKILL.md" ]; then
      info "Ling GUI Agent Skill remains installed at: $gui_skill_dir"
      if [ -r "$gui_skill_dir/.env" ]; then
        info "Its .env and GUI Agent token were retained for other Agent integrations."
      fi
    fi
  else
    info "卸载完成：已移除安装器管理的 Ling provider 和令牌。"
    info "已恢复此前的默认 provider/模型；其他 Hermes 数据均已保留。"
    if [ -r "$gui_skill_dir/SKILL.md" ]; then
      info "Ling GUI Agent Skill 保留在：$gui_skill_dir"
      if [ -r "$gui_skill_dir/.env" ]; then
        info "其中的 .env 和 GUI Agent 令牌已为其他 Agent 集成保留。"
      fi
    fi
  fi
}

self_test() {
  local answer=""
  resolve_hermes
  is_managed_install || die "Install the Ling provider before running --self-test"
  if is_english; then
    info "Running Hermes -> OpenRouter Chat Completions -> Ling self-test..."
  else
    info "正在执行 Hermes → OpenRouter Chat Completions → Ling 自检……"
  fi
  answer="$("$hermes_executable" -z 'Reply with exactly 42 and no other text.')"
  [ "$(printf '%s' "$answer" | tr -d '[:space:]')" = "42" ] || {
    if is_english; then die "Self-test returned an unexpected answer (content hidden)"; else die "自检返回了非预期内容（内容已隐藏）"; fi
  }
  if is_english; then info "Self-test passed: Hermes -> OpenRouter -> $MODEL returned 42."; else info "自检通过：Hermes → OpenRouter → $MODEL 返回 42。"; fi
}

confirm_uninstall() {
  local answer=""
  [ -r /dev/tty ] || die "No interactive terminal; append --yes for automated uninstall"
  if is_english; then
    printf 'Remove the Ling provider and restore the previous Hermes default? [y/N] ' > /dev/tty
  else
    printf '移除 Ling provider 并恢复此前的 Hermes 默认模型？[y/N] ' > /dev/tty
  fi
  IFS= read -r answer < /dev/tty || return 1
  case "$answer" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

interactive_menu() {
  if is_english; then
    cat <<EOF
Ling-3.0-flash-VL Hermes setup v$SCRIPT_VERSION (English)

1) Add/update Ling and make it the default
2) Show status
3) Run end-to-end self-test
9) Uninstall Ling and restore the previous default
0) Exit
EOF
    printf 'Select [0/1/2/3/9]: ' > /dev/tty
  else
    cat <<EOF
Ling-3.0-flash-VL Hermes 配置工具 v$SCRIPT_VERSION

1) 新增/更新 Ling 并设为默认模型
2) 查看状态
3) 运行端到端自检
9) 卸载 Ling 并恢复此前默认模型
0) 退出
EOF
    printf '请选择 [0/1/2/3/9]: ' > /dev/tty
  fi
  IFS= read -r choice < /dev/tty || die "Could not read the menu selection"
  case "$choice" in
    1) read_api_key; install_ling ;;
    2) show_status ;;
    3) self_test ;;
    9) confirm_uninstall && uninstall_ling || info "Cancelled." ;;
    0) info "Exited." ;;
    *) die "Invalid selection: $choice" ;;
  esac
}

resolve_paths

case "${1:-}" in
  -h|--help)
    usage
    ;;
  "")
    [ -r /dev/tty ] || die "No interactive terminal; use --install, --status, or --uninstall --yes"
    interactive_menu
    ;;
  --install)
    [ "$#" -eq 1 ] || die "--install does not accept additional arguments"
    install_ling
    ;;
  --status|--self-test)
    [ "$#" -eq 1 ] || die "$1 does not accept additional arguments"
    if [ "$1" = "--status" ]; then show_status; else self_test; fi
    ;;
  --uninstall)
    if [ "$#" -eq 2 ] && [ "$2" = "--yes" ]; then
      uninstall_ling
    elif [ "$#" -eq 1 ]; then
      if confirm_uninstall; then uninstall_ling; else info "Cancelled."; fi
    else
      die "Uninstall usage: --uninstall [--yes]"
    fi
    ;;
  *)
    usage >&2
    die "Unknown argument: $1"
    ;;
esac
