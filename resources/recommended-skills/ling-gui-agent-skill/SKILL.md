---
name: ling-gui-agent-skill
description: Configure and run the bundled Ling GUI Agent for vision-driven Android Appium or desktop automation on macOS, Windows, and Linux. Use when a task must interact with a real local GUI through screenshots, pointer actions, keyboard input, scrolling, or Android system actions.
---

# Ling GUI Agent

Use `scripts/run.py` to run the project bundled in `assets/ling-gui-agent/`. Keep the runner in the foreground so framework cancellation reaches the complete child process group.

## Configuration

Use the skill-root `.env` by default. For an externally managed secret file, use `--config /private/path/ling.env`; `LING_CONFIG_FILE` is also supported. Precedence is `--config`, `LING_CONFIG_FILE`, then the skill-root `.env`.

The selected file must define `LING_BASE_URL`, `LING_MODEL`, and `LING_API_KEY`. `LING_BASE_URL` must be the real base URL of an OpenAI-compatible API and normally has the form `https://<host>/v1`. Empty, malformed, or placeholder values such as `https://example.com/v1` are not usable. If the value is invalid, do not guess or substitute an endpoint; ask the user for the exact OpenAI-compatible API base URL supplied by their service provider.

Never print configuration contents or pass credentials as command-line arguments. If the file or another required value is missing, do not start the GUI agent. Tell the user to copy `.env.example` to a private file, fill in the required values, and retry.

## Run

From the skill root, run a bounded foreground task:

```bash
python3 scripts/run.py "<goal>" --platform desktop --max-steps 100
```

For an external configuration file:

```bash
python3 scripts/run.py "<goal>" --platform desktop --config /private/path/ling.env --max-steps 100
```

Use `--platform mobile` for Android and `--platform desktop` for desktop applications or browsers. Do not detach the command or launch it through an independent background shell.

The runner permits only one active controller per desktop or selected Android device. If it reports that the device is busy, do not start a concurrent retry. The total task timeout defaults to 600 seconds; use `--timeout <seconds>` to override it. A timeout stops the child process group and returns exit code `124`. User cancellation returns a signal-derived nonzero status, normally `130` for `SIGINT` or `143` for `SIGTERM`. Treat every other nonzero status as a failed run.

## Diagnose and set up

Do not run preflight before every task. Use it after first-time setup or when configuration, dependency, device, or permission errors occur:

```bash
python3 scripts/run.py --check --platform desktop
```

Preflight validates the selected configuration, Python dependencies, and basic platform prerequisites without starting an agent task, clicking, typing, or calling the model.

If the dependency probe fails, create the skill-local environment once and retry the original command.

macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e assets/ling-gui-agent
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -e assets\ling-gui-agent
```

Do not reinstall dependencies on successful runs.

## Platform and safety boundaries

- The Skill may detect macOS permission status, but it must not grant, change, bypass, or directly operate system permissions, open System Settings, or run permission-modifying commands. Only the user may authorize access.
- macOS requires **Accessibility** permission so the agent can click, type, and scroll, and **Screen Recording** permission so it can read the screen. If either permission is missing, stop the task, name the missing permission, and ask the user to enable it manually for Codex, Claude Code, or Terminal under **System Settings > Privacy & Security > Accessibility** or **Screen Recording**. Retry only after the user confirms that they enabled the permission and restarted the host application.
- Android automation requires `adb`, a connected device, Appium Server, and the UiAutomator2 driver.
- The agent controls the real mouse, keyboard, clipboard, desktop, browser, or Android device. Obtain explicit user confirmation before payments, deletion, publishing, sending messages, or other irreversible or externally visible actions.
- Runs save screenshots, task text, model exchanges, and parsed actions. Treat generated logs as sensitive and never commit or expose them.
- Set `LING_DEBUG=true` in the selected configuration file only when annotated HTML or video diagnostics are needed.
