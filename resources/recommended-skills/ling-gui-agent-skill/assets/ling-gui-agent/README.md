# Ling GUI Agent

A compact local GUI-agent CLI powered only by Ling. It repeats a screenshot, planning, and action loop until the task finishes or reaches its step limit.

- Android control uses Appium and ADB.
- Desktop control uses PyAutoGUI, with optional native Quartz multi-click events on macOS.

## Installation

Python 3.10 or newer is required:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

Copy `.env.example` to `.env` and fill in the three required connection settings:

```dotenv
LING_BASE_URL=https://example.com/v1
LING_MODEL=your-model-name
LING_API_KEY=your-api-key
```

The local `.env` is ignored by Git. When `LING_CONFIG_FILE` is used by the bundled skill runner, its local `LING_*` values override inherited process values so credentials never need to appear in a command.

## Quick start

### Android

Android mode requires Node.js, Android SDK Platform-Tools, Appium Server, and the UiAutomator2 driver:

```bash
node --version
npm --version
npm install -g appium
appium driver install uiautomator2
appium driver doctor uiautomator2
adb devices
appium
PLATFORM=mobile python -m ling_gui_agent "Open the photo gallery"
```

Appium Server and the UiAutomator2 driver are separate installations. The environment is ready when the driver doctor reports no required fixes.

### Desktop

Desktop mode does not require Appium or ADB:

```bash
PLATFORM=desktop python -m ling_gui_agent "Open a browser and search for the weather"
```

On macOS, grant Accessibility and Screen Recording permission to the terminal or host application. On Windows, PyAutoGUI uses the native desktop coordinate space and the skill runner supports PowerShell and Command Prompt. Desktop mode controls the real mouse, keyboard, and clipboard, so test it in a safe environment first.

## Configuration

| Variable | Default | Description |
|---|---:|---|
| `LING_BASE_URL` | required | OpenAI-compatible API base URL. |
| `LING_API_KEY` | required | API key. Never commit or pass it on the command line. |
| `LING_MODEL` | required | Model name supported by the endpoint. |
| `LING_TIMEOUT_SECONDS` | `180` | Request timeout in seconds. |
| `LING_MAX_TOKENS` | `4096` | Maximum output tokens for one model turn. |
| `LING_TEMPERATURE` | `0.3` | Sampling temperature. |
| `LING_MIN_PIXELS` | `102400` | Minimum image-pixel budget sent with each model image. |
| `LING_MAX_PIXELS` | `2621440` | Maximum image-pixel budget. Images retain aspect ratio and align dimensions to multiples of 32. |
| `LING_MAX_VISIBLE_SCREENSHOTS` | `3` | Number of context screenshots including the current one. |
| `LING_DEBUG` | `false` | Create annotated frames, HTML, and MP4 visualization when enabled. |
| `PLATFORM` | `mobile` | `mobile`, `desktop`, `web`, or `auto`. |
| `MAX_STEPS` | `50` | Maximum action steps. |
| `STEP_DELAY_SECONDS` | `2.0` | Delay after each action. |
| `SCREENSHOT_DIR` | `logs` | Run screenshot and JSON artifact directory. |
| `APPIUM_SERVER_URL` | `http://127.0.0.1:4723` | Android Appium server URL. |
| `APPIUM_UDID` | empty | Optional ADB device serial. |
| `LING_GUI_LOG_DIR` | `logs/llm_logs` | Raw Ling response-log directory. |

Override the step count for one run with:

```bash
ling-gui-agent --max-steps 10 "Open Settings"
```

## Actions

Mobile supports click, double-click, long press, drag, swipe, text input, app launch, system keys, wait, user handoff, and completion.

Desktop additionally supports triple-click, right-click, middle-click, hover, keyboard shortcuts, modified clicks and drags, and desktop-style scrolling for `Swipe`. On macOS, `ctrl` in model actions maps to `command`. Text input pastes Unicode through the clipboard, and a trailing `\\n` presses Enter.

## Logs and debug visualization

Each run stores screenshots, model request and response JSON, and parsed actions under `logs/run_YYYYMMDD_HHMMSS/`. These files may contain screen content and task text; do not commit them.

When `LING_DEBUG=true`, the executor also creates annotated `step_<n>_out.png` frames and `visualization.html`. If FFmpeg is installed, it creates `sequence.mp4`. Normal runs do not generate these visualization files.

Generate visualization explicitly from an existing log without calling the model:

```bash
python -m ling_gui_agent.log_visualizer logs/run_YYYYMMDD_HHMMSS
```

## Testing

```bash
python -m compileall -q ling_gui_agent tests
python -m unittest discover -s tests -v
```

Real API tests are disabled unless `LING_RUN_API_TEST=1` is explicitly set.

## Project layout

```text
ling_gui_agent/       # Source package
├── device/           # Appium and desktop controllers
└── planner/          # Ling prompts, parsing, and LLM client
tests/                # Unit tests and optional API validation
.env.example          # Non-secret configuration template
pyproject.toml        # Packaging and dependencies
```

## Limitations

- This is vision-coordinate automation, not DOM or native-control automation. UI changes can invalidate an action.
- Desktop screenshots and PyAutoGUI coordinates use the primary display. Multi-display layouts require platform-specific verification.
- Android support is limited to UiAutomator2; iOS is not supported.
- Do not perform payments, deletion, or other irreversible actions without a human confirmation boundary.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
