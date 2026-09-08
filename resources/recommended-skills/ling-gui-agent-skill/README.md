# Ling GUI Agent Skill

Ling GUI Agent Skill is a GUI automation skill built for the Ling VL (Vision-Language) model family. Powered by Ling VL models—such as Ling-3.0-flash-VL—it interprets on-screen visuals and carries out actions across desktop and Android interfaces.

It uses visual, coordinate-based interaction. It can observe the screen, click, type, scroll, and perform other supported GUI actions in response to a task.

## Before You Start

Prepare the following before using the Skill:

- A Ling VL model endpoint that accepts image input through an OpenAI-compatible API.
- The endpoint URL, model name, and API key.
- Network access to the selected API endpoint.
- A writable local Skill directory and a supported target device or desktop environment.

The Skill can create or use a Skill-local Python environment and install its Python runtime dependencies. It will report a clear setup error if the host does not permit that setup or if required dependencies cannot be installed.

## Environment Requirements

### Desktop Automation

You need to prepare:

- Python 3.10 or later.
- An interactive desktop session for the application you want to automate.
- On macOS, Accessibility and Screen Recording permission for the application that runs the Skill.
- On Linux, the system screenshot and graphical toolkit components required by PyAutoGUI.

### Android Automation

You need to prepare:

- Python 3.10 or later.
- Node.js and npm.
- Android SDK Platform-Tools, including ADB.
- A Java JDK.
- An Appium server with the UiAutomator2 driver installed.
- A connected Android device with USB debugging enabled, or an available Android emulator.

The Skill does not install or grant system permissions, connect a device, or install the Android toolchain on your behalf.

## API Configuration

Create a local configuration file from `.env.example` and provide the following values:

| Setting | Description |
| --- | --- |
| `LING_BASE_URL` | The OpenAI-compatible API base URL. |
| `LING_MODEL` | A Ling VL model name, such as `Ling-3.0-flash-VL`. |
| `LING_API_KEY` | The API key for the selected endpoint. |

The configuration must be complete before the Skill can start a task. Do not share the configuration file, commit it to a repository, or provide API keys in command-line arguments.

## Permissions and Safety

This Skill controls real GUI input, including the mouse, keyboard, clipboard, desktop, browser, or Android device.

- You must manually grant required operating-system permissions. On macOS, this includes Accessibility and Screen Recording.
- The Skill must obtain confirmation before payments, deletion, publishing, sending messages, or other irreversible or externally visible actions.
- Only one controller may operate the same desktop or Android device at a time.

## Privacy and Logs

Each run may save screenshots, task text, model exchanges, and parsed actions as local logs for diagnosis. These files can contain sensitive screen content.

- Keep logs and configuration files private.
- Do not upload, share, or commit them to source control.
- Enable debug output only when diagnostic artifacts are necessary.

## License

This Skill is released under the [MIT License](LICENSE).
