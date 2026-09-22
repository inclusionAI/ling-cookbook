<p align="right">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

# Ling GUI Agent Skill

Ling GUI Agent Skill 是专为 Ling VL（Vision-Language 视觉语言）模型家族构建的跨平台 GUI 自动化技能。由 Ling-3.0-flash-VL 等多模态模型提供驱动，能够直接理解屏幕像素视觉信息，在操作系统桌面与 Android 移动端界面中执行自动化交互。

该 Skill 采用纯视觉、基于坐标（Coordinate-based）的交互体系，能够自主观测屏幕状态、执行光标点击、文本键入、手势滑动及各类系统级 GUI 操作以达成用户目标。

## 使用前准备

在启动 Skill 之前，请准备好以下前置条件：

- 一个支持多模态图像输入的 OpenAI 兼容 API 端点（适配 Ling VL 模型系列）。
- API 端点 URL、模型名称（如 `Ling-3.0-flash-VL`）及有效的 API Key。
- 具备对该 API 端点的网络访问能力。
- 具备写权限的本地 Skill 目录，以及受支持的目标桌面或移动设备环境。

Skill 支持在本地初始化独立的 Python 虚拟环境并安装所需运行时依赖；若宿主环境权限不足或依赖安装失败，会提供明确的配置报错提示。

## 环境要求

### 1. 桌面端自动化 (Desktop Automation)

需准备以下环境：

- Python 3.10 或更高版本。
- 待操作应用程序处于已登录、未锁屏的交互式桌面会话中。
- **macOS**：必须为运行本 Skill 的终端应用或 Agent 宿主程序手动授予系统的 **「辅助功能 (Accessibility)」** 与 **「屏幕录制 (Screen Recording)」** 权限。
- **Linux**：必须安装 PyAutoGUI 所依赖的系统级截图与剪贴板工具组件（如 `scrot`、`xclip`、`python3-tk` 等）。

### 2. Android 移动端自动化 (Android Automation)

需准备以下工具链：

- Python 3.10 或更高版本。
- Node.js 与 npm。
- Android SDK Platform-Tools（确保 `adb` 命令在系统 PATH 中可用）。
- Java JDK（建议 JDK 17+）。
- 正常运行且已安装 `UiAutomator2` 驱动的 Appium 服务（可通过 `appium driver install uiautomator2` 安装）。
- 已开启「USB 调试」的已连接 Android 实机，或可用且在线的 Android 模拟器。

*注：本 Skill 不会自动申请操作系统权限、配置设备连接或安装 Android 开发工具链，上述前置步骤需由开发者自行准备完毕。*

## API 配置

在当前目录下基于 `.env.example` 创建本地配置文件，并填入以下配置项：

| 配置键名 | 说明 |
| :--- | :--- |
| `LING_BASE_URL` | OpenAI 兼容 API 基础地址（如 `https://<host>/v1`）。 |
| `LING_MODEL` | Ling VL 模型名称，例如 `Ling-3.0-flash-VL`。 |
| `LING_API_KEY` | 对应端点的有效 API Key。 |

必须在配置填写完整后方可执行任务。切勿将含有密钥的配置文件提交至代码仓库，也严禁在命令行执行参数中暴露密钥。

## 权限与安全卡点 (Permissions and Safety)

本 Skill 具备真实的输入控制能力，可直接操控鼠标、键盘、系统剪贴板以及移动端界面：

- 必须确保已在操作系统中显式授权对应权限。
- **高危操作人工卡点**：在涉及**资金支付、文件删除、网络公开发布、对外发送即时消息**等任何具有不可逆后果或外部可见的操作之前，Skill **必须先获得人类用户的明确授权与确认**，严禁擅自跳过。
- 为避免输入竞争与动作错乱，同一台桌面或同一部 Android 设备在同一时间仅允许一个活跃控制器独占运行。

## 隐私与运行日志 (Privacy and Logs)

每次自动化运行可能会在本地记录任务截图、指令上下文与动作序列以辅助诊断。这些数据可能包含屏幕上的敏感隐私内容：

- 严格将本地日志与配置文件保留在安全受控目录内。
- 严禁将含有屏幕截图与个人数据的日志上传或提交至公开版本控制系统。
- 仅在遇到故障需要诊断时开启 Debug 级输出。

## 开源许可证 (License)

本项目采用 [MIT License](LICENSE) 许可证发布。
