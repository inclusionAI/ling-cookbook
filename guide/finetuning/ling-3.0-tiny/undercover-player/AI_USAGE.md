# AI 使用策略报告

本项目（Ling-3.0-tiny「谁是卧底」玩家决策组件 Recipe）在 **Claude Code**
（Anthropic，模型 claude-opus-5-5 / claude-sonnet-5-5）辅助下完成，遵循
「人工决策、AI 执行、可验证交付」的原则。按轻训营要求如实披露如下。

## AI 承担的部分

| 环节 | AI 参与方式 |
|---|---|
| 方案调研 | 阅读 Ling/MLX 官方 notebook 与文档，确认 `bailing_hybrid` 架构在 mlx-lm≥0.32.0 可用、需本地量化、QLoRA 可行性（内存估算） |
| 场景设计 | 提出「谁是卧底」玩家决策组件等候选方向，给出利弊分析 |
| 数据合成 | `data/` 全部代码（词对库、提示词格式、合成规则、规则式投票标签） |
| 训练配置 | `train/lora_config.yaml`、`train.sh`（QLoRA 超参与内存优化取舍） |
| 评测 | `eval/evaluate.py`（含反刷指标的 `looks_natural` 检查） |
| 调试 | 全部报错定位与修复（见 README Troubleshooting 8 项） |
| 应用集成 | Avernet #2611 玩家 Bot 代码（BCN V3 协议层 / 游戏逻辑层 / 清洗层）与回环自测 |
| 数据飞轮 | `flywheel.py`（实战对局 → 真值投票/自我模仿增量）与 `distill.py`（教师蒸馏增量，见下节） |
| 文档 | README、本报告 |

## 人工承担的部分

- **方向决策**：选定「谁是卧底」场景与「一鱼两吃」（ling-cookbook #11 +
  Avernet #2611）总体方案，批准后才动工；
- **账号与认领**：GitHub PAT、issue 认领、云燕平台操作（AI 不代持凭据写入任何
  产物）；
- **资源与验收**：提供本机（M4 16GB）与网络环境，对数据质量、指标口径、最终
  PR 内容把关。

## 验证方式

所有 AI 产出均可独立验证，不依赖 AI 的自我声明：

1. **数据可复现**：`data/synthesize.py` 固定随机种子，重新生成即可比对；
2. **指标可复算**：`eval/evaluate.py` 对 test.jsonl 逐条推理并落盘 JSON，
   任何人可重跑核对 README 中的对比表；
3. **集成可自测**：Avernet Bot 附带 `test_loopback.py`（mock 服务器回环），
   不起真环境即可验证协议链路；
4. **过程可追溯**：训练日志、checkpoint（每 200 步）、评测 JSON 均随
   Release/PR 留档。

## 第三方模型 API 使用披露（数据飞轮·蒸馏轨道）

按「如实披露」原则单独说明：迭代进阶的**蒸馏数据管线**（`distill.py`）在
**离线阶段**调用了第三方大模型 API（Anthropic 兼容 Messages 接口，教师模型
`mimo-v2.6-pro`），对「谁是卧底」状态生成发言/投票动作作为训练标签增量
（`data/api-distill-vN.jsonl`），用于下一轮 QLoRA 微调。

边界与保证：

- **绝不进入游戏循环**：Avernet 对局中玩家 Bot 的每一次发言/投票都由本机
  MLX 推理的微调小模型产生（引擎身份「Apple MLX 本地端侧模型引擎」的
  「不调用任何云端大模型 API」承诺不受影响），验收对局日志可逐条核对；
- **只做离线数据**：教师仅在数据管线里出标签，样本经质检（不泄词/≤25 字/
  票面格式/状态去重）与固定 test.jsonl 评测门槛后才可能被合并进训练集；
- **凭据安全**：API token 只存本机 `~/.distill-api.env`（chmod 600），
  不进 Git、不在任何产物中回显；
- 合成数据（`synthesize.py`）与实战真值增量（`flywheel.py`）不依赖任何
  第三方 API；蒸馏轨道是可选进阶，去掉它 recipe 主线完整可复现。

## 边界声明

- 未使用任何非公开数据集；训练数据主体为模板程序化合成，增量来自自建牌桌
  对局（`flywheel.py`）与上节披露的教师蒸馏（`distill.py`），均可复算；
- 未伪造评测数字：README 对比表以 `eval/*.json` 实测为准；
- AI 对开源协议（Apache-2.0 等）只做提示，最终许可证合规由人工确认。
