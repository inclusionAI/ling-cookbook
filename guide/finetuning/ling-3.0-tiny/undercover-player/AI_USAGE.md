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
| 调试 | 全部报错定位与修复（见 README Troubleshooting 5 项） |
| 应用集成 | Avernet #2611 玩家 Bot 代码（BCN V3 协议层 / 游戏逻辑层 / 清洗层）与回环自测 |
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

## 边界声明

- 未使用任何非公开数据集；训练数据全部由模板程序化合成；
- 未伪造评测数字：README 对比表以 `eval/*.json` 实测为准；
- AI 对开源协议（Apache-2.0 等）只做提示，最终许可证合规由人工确认。
