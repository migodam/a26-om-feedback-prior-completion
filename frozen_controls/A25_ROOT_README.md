# A25：自适应OM反馈

逆散射研究希望从有限的接收数据恢复空间材料。理解这一问题，先要从场的传播和多次散射出发：每个照明源激发的电流受同一材料控制，电流又通过内部传播改变局部场。已知强背景会重塑这条反馈链。材料图准确、满足完整波动方程、以及获得部署速度，是三个需要各自证据的问题。

A25保持24×24二维TM、8个源和32个接收点。材料是576个复数单元，真实材料坐标1152维；每个场景有512个实数观测。所有方法使用同一全网格材料、box约束和TV先验。二维证据不扩展为三维vector-Maxwell结论。

[从这里开始](A25_START_HERE.md) · [完整中文研究报告](A25_RESEARCH_REPORT_ZH.md) · [最终裁决](A25_FINAL_RESEARCH_DECISION.md) · [完整动作账本Release压缩包](https://github.com/migodam/a25-adaptive-om-feedback/releases/download/a25-v0.1.0/A25_COMPLETE_ACTION_LEDGERS.zip) · [失败](FAILURE_LEDGER.jsonl)

| 问题 | 父线程裁决 |
| --- | --- |
| solver_health | PASS |
| feedback_imaging | PARTIAL |
| compression | PARTIAL_DEVELOPMENT_PASS_VALIDATION_FAIL |
| learning | NOT_RUN |
| sample_efficiency | NOT_RUN |
| OOD | NOT_RUN |
| depth | NOT_RUN |
| locked_validation | CLOSED_NOT_RUN |
| deployment | NOT_ESTABLISHED |

已接受作业 CPU计费 1640.92 秒 /7200秒；GPU占用 116.575 秒 /14400秒。报告自身仍在运行的尾部单独记为 provisional；完整截止作业清单见 results/a25/report/COST_SUMMARY.json，最终资源以不可变 job_receipt.json 为准。[完整动作账本压缩包](https://github.com/migodam/a25-adaptive-om-feedback/releases/download/a25-v0.1.0/A25_COMPLETE_ACTION_LEDGERS.zip)包含项目相对路径 `COST_LEDGER.jsonl`及各作业的 `results/a25/jobs/<job>/COST_LEDGER.jsonl`；大文件通过Release提供，公开CSV、图表和小型metadata仍保留在Git。

## 报告与原始证据

[96项数值测试与证据边界](NUMERICAL_VERIFICATION.md) · [复现说明](REPRODUCING_A25.md)。

- [A25_START_HERE.md](A25_START_HERE.md)
- [SOLVER_REPAIR_REPORT.md](SOLVER_REPAIR_REPORT.md)
- [FEEDBACK_HEADROOM.md](FEEDBACK_HEADROOM.md)
- [CONVERGENCE_VS_RUNTIME.md](CONVERGENCE_VS_RUNTIME.md)
- [FEEDBACK_COMPRESSION_EXPLORATION.md](FEEDBACK_COMPRESSION_EXPLORATION.md)
- [FEEDBACK_REGIME_MAP.md](FEEDBACK_REGIME_MAP.md)
- [SOMNET_MIGRATION_PILOT.md](SOMNET_MIGRATION_PILOT.md)
- [EXPLORATION_LEDGER.md](EXPLORATION_LEDGER.md)
- [LOCKED_VALIDATION.md](LOCKED_VALIDATION.md)
- [A25_FINAL_RESEARCH_DECISION.md](A25_FINAL_RESEARCH_DECISION.md)

图表和plot_data均从保存的真实输出生成。NOT_RUN项目不绘制虚拟数据。完整数据和checkpoint发布位置由父线程最终publication记录提供，发布前不声称匿名链接已经可访问。

[A24 冻结研究报告](https://github.com/migodam/a24-source-anchored-om-feedback/blob/ca80ca2a29867e1b03589061dd5c17688c9acbe0/A24_RESEARCH_REPORT_ZH.md)，[SOM-Net 原论文](https://arxiv.org/abs/2209.03567)，[A24 文献读取范围](theory/SOURCES_AND_READING_LEDGER.md)。
