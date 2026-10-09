# A26：真实训练已完成，当前 feedback 扩展 NO_GO

N1/N2/N3 各2000更新、31,810参数。复杂材料 CNN 先验收益约23%–24%，OM/feedback 对匹配 BP/无反馈对照的独立优势未建立；N2 warm慢 3.274 倍。第二seed和locked ID/OOD未运行。

- [完整中文研究报告](A26_RESEARCH_REPORT_ZH.md)
- [科学裁决](A26_FINAL_DECISION.md) / [冻结 gates](GATE_DECISION.json)
- [网络与实际 checkpoint 记录](A26_NETWORK_COMPARISON.md)
- [复杂结构结果](A26_COMPLEX_SHAPE_RESULTS.md)
- [反馈增量](A26_FEEDBACK_INCREMENTAL_VALUE.md)
- [物理/先验边界](A26_PHYSICS_PRIOR_SPLIT.md)
- [完整成本](A26_COST_AND_RUNTIME.md)
- [探索与失败](A26_EXPLORATION_LEDGER.md)
- [复现说明](REPRODUCING_A26.md)

所有推断限于已知背景、二维 TM 24×24、固定复相位材料增量与已暴露开发验证。只有一个训练 seed、一个训练规模和一种 CNN 深度；不支持独立泛化、任意复材料、3D、样本效率或层数减少结论。

[GitHub](https://github.com/migodam/a26-om-feedback-prior-completion) · [匿名 raw 报告](https://raw.githubusercontent.com/migodam/a26-om-feedback-prior-completion/main/A26_RESEARCH_REPORT_ZH.md) · [实际 checkpoint/数据 Release](https://github.com/migodam/a26-om-feedback-prior-completion/releases/tag/a26-v0.1.0)
