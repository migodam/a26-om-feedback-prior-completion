# A26 科学与可视证据审阅

主 Codex 审阅实际 per-parent 配对、冻结 gates、物理诊断、原始输出和渲染图。
本记录不把 unit tests、图表 QA 或公开发布当作科学 PASS。

- 三模型各2000次真实训练更新，31,810参数，训练/验证采样与权限匹配；
  真实 best/middle/final checkpoint 存在并在评估中重载。
- 复杂三家族的主配对没有支持 N2 的10%材料改善；数据残差与完整
  inference latency 也未通过门槛，保留 NO_GO。
- 所有带噪观测的初值反馈 beta 小于1，缺少高反馈信号组，H4无法在
  此分布中作高/低信号机制判定。没有事后加对象或改初始化救主结果。
- 独立 predicted-state 审核保留 TASK_R32 梯度方向失败；主 encoder
  固定在初值，后续诊断不能被写成主训练每一步的物理误差。
- 检视实际 Δχ 实部图：shell 内腔被填平；smooth/ring 出现真值没有的
  中心空缺；asymmetric 离散边界和小部件模糊。先验改善误差与形状偏置
  同时存在，报告标记 prior-dependent reconstruction。
- 配对家族图与数据对应，零线、区间、单位清楚；quality–time 图明确
  resident warm 和 fresh encoder 的不同范围，没有 cached-CNN 提速声称。
- 完整χ与Δχ分别公开，避免已知强 ring 背景掩盖增量材料错误。
- 验证集是已暴露 development，checkpoint 按它选择；bootstrap 为
  描述性开发不确定性。第二 seed 与 locked ID/OOD 都 NOT_RUN。
- SOM-Net官方、完整TV成像轨迹、sample-efficiency、层数减少、任意
  独立复相位材料和3D指标没有本次证据，不给优越性结论。
- 135项测试运行、132通过、3个继承的可选Clarabel测试跳过；所有失败、
  入口修复与必要重跑保留费用。原始终端 receipts 不被报告改写。

发布前检查凭证、未打开的locked数组和第三方文件来源；来源保留Git
commit，按用户要求不执行SHA256。发布状态独立于科学NO_GO。
