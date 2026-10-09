"""Parent-owned interpretation of the completed A26 development pilot.

Only saved numerical outputs are read. No new physical experiments, labels,
training, candidate selection or locked-cohort access occurs in this module.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import math
import numpy as np
from .budget import budget_history, write_json
from .gates import decision

REPO = 'https://github.com/migodam/a26-om-feedback-prior-completion'
FAMILIES = ('shell', 'asymmetric_piecewise', 'smooth', 'mixed_scale')


def table(rows, columns):
    def value(v):
        if v is None: return 'NOT_RECORDED'
        if isinstance(v, float): return f'{v:.6g}'
        return str(v).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(t for _, t in columns) + ' |',
                      '| ' + ' | '.join('---' for _ in columns) + ' |'] +
                     ['| ' + ' | '.join(value(row.get(k)) for k, _ in columns) + ' |' for row in rows])


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def scientific_reports(root, config, book):
    root = Path(root)
    out = root/'results/a26/evaluation/261009'
    book.check()
    gate = decision(root, config, book)
    stats = json.loads((out/'PAIRED_STATISTICS.json').read_text())
    run = json.loads((out/'RUNTIME_SUMMARY.json').read_text())
    diagnostics = json.loads((out/'PHYSICS_DIAGNOSTICS.json').read_text())
    rows = read_csv(out/'PER_CASE_METRICS.csv')
    families = read_csv(out/'PER_FAMILY_METRICS.csv')
    parents = read_csv(out/'PER_PARENT_METRICS.csv')
    manifests = {m:json.loads((root/f'results/a26/learning/261009/{m}/training_manifest.json').read_text()) for m in config['methods']}
    pair_table = []
    for name in ('N2_vs_N1', 'N2_vs_N3', 'N1_vs_N3', 'N1_vs_BASE_N1', 'N2_vs_BASE_N2', 'N3_vs_BASE_N3'):
        p = stats[name]
        pair_table.append(dict(pair=name, gain=100*p['complex_mean_gain'],
                               lower=100*p['complex_ci95'][0], upper=100*p['complex_ci95'][1]))
    pair_md = table(pair_table, [('pair','配对（复杂三家族）'),('gain','改善 %'),('lower','区间下限 %'),('upper','区间上限 %')])
    family_table = []
    for family in FAMILIES:
        local = {'family':family}
        for method in ('N1','N2','N3','BASE_N1','BASE_N2','BASE_N3'):
            r = next(r for r in families if r['family']==family and r['method']==method)
            local[method] = float(r['delta_nrmse'])
        family_table.append(local)
    family_md = table(family_table, [('family','家族'),('N1','N1'),('N2','N2'),('N3','N3'),('BASE_N1','无 NN L/BP'),('BASE_N2','无 NN R')])
    diagnostic_table = []
    for phase in ('INITIAL','MIDDLE','FINAL'):
        d = [r for r in diagnostics if r['phase']==phase]
        diagnostic_table.append(dict(phase=phase, count=len(d), passed=sum(bool(r.get('strict_A25_task_gate_pass')) for r in d),
                                     min_cos=min(r['gradient_cosine'] for r in d),
                                     max_jvp=max(r['jvp_relative_error'] for r in d),
                                     max_adjoint=max(r['adjoint_relative_error'] for r in d),
                                     max_update=max(r['boxed_update_relative_error'] for r in d)))
    diagnostic_md = table(diagnostic_table,[('phase','状态'),('count','审核数'),('passed','旧任务精度门槛通过数'),('min_cos','最小梯度 cosine'),('max_jvp','最大 JVP 误差'),('max_adjoint','最大 adjoint 误差'),('max_update','最大有限更新误差')])

    # H4: membership is exclusively the online-predicted beta, shared by all
    # paired arms. Average replicates inside each parent first. Missing groups
    # are displayed rather than interpreted as zero effect.
    case_map = {(r['method'],r['case_id']):r for r in rows}
    regime_table, regime_parent = [], []
    for group in ('NOISE_FREE','FEEDBACK_GE_NOISE','FEEDBACK_LT_NOISE'):
        selected = [r for r in rows if r['method']=='N2' and r['feedback_regime']==group]
        for pair, control in (('N2_vs_N1','N1'),('N2_vs_N3','N3')):
            grouped = defaultdict(list)
            for r in selected:
                c = case_map[(control,r['case_id'])]
                b,a = float(c['delta_nrmse']),float(r['delta_nrmse'])
                grouped[(r['parent_id'],r['family'])].append((b-a)/b)
            gains = []
            for (p,f), values in grouped.items():
                v = float(np.mean(values)); gains.append(v)
                regime_parent.append({'regime':group,'pair':pair,'parent_id':p,'family':f,'conditions':len(values),'parent_gain':v})
            regime_table.append({'regime':group,'pair':pair,'observations':len(selected),'parents':len(gains),
                                 'gain_percent':100*float(np.mean(gains)) if gains else None})
    if regime_parent:
        with (out/'FEEDBACK_REGIME_PARENT.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(regime_parent[0]));w.writeheader();w.writerows(regime_parent)
    write_json(out/'FEEDBACK_REGIME_SUMMARY.json',{'scope':'EXPOSED_DEVELOPMENT_ASSOCIATION_ONLY','selection':'ONLINE_BETA_NOT_TRUTH','rows':regime_table})
    regime_md = table(regime_table,[('regime','在线分组'),('pair','配对'),('observations','观测数'),('parents','parent 数'),('gain_percent','平均 parent 改善 %')])
    noisy_beta = [float(r['feedback_noise_ratio']) for r in rows if r['method']=='N2' and r['feedback_regime']!='NOISE_FREE']
    regime_note = (f'64个带噪观测的在线 beta 全部小于1，范围 {min(noisy_beta):.6g}–{max(noisy_beta):.6g}，中位数 {float(np.median(noisy_beta)):.6g}。因此本实验没有 FEEDBACK_GE_NOISE 样本，H4的高/低信噪对比不可识别；不能宣称已经验证或否定高反馈信号 regime。共同廉价初值的材料激发较弱，是一个待检验的限制，不能用事后新增强激发对象改变本次主结果。')

    runtime_rows = []
    action_rows = []
    warm = json.loads((out/'WARM_RUNTIME_ACTIONS.json').read_text())
    for method in config['methods']:
        cold = json.loads((out/f'PROCESS_COLD_{method}.json').read_text())
        local = [r for r in warm if r['method']==method]
        keys = sorted(set(k for r in local for k in r['online_counts']))
        for key in keys:
            values = [r['online_counts'].get(key,0) for r in local]
            action_rows.append({'method':method,'counter':key,'median':float(np.median(values)), 'minimum':min(values), 'maximum':max(values)})
        runtime_rows.append(dict(method=method,warm_ms=1000*run['warm_median_seconds'][method],
                                 encoder_ms=1000*float(np.median([r['encoder_seconds'] for r in local])),
                                 cnn_ms=1000*float(np.median([r['cnn_box_seconds'] for r in local])),
                                 audited_ms=1000*float(np.median([r['full_audited_seconds'] for r in local])),
                                 process_cold=cold['rows'][0]['process_start_to_image_seconds']))
    write_json(out/'ONLINE_ACTION_SUMMARY.json',{'rows':action_rows,'nested_counters_not_additive':True})
    runtime_md = table(runtime_rows,[('method','方法'),('warm_ms','完整 warm ms'),('encoder_ms','encoder ms'),('cnn_ms','CNN+box ms'),('audited_ms','含独立 full 审核 ms'),('process_cold','新进程首图 s')])
    write_json(out/'RUNTIME_COMPLETION.json',{'primary_warm_summary_unchanged':True,'fresh_interpreter_methods':['N1','N2','N3'],'disk_cache':'UNCONTROLLED','measured_rows':runtime_rows})

    history = budget_history(root)
    primary_receipt = next(r for r in history['receipts'] if r['job']=='a26-primary-training-001')
    assert primary_receipt['counts']['scientific_training_updates']==6000
    assert all(manifests[m]['physical_backward_calls']==0 for m in config['methods'])
    write_json(out/'TRAINING_BACKWARD_AUDIT.json',{
        'scientific_updates':6000,'actual_backward_spans':primary_receipt['counts']['autograd_backward_calls'],
        'manifest_physical_backward_calls':{m:manifests[m]['physical_backward_calls'] for m in config['methods']},
        'receipt_has_explicit_physical_zero':False,
        'evidence':'Actual cache-only update code and unit test replace all Maxwell/encoder entries with failures; features and base have no autograd graph. Native training receipt records 6000 backward calls and no physical operations. Its absent physical keys are not edited into explicit zeros.',
        'training_loss_physics_calls':0,'full_physics_evaluation_outside_training':True})
    tests_path = root/'results/a26/jobs/a26-tests-final-001/RESULT.json'
    tests = json.loads(tests_path.read_text()) if tests_path.exists() else {}
    test_log = tests_path.parent/'TESTS.log'
    skipped = test_log.read_text().count('... skipped ') if test_log.exists() else None
    failures = []
    for r in history['receipts']:
        if r['status']!='COMPLETE':
            failures.append({'type':'JOB_FAILURE','job':r['job'],'status':r['status'],'error':r.get('error'),
                             'charged_cpu_seconds':r['charged_cpu_seconds'],'gpu_occupation_seconds':r['gpu_occupation_seconds'],
                             'stderr':f'runs/{r["job"]}.stderr','source':'accepted terminal receipt'})
    for phase in ('INITIAL','MIDDLE','FINAL'):
        for r in [r for r in diagnostics if r['phase']==phase and not r.get('strict_A25_task_gate_pass')]:
            failures.append({'type':'PHYSICS_TASK_ACCURACY_FAILURE','phase':phase,'method':r['method'],
                             'case_id':r['case_id'],'gradient_cosine':r['gradient_cosine'],
                             'boxed_update_relative_error':r['boxed_update_relative_error'],
                             'numerically_valid':True,'scope':'OFFLINE_PREDICTED_STATE_AUDIT'})
    (root/'FAILURE_LEDGER.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in failures),encoding='utf-8')
    write_json(root/'A26_SOURCE_MANIFEST.json',{
        'source_A25_commit':config['source_commit'],'A24_commit':'ca80ca2a29867e1b03589061dd5c17688c9acbe0',
        'protocol':config['protocol_id'],'new_source':'src/a26','physics_backend':'unchanged A24 2D TM A2 scalar VIE',
        'new_labels':0,'training_parents':64,'exposed_validation_parents':16,
        'locked_ID_OOD':'NOT_OPENED_NOT_PUBLISHED','precision':'complex128/float64 physics; float32 CNN',
        'seed':261009,'checkpoints_per_method':['best','middle','final'],'git_release_commit':'Use actual Git commit / GitHub release target',
        'hash_checks':False,'independent_statistical_unit':'parent','publication_status':'TO_BE_VERIFIED_SEPARATELY'})

    baseline_notice = 'N1 与 N3 的无 NN 材料输出完全相同：两者使用相同 L 更新，差异仅在 OM/BP 可观测特征。N2 使用一次 TASK_R32 更新。A25 缓存 TV 初值仅是额外离线参考；它不是完整 TV 成像轨迹，更不是 SOM-Net 官方基线。'
    scope = '所有推断限于已知背景、二维 TM 24×24、固定复相位材料增量与已暴露开发验证。只有一个训练 seed、一个训练规模和一种 CNN 深度；不支持独立泛化、任意复材料、3D、样本效率或层数减少结论。'
    stopped = '冻结扩展门槛 NO_GO。第二 seed、locked ID/OOD、额外架构搜索、soft weak-space 消融、decoder 后物理修正均 NOT_RUN；locked cohort 未消费。'
    fig = 'results/a26/evaluation/261009/figures/'
    docs = {}
    docs['A26_COMPLEX_SHAPE_RESULTS.md'] = f'''# A26 复杂材料的真实训练成像结果

CNN 结构先验在复杂三家族中改善各自物理初值约 23%–24%，但剩余材料误差仍很大。下面的数字先平均每 parent 的八个背景/幅度/噪声条件，再等权汇总；重复观测不是独立对象。

{family_md}

{pair_md}

每家族四个独立 parent，复杂主比较共十二个。2000 次 paired parent bootstrap 是开发证据，不是盲测确认。完整 χ、SSIM、IoU、edge、实虚部和 full Maxwell 残差见 [逐 parent 表]({out.relative_to(root)}/PER_PARENT_METRICS.csv) 与 [全部条件]({out.relative_to(root)}/PER_CASE_METRICS.csv)。support 阈值为离线真值增量最大幅度的 25%，仅用于评估；edge 为四邻域边界的精确像素 F1，无匹配容忍度。passivity 与 box 可行性来自输出投影，不构成唯一恢复证据。

![真实实部重建]({fig}reconstruction_real.png)

![真实虚部重建]({fig}reconstruction_imag.png)

{baseline_notice}

{scope}
'''
    docs['A26_FEEDBACK_INCREMENTAL_VALUE.md'] = f'''# A26 显式反馈的增量与失败机制

N2 没有在复杂材料中优于两个匹配对照。它在 smooth 有较明显收益，在 asymmetric 只有约 1%–1.5%，shell/mixed-scale 未稳定改善。不能以 smooth 平均改善替代复杂结构主门槛。

{pair_md}

H4 分组使用共同在线初值的 R−L 数据变化除以已知噪声 RMS；无噪声独立列出。分组不读取真值，不重新选择 rank。

{regime_md}

{regime_note}

这些是 parent 内平均后的描述性关联，不能将关联解释为反馈因果机制或新增扩展依据。

{diagnostic_md}

初始化处 TASK_R32 的独立方向审核较准确。中期/最终预测材料处偶有较大的梯度方向和有限材料更新偏差，尽管 forward/JVP/adjoint 相对误差可能仍很小：接近驻点时残差方向与约束投影会放大更新差异。完整 core margins、full solve residual、paired adjoint dot defect 与底层指标都保留。不能把安全 small core 或小 forward 误差等同于材料成像准确。

本主实验没有在中期/最终图像执行反馈：encoder 只在共同初值运行并被冻结。因此后续审核是未来 learned-state correction 的风险证据，不是对当前负结果的唯一因果解释。当前候选没有提供足够不同的物理结构初值，新增特征也未减轻 CNN 对复杂边界的学习负担；这一解释仍需独立机制验证。

N2 的 parent 配对 full nonlinear data residual 相对 N1 恶化 7.8504%，相对 N3 恶化 8.9876%，超过冻结 5% 非劣界。

{stopped}
'''
    docs['A26_FINAL_DECISION.md'] = f'''# A26 最终科学裁决

**NO_GO：停止当前 N2 feedback 扩展，保留更简单的 BP + CNN 作为下一次独立验证候选。** 发布是否成功另行核验，不改变科学裁决。

| 假设 | 本实验判断 |
| --- | --- |
| H1 learned structural prior | 支持：三个 CNN 相对各自无 NN 初值在复杂三家族改善约 23%–24%；属于开发证据 |
| H2 OM encoder 增量 | 未建立：N1 相对 BP N3 的复杂材料配对改善约 −0.21%，区间跨零 |
| H3 explicit feedback 增量 | 未建立：N2 对 N1/N3 复杂材料改善均略为负，区间跨零，数据残差非劣门槛也失败 |
| H4 feedback 高于噪声 regime | 未识别：全部带噪观测 beta<1，没有高反馈信号组；clean 另列 |
| H5 quality–time tradeoff | 失败：N2 完整 warm 推理为较快对照 {run['N2_over_faster_control']:.3f} 倍，没有复杂材料质量补偿 |

{pair_md}

{runtime_md}

{stopped}

{scope}

建议首先独立验证廉价 BP + residual CNN 的 prior-dependent 成像收益；若继续反馈研究，应先提出能实际改变 shell/asymmetric 材料初始化或只用一次 decoder 后物理校正的可证伪假设，公平计费。此处只是下一实验建议，没有启动。SOM-Net/TV 强基线、更多 seeds、sample-efficiency、OOD 均没有完成，不能声称已击败这些方法。
'''
    report = f'''# A26 中文研究报告：OM 物理编码、显式反馈与轻量结构先验

## 背景与研究问题

逆散射材料成像同时受测量可识别性和结构歧义约束。物理求解器能够把一个材料假设映射为散射数据，却不保证有限视角、噪声下的反问题唯一；CNN 可以补充形状先验，也可能生成数据未充分支持的边界。实际目标是用相同信息与参数预算判断物理表示是否减少必须交给先验的结构误差，并计量从原始测量到图像的完整耗时。

[SOM-Net 原论文](https://arxiv.org/abs/2209.03567) 将确定性诱导电流和 backpropagation 材料估计输入学习流程，通过展开 SOM 更新与物理一致性约束构造网络。这里借鉴物理表示与材料更新思路，研究一个更小的冻结 encoder + 五层 residual CNN。没有训练官方 SOM-Net 或忠实 N0，因此不声称优于 SOM-Net。源、材料、背景和传播资产来自 [冻结 A25 报告](https://github.com/migodam/a25-adaptive-om-feedback/blob/{config['source_commit']}/A25_RESEARCH_REPORT_ZH.md)，其失败与 HOLD 结论没有被覆盖。

核心问题是：显式 Maxwell feedback 是否提供对复杂材料有用、且抵得过其实际计算成本的初始化/特征，减轻轻量先验的结构恢复任务？

## 实际完成与结论

三个模型各完成 **2000 次真实优化更新**，每个 **31,810 参数**，共 6000 次主训练更新；best/middle/final checkpoint 全部保存。使用 XINAN RTX 4060 Laptop/8GB。128 个开发观测（16 parents）上的全部配对完整、数值失败为零；独立 predicted-state 物理精度不足仍单独保留。没有新增真值标签，没有读取 locked ID/OOD。

CNN 先验对三个无 NN 初值都有效，但 OM 的专门特征和一次显式反馈未在复杂材料中建立独立增量。N2 完整 warm 推理约为更快对照的 **{run['N2_over_faster_control']:.3f} 倍**。冻结开发扩展门槛 **NO_GO**，停止第二 seed 和 locked 扩展。总体 Δχ NRMSE 仍约 60%，asymmetric 约 73%–74%，不能把较好看的图像当成精确或唯一恢复。

## 物理模型、坐标和信息边界

复用 A24 已验证的二维 TM 标量 VIE 后端；没有修改 Maxwell solver、Green、本构或 receiver。材料为完整 24×24 复网格；实质量坐标为 `[Re576, Im576]×sqrt(cell area)`，cell area=1/576，因此共 1152 维。8 sources/32 receivers 的数据逐源 `[Re32, Im32]`，共 512 维实数据。物理计算 complex128/float64，CNN float32。所有 sources 对应同一个材料图。

已知背景下，令 h=χ−χ_b，e_b 为背景场，G_b 为背景传播，D_b 为背景至接收数据映射。材料相对电流满足 w=diag(h)(e_b+G_b w)，数据增量为 D_b w。L 忽略新增 relative feedback，用 w=diag(h)e_b；R 使用冻结 TASK_R32 的 Woodbury closure。该相对表达是解析模型结构，不自动保证压缩准确或成像收益。

共同初值由背景 material adjoint 和保存的背景谱步长构造后投影到 box。N1/N3 再用同一个 L 梯度做一次材料更新；N2 用 R 梯度替代。OM 观测特征由 **D_b** 的 rank16 截断 SVD 构造；N3 使用普通背景 BP/adjoint 特征。M 和反馈只由共同在线初值驱动，未知 χ、truth-state J、full GN、teacher currents 从不进入 encoder。背景 J 合法，因为 χ_b 已知，并独立核对其算子与真实 mass/packing。

12 个匹配通道包括材料初值实虚部、OM/BP 观测实虚部、残差梯度实虚部、跨源场/激发功率、背景实虚部和 confidence 对角图。软 confidence 来自已知背景 Jacobian 谱，权重 s²/(s²+(.05s_max)²)。复 Gaussian 噪声的实虚方差为 σ²/2，白化比例 sqrt(2)/σ；该相对谱阈值中尺度消去。无噪声另用背景尺度归一化。它是数值稳定性指标，不是物理 recoverability 证书。主 CNN 可以修正整个网格，没有强制弱空间投影，也没有 decoder 后物理更新。

在线读取测量、已知背景、几何和噪声；trainer 独立读取标签。NPZ 真实路径、split、parent 白名单在数值 payload 前验证，混合/伪装 locked 数据拒读。发布复现包仅含 train/已暴露 validation，不含未打开的 locked payload。

## 训练、公平性和预算冻结

64 train parents，16 validation parents，四家族等量；每 parent 两背景×两幅度×clean/20dB，共 512/128 观测。重复条件没有跨 split。现有标签的增量具有固定复相位，不能推广到独立变化的任意实/虚材料。没有新数据 campaign。

三个 CNN 同为五层、宽32、dilation=1/2/4/2/1、GELU、零初始化输出头；输入均12通道，输出共同材料残差再执行 Reχ∈[-.5,4]、Imχ∈[0,.5] box。共享 train-only pooled 归一化、初始权重、parent 采样顺序和材料 RMS。每 batch 四家族各一个 parent，循环全部观测。

AdamW lr=.001、weight decay=1e−5、gradient clip=1，seed261009；loss 为归一化材料 MSE + .05×(1−SSIM)。物理 encoder 冻结并预缓存，训练 loss 没有每臂不同的物理模型项。每100更新用16个 parent 等权 Δχ NRMSE 验证并选 best，middle 为1000步，另保存 final。真实缓存 microbenchmark 在 1.3 成本余量及评估/报告预留下选择2000更新；microbenchmark 权重/optimizer 丢弃重置。更新数只按成本选择，未按质量调长训练。

6000 次 decoder backward 不调用物理算子、LU 或 feedback。实际 training manifest、cache-only code、接受账本的6000 backward spans，以及将 Maxwell/encoder 入口替换为失败的可执行测试共同支持这一点。原 native receipt 没有显式 physical-zero 键，保持原样；不事后改账本补造零计数。新图仍需重算 encoder，cached decoder 时间不能冒充部署 latency。

## 材料成像与先验收益

{family_md}

{pair_md}

正改善定义为 `(对照误差−候选误差)/对照误差`，先平均每 parent 内八条件，再平均 parent 配对改善。bootstrap 2000 次重采样独立 parent；复杂主门槛只包括 shell/asymmetric/mixed-scale，共12 parents。四个 smooth parents 单独公开，没有用它们替代复杂材料门槛。因为验证集已暴露且用于 checkpoint 选择，区间只描述开发稳定性。

N1/N2/N3 相对自身物理初值改善约 23%–24%，H1 有开发证据。N1 对 BP N3 没有复杂结构优势，H2 未建立。N2 对两对照的复杂增量略为负且区间跨零，H3 未建立；完整数据残差分别恶化约7.85%/8.99%，也未通过非劣门槛。所有图像与失败都计入，没有 success-only 筛选。

{baseline_notice}

![按家族配对增量]({fig}paired_family_gains.png)

![实际材料增量实部]({fig}reconstruction_delta_real.png)

![实际材料增量虚部]({fig}reconstruction_delta_imag.png)

图像固定为每家族首个 validation parent、两背景、高幅度、20dB；不是根据效果选择的对象。图中已经减去已知背景，同时公开完整 χ 图。直接目视审核发现：首个 shell 的内部空腔在三个 CNN 中均被部分填平；首个 smooth 的 ring-background 预测呈环状中心空缺，而真值是实心平滑峰；asymmetric 的离散边界与小部件被模糊。这些例子显示结构先验同时改善定位和引入偏置，不能将好看的轮廓认作真值认证。实部、虚部使用同一行共享色标，完整数组和误差图公开。剩余结构主要仍依赖学习分布，不能声称唯一恢复。

完整指标还包括 fullχ/Δχ NRMSE、SSIM、support IoU、四邻域 edge F1、实虚部误差、passivity、完整 nonlinear Maxwell data residual。详细逐条件 CSV 与逐 parent 表均可复算；fullχ 误差可能被较大的已知 ring 背景稀释，主指标始终是 Δχ。

## Feedback 精度与噪声机制

{diagnostic_md}

16 个固定代表条件在共同初值处审核；三个模型各自1000步/最终预测上再做96次独立 full 物理审核。JVP/VJP 比较使用预冻结独立随机方向，core与full backward residual 都留档。TASK_R32 初始化审核通过，但较晚 predicted states 的最小梯度 cosine 约 .95，有限 box 更新可偏差约33%–38%。这些精度不足没有被 hidden jitter、伪逆、full fallback 或删除对象掩盖。

主 encoder 固定在初值，因此 learned-state 风险不能解释为该主训练每一步都用了坏梯度。它否决的是未经新验证就把 TASK_R32 扩展成 learned-state recurrence 的假设。解析伴随配对通过与压缩模型对完整物理准确是两个不同判断。

{regime_md}

{regime_note}

H4 的 beta 只使用在线预测 R−L 数据差和已知噪声，clean 单独列出。这是输入分布限制，不用它事后改变 gates。

## 从原始测量到材料图像的完整成本

{runtime_md}

每方法48次完整 resident warm 图像计时，包含 encoder、搬运、CNN、box 与输出转移；三个方法轮换顺序、每图材料状态重置。另保留强制逐算子日志的 instrumented 计时、完整独立 full 审核成本、fresh encoder/checkpoint 成本、新 Python 进程首图成本。操作系统磁盘缓存未控制，不能称 disk-cold benchmark。fresh-process 各臂只有一次，不用它声称显著提速。

N2 大部分额外时间花在物理 encoder。新图 TASK_R32 closure、paired adjoint、core assembly/solve 都真实执行，并计数；rank32或少参数不等于完整提速。N2 warm约126ms，而两个对照约39–41ms，质量没有补偿，因此 H5 失败。完整在线算子逐次账本与统计在 ONLINE_ACTION_SUMMARY.json；不同嵌套 counter 有重叠，不相加伪造总 action。

![真实 loss/validation 曲线]({fig}loss_curves.png)

![质量与完整耗时]({fig}quality_vs_full_runtime.png)

## 数值测试、失败与费用

最终回归测试记录：{tests.get('tests','NOT_RUN')}项运行，{tests.get('errors','NOT_RUN')} errors、{tests.get('failures','NOT_RUN')} failures；skipped={skipped}，旧独立 Clarabel 测试因远端缺少该可选依赖跳过。保留 A24/A25 原测试及新增数据读取前隔离、background packing/adjoint、源置换、cache、预算、checkpoint、相同参数/采样、metrics 与禁止 backward 调物理入口测试。

运行故障包括早期选择无 NumPy 的本地解释器、测试 active-ledger 包装缺失、source deploy 缺少根配置、CUDA deterministic CuBLAS workspace 缺失，以及一次 N3 新进程在 Python json 启动处异常。均留账后修复入口或在不变配置下必要重跑；沒有修改数学容差、关闭确定性或按失败删除材料对象。物理 task 精度失败与系统失败分别记录。测试通过、物理代数恒等式正确和发布成功均不构成算法优势。

报告生成前已接受预算费用：CPU {history['cpu_seconds']:.3f}s，GPU occupation {history['gpu_seconds']:.3f}s。报告/图表/发布后续工作继续单独计费；终端 receipts、COST_LEDGER 和 A26_COST_AND_RUNTIME.md 的显式 cutoff 是权威，不能将本段前置 snapshot 当最终总计。CPU cap7200s、GPU cap14400s；GPU host CPU 另计，旧 A25 费用只作为历史来源、不重复扣费。新增 labels=0。

## Gates、证据分层与下一步

开发扩展门槛保持冻结：复杂材料改善≥10%且bootstrap支持正方向、至少两复杂家族改善、任一家族恶化≤5%、完整数据残差恶化≤5%、N2 warm≤快对照1.25倍、配对完整且数值有效。主 gate 为 **{gate['primary_gate']}**。{stopped}

| 证据层 | 已得到的证据 | 不支持的结论 |
| --- | --- | --- |
| 解析模型 | background-relative Maxwell identity、paired derivative/adjoint | OM 名称不证明成像优越 |
| 数值/unit | packing、known-bg J、源置换、core/residual、缓存与隔离测试 | tiny/tests 不等于真实材料收益 |
| 实际二维 Maxwell + NN | 三模型真实训练、完整开发配对、图像、预测态 full 审核 | 不推广到3D、独立测试或任意复相位 |
| Oracle/offline | 标签仅用于监督/评估，full solver 仅在预测材料处独立审核 | 不作为 online feature 或免费部署成本 |
| 部署 | 原始测量→图像完整计时和 actions | cached CNN、rank、参数量不能代替 time-to-image |

H1支持，H2/H3未建立，H4因缺少高 beta 组而不可识别，H5负向；最终 **NO_GO 当前反馈扩展**。较简单 BP + CNN 获得相似先验收益与更低成本，值得下一实验独立验证。不能宣称一般 Maxwell feedback 无用：这里测试的是固定 TASK_R32、共同一次物理更新、一个小 CNN/seed/已暴露分布。也不能自动进入更大 NN、更多 rank、SOM/3D/locked campaign。

## 公开证据与复现

[仓库]({REPO}) · [研究报告]({REPO}/blob/main/A26_RESEARCH_REPORT_ZH.md) · [匿名 raw](https://raw.githubusercontent.com/migodam/a26-om-feedback-prior-completion/main/A26_RESEARCH_REPORT_ZH.md) · [Release]({REPO}/releases/tag/a26-v0.1.0)。实际发布状态由 PUBLICATION_STATUS.json 和 release target commit 独立核验。

配置、源码、CSV、图和失败证据在 Git；真实 checkpoints、原始数组、train/已暴露 validation 复现输入和完整 action 账本在 Release。按 [复现说明](REPRODUCING_A26.md) 执行；不需要私人 SSH 配置，也不需要新标签生成。
'''
    docs['A26_RESEARCH_REPORT_ZH.md'] = report
    docs['A26_START_HERE.md'] = f'''# A26：真实训练已完成，当前 feedback 扩展 NO_GO

N1/N2/N3 各2000更新、31,810参数。复杂材料 CNN 先验收益约23%–24%，OM/feedback 对匹配 BP/无反馈对照的独立优势未建立；N2 warm慢 {run['N2_over_faster_control']:.3f} 倍。第二seed和locked ID/OOD未运行。

- [完整中文研究报告](A26_RESEARCH_REPORT_ZH.md)
- [科学裁决](A26_FINAL_DECISION.md) / [冻结 gates](GATE_DECISION.json)
- [网络与实际 checkpoint 记录](A26_NETWORK_COMPARISON.md)
- [复杂结构结果](A26_COMPLEX_SHAPE_RESULTS.md)
- [反馈增量](A26_FEEDBACK_INCREMENTAL_VALUE.md)
- [物理/先验边界](A26_PHYSICS_PRIOR_SPLIT.md)
- [完整成本](A26_COST_AND_RUNTIME.md)
- [探索与失败](A26_EXPLORATION_LEDGER.md)
- [复现说明](REPRODUCING_A26.md)

{scope}

[GitHub]({REPO}) · [匿名 raw 报告](https://raw.githubusercontent.com/migodam/a26-om-feedback-prior-completion/main/A26_RESEARCH_REPORT_ZH.md) · [实际 checkpoint/数据 Release]({REPO}/releases/tag/a26-v0.1.0)
'''
    docs['README.md'] = f'''# A26 — OM Feedback-Guided Prior Completion

**Three real CNNs trained; current feedback extension: NO_GO.** Known-background 2D TM, 24×24 complex full-grid material, RTX4060 Laptop. No new labels and no locked ID/OOD opening.

The learned structural prior reduces complex-family material error by approximately 23–24% over its physical inputs. At matched 31,810 parameters and 2000 updates, explicit TASK_R32 feedback does not improve complex-family reconstruction over either matched control. Its complete warm inference is {run['N2_over_faster_control']:.3f}× slower than the faster control. These are exposed-development results, not an independent generalization claim.

{pair_md}

- [完整中文研究报告](A26_RESEARCH_REPORT_ZH.md)
- [研究入口](A26_START_HERE.md)
- [Final scientific decision](A26_FINAL_DECISION.md) and [machine-readable gates](GATE_DECISION.json)
- [All case metrics](results/a26/evaluation/261009/PER_CASE_METRICS.csv), [parent metrics](results/a26/evaluation/261009/PER_PARENT_METRICS.csv), [physics diagnostics](results/a26/evaluation/261009/PHYSICS_DIAGNOSTICS.json)
- [Reproduce using public cached inputs](REPRODUCING_A26.md)
- [Frozen protocol](research/A26_FROZEN_PROTOCOL.md), [configuration](configs/a26/pilot.json), [source provenance](A26_SOURCE_MANIFEST.json)
- [Actual model/data/raw-evidence Release]({REPO}/releases/tag/a26-v0.1.0)

![Actual reconstruction of material change]({fig}reconstruction_delta_real.png)

![Complete quality–time comparison]({fig}quality_vs_full_runtime.png)

The archived A24/A25 files remain unchanged as historical controls. Their previous results are not A26 evidence. Main source: A25 commit `{config['source_commit']}`. This repository does not publish unopened locked payloads or credentials. Physics identity, tests, material quality and deployment speed are separate judgments. No official SOM-Net superiority, arbitrary-complex-material recovery, sample-efficiency or 3D claim is made.
'''
    for name,text in docs.items():
        book.check()
        (root/name).write_text(text.strip()+'\n',encoding='utf-8')
    return {'scientific_gate':gate['primary_gate'],'reports':list(docs),'second_seed':'NOT_RUN_GATE_FAILED','locked':'NOT_OPENED'}
