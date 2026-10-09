"""Frozen development expansion checks; no automatic train or locked open."""
from pathlib import Path
import csv,json,math
import numpy as np
from .budget import write_json

def decision(root,config,book=None,seed=261009):
    out=Path(root)/'results/a26/evaluation'/str(seed)
    stats=json.loads((out/'PAIRED_STATISTICS.json').read_text())
    run=json.loads((out/'RUNTIME_SUMMARY.json').read_text())
    summary=json.loads((out/'EVALUATION_SUMMARY.json').read_text())
    pairs={};checks={}
    for name in ('N2_vs_N1','N2_vs_N3'):
        p=stats[name];gain=p['complex_mean_gain'];ci=p['complex_ci95']
        families=p['family_mean_gains'];data=p['all_families']['full_data_relative_residual']
        checks[name]={
            'complex_gain_at_least_10pct':gain is not None and gain>=.10,
            'positive_bootstrap_lower':ci is not None and ci[0]>0,
            'two_complex_families_improve':p['positive_complex_families']>=2,
            'no_family_over_5pct_worse':all(g is not None and g>=-.05 for g in families.values()),
            'data_residual_noninferior_5pct':data['point_estimate'] is not None and data['point_estimate']>=-.05,
            'complete_parent_pairing':p['complex_coverage_complete'] and data['coverage_complete']}
        pairs[name]={'complex_relative_gain':gain,'bootstrap95':ci,'family_gains':families,
            'full_data_residual_parent_relative_gain':data['point_estimate']}
    numerical=summary['numerical_failures']==0 and summary['parents']==16 and summary['cases_per_arm']==128
    checks['numerical_valid_and_complete']=numerical
    checks['native_full_warm_latency_within_1_25']=run['N2_over_faster_control']<=1.25
    passed=all(all(v.values()) if isinstance(v,dict) else v for v in checks.values())
    result={'protocol_id':config['protocol_id'],'seed':seed,'primary_gate':'PASS' if passed else ('NO_GO' if numerical else 'HOLD'),
        'evidence_scope':'EXPOSED_DEVELOPMENT_SINGLE_TRAINING_SEED','formal_independent_validation':'NOT_RUN',
        'checks':checks,'paired_evidence':pairs,'warm_latency_ratio':run['N2_over_faster_control'],
        'repeat_authorized':passed,'repeat_seed_status':'CONDITIONALLY_AUTHORIZED' if passed else 'NOT_RUN_GATE_FAILED',
        'locked_ID_OOD':'CLOSED_NOT_OPENED','new_labels':0,'NN_training':'THREE_MODELS_ACTUALLY_TRAINED',
        'quality_time_claim':'NO_ACCELERATION' if not checks['native_full_warm_latency_within_1_25'] else 'QUALITY_MATCH_REQUIRED',
        'TASK_R32':'EXPLORATORY_NOT_UNIFORMLY_GRADIENT_CERTIFIED',
        'decision':'Proceed only to identical repeat if all preregistered checks pass; otherwise stop N2 expansion and preserve simpler comparators.'}
    write_json(out/'GATE_DECISION.json',result);write_json(Path(root)/'GATE_DECISION.json',result)
    return result
