"""Paid orchestration around the target-free cache trainer; no physics backward."""
from pathlib import Path
import json
import numpy as np
import torch
from .budget import write_json
from .training import (CachedDataset,load_feature_cache,fit_normalizer,
    benchmark_training,select_update_budget,train_one_model)
from .data import load_split


def datasets(root,data_root,config,book):
    with book.span('a26_trainer_cache_and_targets_load'):
        backgrounds={}
        for name in ('weak','ring'):
            with np.load(Path(data_root)/'background'/(name+'.npz'),allow_pickle=False) as a:
                backgrounds[name]=a['chi_b'].copy()
        result={}
        for role in ('train','validation'):
            labels=load_split(data_root,role)
            result[role]={m:CachedDataset(load_feature_cache(Path(root)/'results/a26/cache'/(role+'_'+m+'.npz'),
                role=role,method=m,config=config),labels,role=role,background_materials=backgrounds)
                for m in config['methods']}
    return result


def available(book):
    wall,process,cpu=book.elapsed()
    gpu=wall if book.accounting_device=='cuda' else 0.
    return book.cpu_limit-book.prior_cpu-cpu,book.gpu_limit-book.prior_gpu-gpu


def benchmark_stage(root,data_root,config,book,device,seed=None):
    sets=datasets(root,data_root,config,book)
    normalizer=fit_normalizer(sets['train'],config=config,book=book)
    out=Path(root)/'results/a26/learning';out.mkdir(parents=True,exist_ok=True)
    if (out/'COST_SELECTED_UPDATES.json').exists():raise RuntimeError('IMMUTABLE_COST_SELECTION_EXISTS')
    benchmark=benchmark_training(sets['train'],sets['validation'],normalizer=normalizer,
        config=config,book=book,width=32,device=device,seed=seed)
    write_json(out/'BENCHMARK_WIDTH32.json',benchmark)
    c,g=available(book)
    selected=select_update_budget(benchmark,config=config,cpu_available=c,gpu_available=g)
    if selected['status']=='INSUFFICIENT_BUDGET':
        benchmark=benchmark_training(sets['train'],sets['validation'],normalizer=normalizer,
            config=config,book=book,width=16,device=device,seed=seed)
        write_json(out/'BENCHMARK_WIDTH16.json',benchmark)
        c,g=available(book)
        selected=select_update_budget(benchmark,config=config,cpu_available=c,gpu_available=g,fallback_from=selected)
    write_json(out/'COST_SELECTED_UPDATES.json',selected)
    torch.save({'state_dict':normalizer.state_dict(),'provenance':normalizer.provenance},out/'TRAIN_ONLY_NORMALIZER.pt')
    return selected


def train_stage(root,data_root,config,book,device,seed=None):
    seed=int(config['primary_seed'] if seed is None else seed)
    if seed!=config['primary_seed']:
        gates=json.loads((Path(root)/'results/a26/evaluation'/str(config['primary_seed'])/'GATE_DECISION.json').read_text())
        if not gates.get('repeat_authorized'):raise RuntimeError('REPEAT_GATE_NOT_PASSED')
    selected=json.loads((Path(root)/'results/a26/learning/COST_SELECTED_UPDATES.json').read_text())
    if selected['status']!='COST_SELECTED':raise RuntimeError('TRAINING_DOES_NOT_FIT_BUDGET')
    selected['seed']=seed
    sets=datasets(root,data_root,config,book)
    normalizer=fit_normalizer(sets['train'],config=config,book=book)
    manifests={}
    for method in config['methods']:
        book.check()
        print(json.dumps({'event':'TRAIN_BEGIN','method':method,'seed':seed,'updates':selected['updates']}),flush=True)
        manifests[method]=train_one_model(method,sets['train'][method],sets['validation'][method],
            selection=selected,normalizer=normalizer,config=config,
            out=Path(root)/'results/a26/learning'/str(seed)/method,book=book,device=device,seed=seed)
        print(json.dumps({'event':'TRAIN_COMPLETE','method':method,'updates':manifests[method]['completed_updates'],
            'best_validation':manifests[method]['best_validation_delta_nrmse']}),flush=True)
    return {'status':'THREE_MODELS_TRAINED','seed':seed,'updates':selected['updates'],'width':selected['width'],
        'methods':{m:{k:manifests[m][k] for k in ('completed_updates','best_step','best_validation_delta_nrmse','parameter_count','physical_backward_calls')} for m in config['methods']}}
