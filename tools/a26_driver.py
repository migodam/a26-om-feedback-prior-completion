"""Foreground paid A26 stage driver; no automatic experiment expansion."""
from __future__ import annotations
import time
START_WALL, START_CPU = time.perf_counter(), time.process_time()
import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

def default_data_root():
    return Path('D:/AI/A25_ADAPTIVE_OM_FEEDBACK/data') if os.name == 'nt' else ROOT.parents[1] / 'A25/a25-adaptive-om-feedback/data'

def metadata(arrays):
    import numpy as np
    keys = ('case_id','parent_id','family','background_id','sigma','noise_db','split','variant_id')
    return {k:np.asarray(arrays[k]).copy() for k in keys if k in arrays}

def cache_features(root, data_root, config, book, device):
    import numpy as np
    import torch
    from a26.data import load_split
    from a26.encoder import load_encoders
    encoders = load_encoders(data_root,config,book,device=device)
    out=root/'results/a26/cache';out.mkdir(parents=True,exist_ok=True)
    summary={}
    with torch.no_grad():
        for role in ('train','validation'):
            arrays=load_split(data_root,role)
            with book.span('a26_observation_only_repro_export'):
                np.savez_compressed(out/(role+'_ONLINE_MEASUREMENTS.npz'),data=arrays['data'],sigma=arrays['sigma'],
                    background_id=arrays['background_id'],case_id=arrays['case_id'])
            for method in config['methods']:
                features=[];base=[];initial=[];beta=[];diags=[]
                # Preserve original row order across backgrounds and methods.
                for start in range(0,len(arrays['data']),8):
                    book.check()
                    row_count=min(8,len(arrays['data'])-start)
                    local=[None]*row_count
                    for bg,enc in encoders.items():
                        idx=np.flatnonzero(np.asarray(arrays['background_id'][start:start+row_count]).astype(str)==bg)
                        if not len(idx):continue
                        absolute=start+idx
                        obs=torch.tensor(arrays['data'][absolute],dtype=torch.complex128,device=device)
                        sigma=torch.tensor(arrays['sigma'][absolute],dtype=torch.float64,device=device)
                        result=enc.encode(obs,sigma,method)
                        for j,k in enumerate(idx):
                            local[k]={key:result[key][j].detach().cpu().numpy() for key in ('features','base_mass','initial_mass','beta')}
                        diags.append({'first_row':start,'background':bg,'result':result['diagnostics']})
                    if any(x is None for x in local):raise RuntimeError('UNREGISTERED_BACKGROUND_ROW')
                    features.extend(x['features'] for x in local);base.extend(x['base_mass'] for x in local)
                    initial.extend(x['initial_mass'] for x in local);beta.extend(x['beta'] for x in local)
                with book.span('feature_cache_export',feature_cache_rows=len(features)):
                    target=out/(role+'_'+method+'.npz')
                    if target.exists():raise RuntimeError('IMMUTABLE_FEATURE_CACHE_EXISTS')
                    from a26.training import write_feature_cache
                    values=dict(features=np.stack(features),base_mass=np.stack(base),initial_mass=np.stack(initial),beta=np.asarray(beta),**metadata(arrays))
                    write_feature_cache(target,values,method=method,role=role,config=config)
                    from a26.budget import write_json
                    write_json(out/(role+'_'+method+'_DIAGNOSTICS.json'),diags)
                summary[role+'_'+method]={'rows':len(features),'path':target.relative_to(root).as_posix()}
                print(json.dumps({'event':'CACHE_COMPLETE','split':role,'method':method,'rows':len(features)}),flush=True)
    for name, enc in encoders.items():
        enc.save_diagnostics(out/(name+'_BACKGROUND_DIAGNOSTICS.npz'))
    return {'status':'COMPLETE','caches':summary,'encoder_target_access':False,'new_labels':0}

def tests_stage(root,book):
    import unittest
    from a24.contracts import active_ledger
    with active_ledger(book),book.span('source_unit_tests'):
        suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_*.py')
        path=root/'results/a26/jobs'/book.metadata['job']/'TESTS.log'
        with path.open('w',encoding='utf-8') as log:
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        output={'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'passed':result.wasSuccessful()}
        if not result.wasSuccessful():raise RuntimeError('UNIT_TESTS_FAILED:'+json.dumps(output))
    return output

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default=str(ROOT));parser.add_argument('--data-root',default=str(default_data_root()))
    parser.add_argument('--config',default='configs/a26/pilot.json');parser.add_argument('--job',required=True)
    parser.add_argument('--stage',required=True,choices=('audit','cache','benchmark','train','evaluate','timing','process-cold','export','locked','tests','report'))
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');parser.add_argument('--seed',type=int)
    args=parser.parse_args();root=Path(args.root).resolve();data_root=Path(args.data_root)
    config=json.loads((root/args.config).read_text())
    if args.device=='cuda':
        from a25.cuda_runtime import configure_host_wait
        configure_host_wait('blocking_sync')
    from a26.budget import charged_job,write_json
    with charged_job(root,args.job,device=args.device,config=config,metadata={'stage':args.stage}) as book:
        # Include this process's import/configuration tail in the native receipt.
        book.started_wall,book.started_cpu=START_WALL,START_CPU
        book.check()
        try:
            if args.stage=='tests':result=tests_stage(root,book)
            elif args.stage=='audit':
                from a26.data import load_split
                result={'status':'COMPLETE','splits':{}}
                for role in ('train','validation'):
                    a=load_split(data_root,role)
                    result['splits'][role]={'rows':len(a['data']),'parents':len(set(map(int,a['parent_id']))),'labels_shape':list(a['chi'].shape),'online_target_separated':True}
                result['locked_opened']=False
            elif args.stage=='cache':result=cache_features(root,data_root,config,book,args.device)
            elif args.stage in ('benchmark','train'):
                from a26.campaign_training import benchmark_stage,train_stage
                result=(benchmark_stage if args.stage=='benchmark' else train_stage)(root,data_root,config,book,args.device,seed=args.seed)
            elif args.stage in ('evaluate','timing'):
                from a26.evaluation import evaluation_stage,timing_stage
                result=(evaluation_stage if args.stage=='evaluate' else timing_stage)(root,data_root,config,book,args.device,seed=args.seed)
            elif args.stage=='process-cold':
                from a26.evaluation import process_cold_stage
                result=process_cold_stage(root,data_root,config,book,args.device,seed=args.seed)
            elif args.stage=='export':
                from a26.export import export_stage
                result=export_stage(root,data_root,config,book)
            elif args.stage=='locked':
                from a26.locked import locked_stage
                result=locked_stage(root,data_root,config,book,args.device)
            else:
                from a26.reporting import report_stage
                from a26.scientific_reports import scientific_reports
                result=report_stage(root,config,book,scientific_callback=scientific_reports)
            write_json(root/'results/a26/jobs'/args.job/'RESULT.json',result)
            print(json.dumps({'job':args.job,'stage':args.stage,'result':result},default=str,allow_nan=False),flush=True)
        except BaseException:
            (root/'results/a26/jobs'/args.job/'ERROR.log').write_text(traceback.format_exc(),encoding='utf-8')
            raise

if __name__=='__main__':main()
