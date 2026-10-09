"""Exposed-development imaging evidence, independent exact-Maxwell audits.

The full solver is the unchanged A24 Physics2D. Only its already frozen arrays
are restored, avoiding a second Green assembly. All full solves in this module
are offline checks at predictions; they are not labels or encoder features.
"""
from pathlib import Path
from collections import Counter
from contextlib import contextmanager
import csv
import json
import math
import time
import numpy as np
import torch
from a24.physics import Physics2D
from a24.contracts import mass_pack,mass_unpack,pack
from a25.torch_physics import pack_data
from .data import load_split
from .training import CachedDataset,load_feature_cache,load_trained_model,predict_cached
from .encoder import load_encoders,FrozenEncoder
from .budget import write_json,BudgetExceeded


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,torch.Tensor):return clean(value.detach().cpu().numpy())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow(clean(r))


class CachedFullPhysics(Physics2D):
    """Restore G/S/incident; inherit every full solve and adjoint operation."""
    def __init__(self,reference,config,book):
        self.book,self.config=book,dict(config)
        self.N=self.n=reference.n;self.grid=reference.grid;self.components=1
        self.s,self.q=reference.s,reference.q
        self.volume=reference.volume;self.sqrt_volume=reference.sqrt_volume
        with book.span('a26_restore_frozen_full_backend_arrays',a26_full_backend_cache_loads=1):
            self.G=reference.G.detach().cpu().numpy().copy()
            self.S=reference.S.detach().cpu().numpy().copy()
            self.incident=reference.incident.detach().cpu().numpy().copy()

    def jvp_full(self,state,direction):
        dchi=mass_unpack(direction,self.volume)
        rhs=dchi[:,None]*state.field
        dc=self.solve_factor(state.factor,rhs,role='full_tangent')
        return pack(self.observe(dc))

    def adjoint_full(self,state,load):
        z=self.solve_factor(state.factor,self.observe(load,adjoint=True),adjoint=True,role='full_data')
        return self.material_pullback(state.field,z)


def representative_indices(labels):
    # Frozen first parent in each family, all two backgrounds/two amplitudes,
    # 20dB only. Selection uses metadata, never reconstruction quality.
    selected=[]
    for family in ('shell','asymmetric_piecewise','smooth','mixed_scale'):
        ii=np.flatnonzero(np.asarray(labels['family']).astype(str)==family)
        parent=min(int(labels['parent_id'][i]) for i in ii)
        selected.extend(i for i in ii if int(labels['parent_id'][i])==parent and float(labels['sigma'][i])>0)
    if len(selected)!=16:raise ValueError('A26_FROZEN_16_REPRESENTATIVE_CONDITIONS_REQUIRED')
    return selected


def _dataset(root,labels,bgs,method,config):
    cache=load_feature_cache(Path(root)/'results/a26/cache'/('validation_'+method+'.npz'),
        role='validation',method=method,config=config)
    return CachedDataset(cache,labels,role='validation',background_materials=bgs)


def _metadata(labels,i):
    return {'case_id':str(labels['case_id'][i]),'parent_id':int(labels['parent_id'][i]),
        'family':str(labels['family'][i]),'background_id':str(labels['background_id'][i]),
        'noise_db':None if labels['sigma'][i]==0 else float(labels['noise_db'][i]),
        'sigma':float(labels['sigma'][i]),'variant_id':str(labels['variant_id'][i])}


def gradient_diagnostic(enc,physics,chi,obs,config,book,key):
    ref=enc.reference
    full=physics.full_state(chi)
    mass=torch.as_tensor(mass_pack(chi,ref.volume)[None],dtype=torch.float64,device=enc.device)
    measurement=torch.as_tensor(obs[None],dtype=torch.complex128,device=enc.device)
    with torch.no_grad():
        reduced=enc.models['R'](mass,recover_current=False)
        enc._forward_residual(reduced)
        gr=enc.models['R'].data_gradient(mass,measurement,reduced)[0].cpu().numpy()
        gf=physics.adjoint_full(full,(full.data-obs)/float(ref.yb_norm2))
        rng=np.random.default_rng(np.random.SeedSequence([261009,2601,int(key)]))
        d=rng.standard_normal(2*ref.n);d/=np.linalg.norm(d)
        l=rng.standard_normal(2*ref.q*ref.s);l/=np.linalg.norm(l)
        jr=enc.models['R'].jvp(mass,torch.as_tensor(d[None],dtype=torch.float64,device=enc.device),reduced)[0].cpu().numpy()
        ar=enc.models['R'].adjoint(mass,torch.as_tensor(l[None],dtype=torch.float64,device=enc.device),reduced)[0].cpu().numpy()
        jf=physics.jvp_full(full,d)
        af=physics.adjoint_full(full,physics.unpack_real(l))
        tr=reduced.data[0].cpu().numpy()
        cosine=float(gf@gr/(np.linalg.norm(gf)*np.linalg.norm(gr))) if np.linalg.norm(gf)*np.linalg.norm(gr)>0 else None
        error=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b)) if np.linalg.norm(b)>0 else None
        direct=error(tr,full.data)
        relative_scattered_error=error(tr-ref.yb.cpu().numpy(),full.data-ref.yb.cpu().numpy())
        linear=enc.models['L'](mass,recover_current=False)
        gl=enc.models['L'].data_gradient(mass,measurement,linear)[0].cpu().numpy()
        linear_data_error=error(linear.data[0].cpu().numpy(),full.data)
        linear_cosine=float(gf@gl/(np.linalg.norm(gf)*np.linalg.norm(gl))) if np.linalg.norm(gf)*np.linalg.norm(gl)>0 else None
        # Boxed finite update consequences; no new optimizer or label.
        xf=ref.project(mass-ref.step*torch.as_tensor(gf[None],dtype=torch.float64,device=enc.device))[0].cpu().numpy()
        xr=ref.project(mass-ref.step*torch.as_tensor(gr[None],dtype=torch.float64,device=enc.device))[0].cpu().numpy()
        update_error=error(xr-mass[0].cpu().numpy(),xf-mass[0].cpu().numpy())
        dot_defect=abs(float(jr@l-ar@d))/max(abs(float(jr@l)),abs(float(ar@d)),1e-30)
        margins=clean(reduced.margins)
        return {'forward_relative_error':direct,'differential_forward_relative_error':relative_scattered_error,
            'adjoint_relative_error':error(ar,af),
            'jvp_relative_error':error(jr,jf),'gradient_relative_error':error(gr,gf),
            'gradient_cosine':cosine,'reduced_adjoint_dot_defect':dot_defect,
            'boxed_update_relative_error':update_error,'core_margins':margins,
            'full_backward_residual':full.residual,'L_forward_relative_error':linear_data_error,
            'L_gradient_cosine':linear_cosine,'strict_A25_task_gate_pass':bool(
                all(v is not None and v<=.05 for v in (direct,error(ar,af),error(jr,jf))) and cosine is not None and cosine>=.99),
            'scope':'OFFLINE_AUDIT_AT_PREDICTED_MATERIAL_NO_TRUTH_STATE_J'}


def evaluation_stage(root,data_root,config,book,device,seed=None):
    from .metrics import image_metrics,paired_statistics,aggregate_rows
    seed=int(config['primary_seed'] if seed is None else seed)
    out=Path(root)/'results/a26/evaluation'/str(seed);out.mkdir(parents=True,exist_ok=True)
    if (out/'PER_CASE_METRICS.csv').exists():raise RuntimeError('IMMUTABLE_EVALUATION_EXISTS')
    labels=load_split(data_root,'validation')
    encoders=load_encoders(data_root,config,book,device=device)
    bgs={b:e.reference.chi_b.cpu().numpy().copy() for b,e in encoders.items()}
    exact={b:CachedFullPhysics(e.reference,config,book) for b,e in encoders.items()}
    data={m:_dataset(root,labels,bgs,m,config) for m in config['methods']}
    predictions={};middle={};final={}
    for method in config['methods']:
        book.check()
        folder=Path(root)/'results/a26/learning'/str(seed)/method
        cp=load_trained_model(folder/'best.pt',device=device,config=config)
        predictions[method]=predict_cached(cp.model,cp.normalizer,data[method],device=device,book=book)['chi']
        for key,dst in (('middle',middle),('final',final)):
            p=load_trained_model(folder/(key+'.pt'),device=device,config=config)
            dst[method]=predict_cached(p.model,p.normalizer,data[method],device=device,book=book)['chi']
        predictions['BASE_'+method]=(data[method].cache['base_mass'][:,:576]+1j*data[method].cache['base_mass'][:,576:])*24
    initial=data['N1'].cache['initial_mass']
    predictions['COMMON_BP_INITIAL']=(initial[:,:576]+1j*initial[:,576:])*24
    if 'initial_mass' in labels:
        im=labels['initial_mass'];predictions['A25_CERTIFIED_TV_INITIAL_OFFLINE_CACHE']=(im[:,:576]+1j*im[:,576:])*24
    np.savez_compressed(out/'PREDICTIONS.npz',**predictions,truth=labels['chi'],
        case_id=labels['case_id'],parent_id=labels['parent_id'],family=labels['family'],background_id=labels['background_id'],
        sigma=labels['sigma'],beta_N2=data['N2'].cache['beta'])
    rows=[];failures=[]
    for method,images in predictions.items():
        for i,chi in enumerate(images):
            book.check();bg=str(labels['background_id'][i]);row={**_metadata(labels,i),'method':method,'valid':True}
            try:
                with book.span('a26_offline_material_image_metrics',offline_image_metrics=1):
                    row.update(image_metrics(chi,labels['chi'][i],bgs[bg]))
                state=exact[bg].full_state(chi)
                residual=np.linalg.norm(state.data-labels['data'][i])
                denominator=np.linalg.norm(labels['data'][i]-encoders[bg].reference.yb.cpu().numpy())
                row.update(full_data_relative_residual=float(residual/denominator) if denominator>0 else None,
                    full_data_absolute_residual=float(residual),full_backward_residual=state.residual,
                    full_white_residual=float(math.sqrt(2)*residual/labels['sigma'][i]) if labels['sigma'][i]>0 else None)
            except BudgetExceeded:raise
            except Exception as exc:
                row.update(valid=False,error=type(exc).__name__+':'+str(exc));failures.append(row.copy())
            beta=float(data['N2'].cache['beta'][i]);row['feedback_noise_ratio']=beta if math.isfinite(beta) else None
            row['feedback_regime']='NOISE_FREE' if labels['sigma'][i]==0 else ('FEEDBACK_GE_NOISE' if beta>=1 else 'FEEDBACK_LT_NOISE')
            rows.append(row)
        print(json.dumps({'event':'IMAGING_EVALUATED','method':method,'cases':len(images)}),flush=True)
    write_csv(out/'PER_CASE_METRICS.csv',rows)
    parent=aggregate_rows(rows,['delta_nrmse','full_nrmse','ssim_delta','support_iou','edge_f1','real_nrmse','imag_nrmse',
        'full_data_relative_residual'],['method','parent_id','family'])
    write_csv(out/'PER_PARENT_METRICS.csv',parent)
    family=aggregate_rows(parent,['delta_nrmse','full_nrmse','ssim_delta','support_iou','edge_f1','full_data_relative_residual'],['method','family'])
    write_csv(out/'PER_FAMILY_METRICS.csv',family)
    stats=paired_statistics(parent,config);write_json(out/'PAIRED_STATISTICS.json',clean(stats))
    diagnoses=[]
    reps=representative_indices(labels)
    diagnostics={'INITIAL':{'COMMON':predictions['COMMON_BP_INITIAL']},'MIDDLE':middle,'FINAL':final}
    for phase,states in diagnostics.items():
        for method,images in states.items():
            for i in reps:
                book.check();bg=str(labels['background_id'][i]);record={**_metadata(labels,i),'phase':phase,'method':method}
                try:record.update(gradient_diagnostic(encoders[bg],exact[bg],images[i],labels['data'][i],config,book,i))
                except BudgetExceeded:raise
                except Exception as exc:record.update(status='FAILED',error=type(exc).__name__+':'+str(exc));failures.append(record.copy())
                diagnoses.append(record)
    write_json(out/'PHYSICS_DIAGNOSTICS.json',clean(diagnoses))
    write_json(out/'FAILURES.json',clean(failures))
    summary={'status':'COMPLETE_EXPOSED_DEVELOPMENT_EVALUATION','seed':seed,'cases_per_arm':128,'parents':16,
        'locked_opened':False,'new_labels':0,'paired_statistics':stats,'numerical_failures':len(failures),
        'task_accuracy_status':'EXPLORATORY_RETAIN_ALL_PRECISION_FAILURES','diagnostics':len(diagnoses)}
    write_json(out/'EVALUATION_SUMMARY.json',clean(summary))
    return clean(summary)


def _sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()


class CountOnlyOnlineBook:
    """In-memory action receipt for a bounded, outer-metered online image.

    Numerical guards stay inside the unchanged encoder. Outer paid job checks
    the budget/lock before and after each image and synchronizes timer ends.
    Per-operator disk tracing/synchronization is not a deployment operation.
    """
    def __init__(self):self.counts=Counter()
    def check(self):pass
    @contextmanager
    def span(self,label,**counts):
        self.counts.update(counts)
        yield


def timing_stage(root,data_root,config,book,device,seed=None):
    seed=int(config['primary_seed'] if seed is None else seed)
    out=Path(root)/'results/a26/evaluation'/str(seed)
    labels=load_split(data_root,'validation');reps=representative_indices(labels)
    encoders=load_encoders(data_root,config,book,device=device)
    cps={m:load_trained_model(Path(root)/'results/a26/learning'/str(seed)/m/'best.pt',device=device,config=config) for m in config['methods']}
    exact={b:CachedFullPhysics(e.reference,config,book) for b,e in encoders.items()}
    rows=[]
    # Warm-up is paid, outside warm timing. Prime lazy O setup and every arm.
    with torch.no_grad(),book.span('a26_paid_full_inference_warmup',inference_warmup_images=6):
        for bg,enc in encoders.items():
            i=next(i for i in reps if str(labels['background_id'][i])==bg)
            for method,cp in cps.items():
                v=enc.encode(labels['data'][i],float(labels['sigma'][i]),method)
                cp.predict_mass(v['features'],v['base_mass'])
        _sync(device)
    # Three rotations, actual raw-measurement inputs; no final-image reuse.
    for repeat in range(3):
        order=config['methods'][repeat:]+config['methods'][:repeat]
        for i in reps:
            for method in order:
                book.check();bg=str(labels['background_id'][i]);enc=encoders[bg];cp=cps[method]
                old=dict(book.counts);_sync(device);t=time.perf_counter()
                with torch.no_grad():
                    encoded=enc.encode(labels['data'][i],float(labels['sigma'][i]),method)
                    _sync(device);t1=time.perf_counter()
                    mass=cp.predict_mass(encoded['features'],encoded['base_mass'])
                    _sync(device);t2=time.perf_counter()
                    chi=(mass[0,:576]+1j*mass[0,576:])*24
                    image=chi.cpu().numpy();_sync(device);t3=time.perf_counter()
                online_counts={k:int(v-old.get(k,0)) for k,v in book.counts.items() if v!=old.get(k,0)}
                before_audit=dict(book.counts)
                full=exact[bg].full_state(image);_sync(device);t4=time.perf_counter()
                row={**_metadata(labels,i),'method':method,'repeat':repeat,'cache':'RESIDENT_WARM',
                    'encoder_seconds':t1-t,'cnn_box_seconds':t2-t1,'image_transfer_seconds':t3-t2,
                    'full_inference_seconds':t3-t,'offline_full_audit_seconds':t4-t3,'full_audited_seconds':t4-t,
                    'online_counts':online_counts,
                    'offline_counts':{k:int(v-before_audit.get(k,0)) for k,v in book.counts.items() if v!=before_audit.get(k,0)}}
                rows.append(row)
    cold=[]
    for method in config['methods']:
        for bg in ('weak','ring'):
            i=next(i for i in reps if str(labels['background_id'][i])==bg)
            book.check();_sync(device);t=time.perf_counter()
            enc=FrozenEncoder(Path(data_root)/'background'/(bg+'.npz'),config,book,device=device)
            cp=load_trained_model(Path(root)/'results/a26/learning'/str(seed)/method/'best.pt',device=device,config=config)
            with torch.no_grad():
                v=enc.encode(labels['data'][i],float(labels['sigma'][i]),method)
                _sync(device);t1=time.perf_counter();mass=cp.predict_mass(v['features'],v['base_mass']);_sync(device)
                image=((mass[0,:576]+1j*mass[0,576:])*24).cpu().numpy();t2=time.perf_counter()
            cold.append({**_metadata(labels,i),'method':method,'cache':'FRESH_ENCODER_AND_CHECKPOINT_OS_CACHE_UNCONTROLLED',
                'setup_and_encoder_seconds':t1-t,'full_inference_seconds':t2-t,'decoder_transfer_seconds':t2-t1,
                'fresh_process':False,'disk_cold_claim':False})
    write_csv(out/'INSTRUMENTED_RUNTIME.csv',rows);write_json(out/'INSTRUMENTED_RUNTIME_ACTIONS.json',clean(rows))
    native=[]
    for repeat in range(3):
        order=config['methods'][repeat:]+config['methods'][:repeat]
        for i in reps:
            for method in order:
                book.check();bg=str(labels['background_id'][i]);enc=encoders[bg];cp=cps[method]
                counter=CountOnlyOnlineBook();enc.bind_book(counter)
                _sync(device);t=time.perf_counter()
                try:
                    with torch.no_grad():
                        v=enc.encode(labels['data'][i],float(labels['sigma'][i]),method)
                        _sync(device);t1=time.perf_counter();mass=cp.predict_mass(v['features'],v['base_mass'])
                        _sync(device);t2=time.perf_counter();image=((mass[0,:576]+1j*mass[0,576:])*24).cpu().numpy()
                        _sync(device);t3=time.perf_counter()
                finally:
                    enc.bind_book(book);book.counts.update(counter.counts)
                before=dict(book.counts);exact[bg].full_state(image);_sync(device);t4=time.perf_counter()
                row={**_metadata(labels,i),'method':method,'repeat':repeat,'cache':'RESIDENT_WARM_COUNTED_NATIVE',
                    'encoder_seconds':t1-t,'cnn_box_seconds':t2-t1,'image_transfer_seconds':t3-t2,
                    'full_inference_seconds':t3-t,'offline_full_audit_seconds':t4-t3,'full_audited_seconds':t4-t,
                    'online_counts':dict(counter.counts),'offline_counts':{k:int(v-before.get(k,0)) for k,v in book.counts.items() if v!=before.get(k,0)}}
                book.append({'event':'COUNTED_NATIVE_ONLINE_IMAGE','status':'OK','method':method,
                    'case_id':row['case_id'],'counters':dict(counter.counts),'wall_seconds':t3-t})
                native.append(row);book.check()
    write_csv(out/'WARM_RUNTIME.csv',native);write_json(out/'WARM_RUNTIME_ACTIONS.json',clean(native))
    write_csv(out/'FRESH_RUNTIME.csv',cold)
    med={m:float(np.median([r['full_inference_seconds'] for r in native if r['method']==m])) for m in config['methods']}
    result={'status':'MEASURED_RAW_MEASUREMENT_TO_IMAGE','seed':seed,'warm_median_seconds':med,
        'warm_cases_per_method':48,'fresh_encoder_cases_per_method':2,'fresh_process_per_method':'NOT_RUN_YET',
        'N2_over_faster_control':med['N2']/min(med['N1'],med['N3']),
        'cache_only_decoder_is_not_total_inference':True,'offline_full_audit_separate':True,
        'timing_instrumentation':'in-memory action counting; outer paid budget, lock and synchronization; full traced comparator also saved'}
    write_json(out/'RUNTIME_SUMMARY.json',result)
    return result


def process_cold_stage(root,data_root,config,book,device,seed=None):
    """One freshly launched interpreter per arm; OS disk cache uncontrolled."""
    method=config.get('process_cold_method')
    if method not in ('N1','N2','N3'):raise ValueError('EXPLICIT_PROCESS_COLD_METHOD_REQUIRED')
    seed=int(config['primary_seed'] if seed is None else seed)
    with np.load(Path(root)/'results/a26/cache/validation_ONLINE_MEASUREMENTS.npz',allow_pickle=False) as a:
        if set(a.files)!={'data','sigma','background_id','case_id'}:raise ValueError('ONLINE_MEASUREMENT_ONLY_REQUIRED')
        obs={k:a[k].copy() for k in a.files}
    rows=[]
    for position,bg in enumerate(('weak','ring')):
        i=next(i for i in range(len(obs['data'])) if str(obs['background_id'][i])==bg and float(obs['sigma'][i])>0)
        _sync(device);t=time.perf_counter()
        enc=FrozenEncoder(Path(data_root)/'background'/(bg+'.npz'),config,book,device=device)
        cp=load_trained_model(Path(root)/'results/a26/learning'/str(seed)/method/'best.pt',device=device,config=config)
        enc.prepare_observable() if method!='N3' else None
        counter=CountOnlyOnlineBook();enc.bind_book(counter)
        with torch.no_grad():
            v=enc.encode(obs['data'][i],float(obs['sigma'][i]),method)
            _sync(device);mass=cp.predict_mass(v['features'],v['base_mass']);_sync(device)
            image=((mass[0,:576]+1j*mass[0,576:])*24).cpu().numpy();_sync(device)
        now=time.perf_counter()
        enc.bind_book(book);book.counts.update(counter.counts);book.check()
        rows.append({'method':method,'background_id':bg,'case_id':str(obs['case_id'][i]),
            'fresh_process':position==0,'disk_cold_claim':False,'new_encoder_seconds':now-t,
            'process_start_to_image_seconds':now-book.started_wall if position==0 else None,
            'finite_image':bool(np.isfinite(image).all())})
    path=Path(root)/'results/a26/evaluation'/str(seed)/('PROCESS_COLD_'+method+'.json')
    result={'status':'ACTUAL_FRESH_INTERPRETER','method':method,'rows':rows,
        'OS_disk_cache':'NOT_CONTROLLED','data_and_truth_read':False,'online_observation_only':True}
    write_json(path,result)
    return result
