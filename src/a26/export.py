"""Repackage only already-authorized TRAIN/exposed validation inputs."""
from pathlib import Path
import shutil
import numpy as np
from .data import load_split,ROW_METADATA,SCALAR_METADATA,BASELINE_KEYS
from .budget import write_json

def export_stage(root,data_root,config,book):
    root,data_root=Path(root),Path(data_root)
    out=root/'results/a26/repro_inputs';out.mkdir(parents=True,exist_ok=True)
    for role in ('train','validation'):
        book.check()
        with book.span('a26_repro_target_and_observation_copy',copied_existing_label_rows=512 if role=='train' else 128):
            arrays=load_split(data_root,role)
            keys=(ROW_METADATA|SCALAR_METADATA|BASELINE_KEYS|{'data','sigma','chi'}) & set(arrays)
            dst=out/'data'/(role+'.npz');dst.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dst,**{k:arrays[k] for k in keys})
    with book.span('a26_repro_known_background_copy'):
        for rel in ('manifest.json','geometry.json','background/weak.npz','background/ring.npz'):
            src=data_root/rel
            if src.is_symlink() or not src.resolve().is_relative_to(data_root.resolve()):raise RuntimeError('UNSAFE_REPRO_INPUT')
            dst=out/'data'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        dest=out/'configs/a25/nn_admitted.json';dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/'configs/a25/nn_admitted.json',dest)
    result={'status':'REPACKAGED_EXISTING_ALLOWED_INPUTS','new_labels':0,'copied_label_rows':640,
        'independent_parents':80,'locked_payload_included':False,'current_teacher_arrays_included':False,
        'background_J':'KNOWN_BACKGROUND_ONLY','source_commit':config['source_commit'],'hash_checks':False}
    write_json(out/'REPRO_INPUT_AUDIT.json',result)
    return result
