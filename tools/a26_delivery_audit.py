"""Metered audit of actual public reproduction inputs and saved neural evidence."""
from pathlib import Path
import json
import sys
import tempfile
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from a26.budget import charged_job, write_json
from a26.data import load_split

def main():
    config=json.loads((ROOT/'configs/a26/pilot.json').read_text())
    with charged_job(ROOT,'a26-delivery-evidence-audit-001',device='cpu',config=config,metadata={'stage':'public_repro_input_audit'}) as book:
        with tempfile.TemporaryDirectory(prefix='a26-public-input-audit-') as tmp:
            dest=Path(tmp)
            with zipfile.ZipFile(ROOT/'publication/A26_REPRO_INPUTS.zip') as z:
                for name in z.namelist():
                    p=Path(name)
                    if p.is_absolute() or '..' in p.parts or any(x in ('private','locked','locked_id','locked_ood') for x in p.parts):raise RuntimeError('FORBIDDEN_PUBLIC_ZIP_MEMBER')
                z.extractall(dest)
            manifest=json.loads((dest/'data/manifest.json').read_text())
            assert set(manifest['splits'])=={'train','validation'}
            assert {r['split'] for r in manifest['parents']}=={'train','validation'}
            splits={}
            for role in ('train','validation'):
                with book.span('public_reproduction_input_load', copied_existing_input_observations=512 if role=='train' else 128):
                    a=load_split(dest/'data',role)
                    assert 'current' not in a
                    splits[role]={'observations':len(a['data']),'parents':len(np.unique(a['parent_id'])),'material_labels':a['chi'].shape[0]}
        with book.span('saved_neural_evidence_audit'):
            with zipfile.ZipFile(ROOT/'publication/A26_CHECKPOINTS.zip') as z:
                assert len(z.namelist())==9
                assert all(n.endswith(('/best.pt','/middle.pt','/final.pt')) for n in z.namelist())
            for method in config['methods']:
                p=ROOT/f'results/a26/learning/261009/{method}'
                m=json.loads((p/'training_manifest.json').read_text())
                losses=json.loads((p/'raw_loss.json').read_text())
                assert m['completed_updates']==2000 and m['parameter_count']==31810 and len(losses)==2000
            with np.load(ROOT/'results/a26/evaluation/261009/PREDICTIONS.npz',allow_pickle=False) as a:
                assert np.array_equal(a['BASE_N1'],a['BASE_N3'])
                assert all(a[m].shape==(128,576) and np.isfinite(a[m]).all() for m in config['methods'])
            summary=json.loads((ROOT/'results/a26/evaluation/261009/EVALUATION_SUMMARY.json').read_text())
            assert summary['numerical_failures']==0 and not summary['locked_opened'] and summary['new_labels']==0
            gates=json.loads((ROOT/'GATE_DECISION.json').read_text())
            assert gates['primary_gate']=='NO_GO' and not gates['repeat_authorized']
        result={'technical_delivery_audit':'PASSED','scientific_gate':'NO_GO','input_splits':splits,
                'NN_models_actually_trained':3,'updates_per_model':2000,'saved_checkpoints':9,
                'N1_N3_noNN_material_outputs':'BITWISE_IDENTICAL','locked_payload_published':False,
                'locked_parent_metadata_in_public_input_manifest':False,'new_labels':0,'sha256_checks':False}
        write_json(ROOT/'PUBLIC_REPRO_AUDIT.json',result)
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
