"""Bounded packaging/publication of the user's explicitly authorized A26 results.

No experimental data generation, hashes, training or locked-data opening.
Every invocation has an independent A26 native terminal cost receipt.
"""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from a26.budget import charged_job, budget_history, write_json

REPO = 'migodam/a26-om-feedback-prior-completion'
TAG = 'a26-v0.1.0'
URL = 'https://github.com/'+REPO


def command(args,book,timeout=60,check=True):
    book.check()
    t=time.perf_counter()
    result=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,timeout=timeout)
    book.append({'event':'PUBLICATION_COMMAND','command':args,'returncode':result.returncode,'wall_seconds':time.perf_counter()-t})
    if check and result.returncode:
        raise RuntimeError('PUBLICATION_COMMAND_FAILED:'+str(args[0])+':'+result.stderr[:1000])
    book.check()
    return result


def pack(path,members,book,overrides=None):
    overrides=overrides or {}
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,name in members:
            book.check()
            if p.is_symlink() or not p.resolve().is_relative_to(ROOT.resolve()):raise RuntimeError('UNSAFE_ASSET_FILE')
            if any(part in ('private','locked','locked_id','locked_ood') for part in Path(name).parts):raise RuntimeError('LOCKED_OR_PRIVATE_IN_ASSET')
            if name in overrides:z.writestr(name,overrides[name])
            else:z.write(p,name)
    return {'name':path.name,'bytes':path.stat().st_size,'members':len(members)}


def prepare(book):
    assets=ROOT/'publication';assets.mkdir(exist_ok=True)
    # The A26 root has new entry/ledger meanings. Preserve the original root
    # documents byte-for-byte in addition to the frozen parent Git commit.
    frozen=ROOT/'frozen_controls';frozen.mkdir(exist_ok=True)
    source='59d8cc57c9da99ccb7978febc308193c43d6f3e5'
    for name,dest in (('README.md','A25_ROOT_README.md'),('FAILURE_LEDGER.jsonl','A25_ROOT_FAILURE_LEDGER.jsonl')):
        p=frozen/dest
        original=subprocess.run(['git','show',source+':'+name],cwd=ROOT,capture_output=True,check=True).stdout
        if p.exists() and p.read_bytes()!=original:raise RuntimeError('FROZEN_ROOT_DOCUMENT_CONFLICT')
        if not p.exists():p.write_bytes(original)
    inputs=ROOT/'results/a26/repro_inputs'
    members=[(p,p.relative_to(inputs).as_posix()) for p in sorted(inputs.rglob('*')) if p.is_file()]
    manifest=json.loads((inputs/'data/manifest.json').read_text())
    manifest['splits']={k:v for k,v in manifest['splits'].items() if k in ('train','validation')}
    manifest['parents']=[r for r in manifest['parents'] if r['split'] in ('train','validation')]
    manifest['A26_public_scope']='Only TRAIN and exposed VALIDATION; unopened locked parent metadata/payload excluded.'
    audit=json.loads((inputs/'REPRO_INPUT_AUDIT.json').read_text())
    audit['public_manifest_filtered_to_train_validation']=True
    overrides={'data/manifest.json':json.dumps(manifest,indent=2), 'REPRO_INPUT_AUDIT.json':json.dumps(audit,indent=2)}
    records=[pack(assets/'A26_REPRO_INPUTS.zip',members,book,overrides)]
    cps=sorted((ROOT/'results/a26/learning/261009').glob('*/*.pt'))
    if len(cps)!=9:raise RuntimeError('NINE_ACTUAL_CHECKPOINTS_REQUIRED')
    records.append(pack(assets/'A26_CHECKPOINTS.zip',[(p,p.relative_to(ROOT).as_posix()) for p in cps],book))
    raw=[]
    for p in sorted((ROOT/'results/a26').rglob('*')):
        rel=p.relative_to(ROOT).as_posix()
        if (not p.is_file() or 'repro_inputs' in p.parts or 'transport_variants' in p.parts or
            p.suffix=='.pt' or rel.startswith('results/a26/jobs/'+book.metadata['job']+'/')):continue
        raw.append((p,rel))
    raw += [(p,p.relative_to(ROOT).as_posix()) for p in sorted((ROOT/'runs').glob('a26-*.std*')) if p.is_file()]
    raw += [(ROOT/name,name) for name in ('COST_LEDGER.jsonl','FAILURE_LEDGER.jsonl','A26_SOURCE_MANIFEST.json')]
    records.append(pack(assets/'A26_RAW_EVIDENCE.zip',raw,book))
    write_json(ROOT/'RELEASE_ASSET_MANIFEST.json',{'tag':TAG,'assets':records,'new_labels':0,'locked_included':False,
        'unopened_locked_parent_metadata_in_public_inputs':False,'hash_checks':False,'receipts_are_immutable':True,
        'cutoff':'Preparation cutoff; subsequent publication is separately charged.'})
    print(json.dumps({'assets':records}),flush=True)
    return records


def sensitive_review(book):
    # Never print a matching secret; only file and category.
    files=command(['git','ls-files','-z','--cached','--others','--exclude-standard'],book).stdout.split('\0')
    patterns={'private_key':re.compile(rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA|PRIVATE) PRIVATE KEY-----'),
              'github_token':re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),
              'github_pat':re.compile(rb'github_pat_[A-Za-z0-9_]{40,}'),
              'openai_token':re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{30,}')}
    rejected=[];checked=0
    for name in files:
        if not name:continue
        p=ROOT/name
        if p.is_symlink():rejected.append({'file':name,'category':'symlink'});continue
        if any(t in ('private','locked','locked_id','locked_ood') for t in p.relative_to(ROOT).parts):
            rejected.append({'file':name,'category':'private_or_locked_path'});continue
        if p.suffix.lower() in ('.png','.pdf','.pt','.npz','.zip'):continue
        raw=p.read_bytes();checked+=1
        for category,pattern in patterns.items():
            if pattern.search(raw):rejected.append({'file':name,'category':category})
        book.check()
    result={'checked_text_files':checked,'rejections':rejected,'hash_checks':False,
            'third_party':'Existing user-owned A2/A17 and previously public A24/A25 source; dependencies/paper full texts not bundled.',
            'locked':'No unopened payloads published; public reproduction manifest also filters unopened parent metadata.'}
    write_json(ROOT/'PUBLICATION_REVIEW.json',result)
    if rejected:raise RuntimeError('PUBLICATION_SENSITIVE_SCAN_REJECTED')
    return result


def anonymous_check(url,book):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'A26-public-verification'}),timeout=30) as r:
        data=r.read()
        result={'url':url,'http_status':r.status,'bytes':len(data),'authentication':'NONE'}
    book.check();return result,data


def publish(book):
    review=sensitive_review(book)
    command(['git','add','.gitignore','README.md','REPRODUCING_A26.md','requirements-a26.txt',
             'A26_START_HERE.md','A26_RESEARCH_REPORT_ZH.md','A26_FINAL_DECISION.md','A26_COMPLEX_SHAPE_RESULTS.md',
             'A26_FEEDBACK_INCREMENTAL_VALUE.md','A26_NETWORK_COMPARISON.md','A26_PHYSICS_PRIOR_SPLIT.md',
             'A26_COST_AND_RUNTIME.md','A26_EXPLORATION_LEDGER.md','A26_SOURCE_MANIFEST.json','GATE_DECISION.json',
             'SCIENTIFIC_REVIEW_A26.md','RELEASE_ASSET_MANIFEST.json','PUBLICATION_REVIEW.json',
             'PUBLIC_REPRO_AUDIT.json',
             'FAILURE_LEDGER.jsonl','configs/a26','src/a26','tools/a26_driver.py','tools/a26_remote.py','tools/a26_publication.py','tools/a26_delivery_audit.py',
             'research/A26_FROZEN_PROTOCOL.md','research/delegated/a26-data','research/delegated/a26-encoder',
             'frozen_controls',
             'research/delegated/a26-training','research/delegated/a26-metrics','research/delegated/a26-evaluation-audit.md',
             'tests/test_a26_data_budget.py','tests/test_a26_encoder_decoder.py','tests/test_a26_training.py','tests/test_a26_metrics.py',
             'results/a26'],book)
    # Check hand-written implementation only. Native CRLF evidence and
    # Matplotlib SVG paths are preserved, not normalized to a Unix style rule.
    command(['git','-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof',
             'diff','--cached','--check','--','src/a26','tools/a26_driver.py','tools/a26_remote.py',
             'tools/a26_publication.py','tools/a26_delivery_audit.py','tests/test_a26*','configs/a26'],book)
    diff=command(['git','diff','--cached','--name-only'],book).stdout.strip()
    if diff:
        command(['git','commit','-m','A26: train matched imaging CNNs and preserve negative feedback evidence'],book)
    sha=command(['git','rev-parse','HEAD'],book).stdout.strip()
    remotes=command(['git','remote'],book).stdout.split()
    if 'a26' not in remotes:
        exists=command(['gh','repo','view',REPO,'--json','nameWithOwner'],book,check=False)
        if exists.returncode:
            command(['gh','repo','create',REPO,'--public','--description','A26 matched neural material imaging pilot: real training, physics/prior evidence and negative feedback quality-time result','--source','.','--remote','a26'],book)
        else:command(['git','remote','add','a26',URL+'.git'],book)
    command(['git','push','a26','HEAD:main'],book,timeout=60)
    # This is a new independent repo; no original remote or prior tag is changed.
    command(['gh','repo','edit',REPO,'--default-branch','main'],book)
    note=ROOT/'publication/A26_RELEASE_NOTES.md'
    note.write_text('''A26 completed three genuine CNN training runs (2000 updates each, 31,810 parameters).

Scientific result: NO_GO current feedback expansion. Learned prior helps complex-family physical inputs by ~23–24%; N2 does not improve the matched complex-family images and complete warm inference is 3.274× slower. Single-seed exposed development only; locked ID/OOD stays unopened.

Assets contain actual nine checkpoints, existing train/exposed-validation inputs, feature/prediction arrays, failures and complete operation ledgers. No new labels, unopened locked payloads, credentials or SHA256 checks. Public reproduction manifest excludes unopened parent metadata. See README and A26_RESEARCH_REPORT_ZH.md for limitations, frozen gates and commands.
''',encoding='utf-8')
    assets=[str(ROOT/'publication'/n) for n in ('A26_REPRO_INPUTS.zip','A26_CHECKPOINTS.zip','A26_RAW_EVIDENCE.zip')]
    existing=command(['gh','release','view',TAG,'--repo',REPO,'--json','tagName'],book,check=False)
    if existing.returncode:
        command(['gh','release','create',TAG,*assets,'--repo',REPO,'--target',sha,'--title','A26 v0.1.0 — Real neural pilot, negative feedback increment','--notes-file',str(note)],book,timeout=60)
    else:command(['gh','release','upload',TAG,*assets,'--repo',REPO,'--clobber'],book,timeout=60)
    verify(book)
    return {'sha':sha,'review':review,'repository':URL,'release':URL+'/releases/tag/'+TAG}


def verify(book):
    sha=command(['git','rev-parse','HEAD'],book).stdout.strip()
    remote=command(['git','ls-remote','a26','refs/heads/main'],book).stdout.split()[0]
    if sha!=remote:raise RuntimeError('REMOTE_COMMIT_MISMATCH')
    repo=json.loads(command(['gh','repo','view',REPO,'--json','nameWithOwner,visibility,url,defaultBranchRef'],book).stdout)
    if repo['visibility']!='PUBLIC' or repo['defaultBranchRef']['name']!='main':raise RuntimeError('PUBLIC_MAIN_REQUIRED')
    endpoints=[URL,URL+'/blob/'+sha+'/A26_RESEARCH_REPORT_ZH.md',
        'https://raw.githubusercontent.com/'+REPO+'/'+sha+'/A26_RESEARCH_REPORT_ZH.md',
        URL+'/releases/tag/'+TAG]
    checks=[]
    for u in endpoints:
        check,data=anonymous_check(u,book);checks.append(check)
        if '/raw.' in u and 'A26'.encode() not in data:raise RuntimeError('REPORT_CONTENT_MISSING')
    release=json.loads(command(['gh','release','view',TAG,'--repo',REPO,'--json','url,assets,targetCommitish,tagName'],book).stdout)
    # Confirm actual annotated/lightweight release tag resolves to published main.
    tag=command(['gh','api','repos/'+REPO+'/git/ref/tags/'+TAG],book).stdout
    obj=json.loads(tag)['object'];tagsha=obj['sha']
    if obj['type']=='tag':tagsha=json.loads(command(['gh','api','repos/'+REPO+'/git/tags/'+tagsha],book).stdout)['object']['sha']
    if tagsha!=sha:raise RuntimeError('RELEASE_TAG_COMMIT_MISMATCH')
    if len(release['assets'])!=3:raise RuntimeError('THREE_RELEASE_ASSETS_REQUIRED')
    for asset in release['assets']:
        u=URL+'/releases/download/'+TAG+'/'+asset['name']
        req=urllib.request.Request(u,method='HEAD',headers={'User-Agent':'A26-anonymous-asset-check'})
        with urllib.request.urlopen(req,timeout=30) as response:
            if int(response.headers.get('Content-Length','0'))!=asset['size']:raise RuntimeError('ASSET_LENGTH_MISMATCH')
            checks.append({'url':u,'http_status':response.status,'content_length':response.headers.get('Content-Length'),
                           'authentication':'NONE','asset_bytes':asset['size'],'check':'HEAD_NOT_FULL_DOWNLOAD'})
        book.check()
    status={'publication':'VERIFIED_PUBLIC','scientific_gate':'NO_GO','repository':URL,
        'report':endpoints[1],'raw_report':endpoints[2],'commit_sha':sha,'release':release['url'],
        'release_tag_commit':tagsha,'anonymous_checks':checks,'new_labels':0,'locked_opened':False,
        'hash_checks':False,'cost_snapshot_before_verification_receipt':budget_history(ROOT)['cpu_seconds']}
    write_json(ROOT/'publication/PUBLICATION_STATUS.json',status)
    print(json.dumps(status),flush=True)
    return status


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','publish','verify'));p.add_argument('--job',required=True)
    a=p.parse_args();config=json.loads((ROOT/'configs/a26/pilot.json').read_text())
    with charged_job(ROOT,a.job,device='cpu',config=config,metadata={'stage':'publication_'+a.stage}) as book:
        result={'prepare':prepare,'publish':publish,'verify':verify}[a.stage](book)
        write_json(ROOT/'results/a26/jobs'/a.job/'RESULT.json',result)

if __name__=='__main__':main()
