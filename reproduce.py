#!/usr/bin/env python3
"""Plan or execute the complete paper campaign, with explicit dependencies."""
from pathlib import Path
import argparse, collections, json, os, shlex, subprocess, sys
import run


def plan(output, threads=1, workers=1):
    jobs=[]
    def add(key, args, dependencies=(), resources=None):
        jobs.append(dict(id=key,command=[sys.executable,'-B',str(run.ROOT/'run.py'),*map(str,args),
            '--threads',str(threads),'--output',str(output/key)],dependencies=list(dependencies),
            historical_resources=resources or {}))
    for key,e in sorted(run.catalog('baseline').items()):
        add('baseline/'+key,['baseline',key],resources=e['historical_resources'])
    for key,e in sorted(run.catalog('schwarz').items()):
        dep='baseline/'+e['operator_case']
        add('schwarz/'+key,['schwarz',key,'--operator',output/dep],[dep])
    for key in sorted(run.catalog('accurate')):
        add('accurate/'+key,['accurate',key,'--workers',workers])
    for p in range(1,9):
        for path in ['axis','diagonal','oblique']:
            for eps in [.03,.003,.0003,.00003]:
                add(f'flux/p{p}-{path}-{eps:g}',['flux','--p',p,'--path',path,'--epsilon',eps])
        add(f'controls/p{p}',['controls','--p',p])
    for p in [10,12,14,16]:
        for disk in ['fixed_A','fixed_B']:
            add(f'radial/p{p}-{disk}',['radial','--p',p,'--disk',disk])
    deps=[f'radial/p{p}-{disk}' for p in [10,12,14] for disk in ['fixed_A','fixed_B']]
    for p in [12,14,16]:
        matrices=[v for dep in deps for v in ['--matrix',output/dep/'output'/dep.split('/')[1]/'q24.npz']] if p==12 else []
        add(f'diagnostics/p{p}',['diagnostics','--p',p,*matrices],deps if p==12 else [])
    add('tubes',['tubes','--seed-audit'])
    add('continuation',['continuation'])
    add('conditioning',['conditioning'])
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='Campaign output root')
    p.add_argument('--study',choices=['all','baseline','schwarz','accurate','flux','controls','radial','diagnostics','tubes','continuation','conditioning'],default='all')
    p.add_argument('--case',action='append',help='Exact plan ID, e.g. baseline/PAPER-004; repeatable; dependencies included')
    p.add_argument('--figure',type=int,choices=[9],help='Reproduce the two radial curves in main-paper Figure 9, including flagged attempts')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--execute',action='store_true',help='Without this switch, only print the plan; no calculations')
    p.add_argument('--resume',action='store_true',help='Skip only completed, matching runs; refuse unfinished directories')
    a=p.parse_args();assert a.threads>0 and a.workers>0
    output=a.output.resolve();jobs=plan(output,a.threads,a.workers);index={j['id']:j for j in jobs}
    selected=set(a.case or [j['id'] for j in jobs if a.study=='all' or j['id'].split('/')[0]==a.study])
    if a.figure is not None:
        if a.case or a.study!='all':p.error('--figure cannot be combined with --case or --study')
        selected={'baseline/'+key for key,e in run.catalog('baseline').items() if e['parameters']['case']['solution_kind']=='radial_power'}
        selected.update(j['id'] for j in jobs if j['id'].startswith('radial/'))
    assert selected<=index.keys(),f'Unknown cases: {selected-index.keys()}'
    def include(key):
        for dep in index[key]['dependencies']:
            if dep not in selected:selected.add(dep);include(dep)
    for key in list(selected):include(key)
    jobs=[j for j in jobs if j['id'] in selected]
    if not a.execute:
        print(json.dumps(dict(count=len(jobs),counts=dict(collections.Counter(j['id'].split('/')[0] for j in jobs)),jobs=jobs),indent=2));return 0
    output.mkdir(parents=True,exist_ok=True)
    expected=run.source_hashes();outcomes=[]
    for j in jobs:
        destination=output/j['id']
        if destination.exists():
            assert a.resume, f'{destination} already exists; use --resume only for matching completed runs'
            completion=run.read(destination/'completion.json')
            assert completion['exit_code'] in [0,2],f'Unfinished/failed run: {destination}; retain it and choose a new output root'
            assert run.read(destination/'source-sha256.json')==expected,f'Source/input changes: {destination}'
            command_file=destination/'campaign-command.json'
            assert command_file.is_file() and run.read(command_file)==j['command'],f'Run arguments not certified for resume: {destination}'
            outcomes.append(dict(id=j['id'],exit_code=completion['exit_code'],resumed=True));continue
        for dep in j['dependencies']:
            assert run.read(output/dep/'completion.json')['exit_code'] in [0,2]
        logs=output/'campaign-logs';logs.mkdir(exist_ok=True)
        print(shlex.join(j['command']),flush=True)
        with (logs/(j['id'].replace('/','-')+'.log')).open('w') as f:
            code=subprocess.run(j['command'],stdout=f,stderr=subprocess.STDOUT).returncode
        if destination.exists():run.dump(destination/'campaign-command.json',j['command'])
        outcomes.append(dict(id=j['id'],exit_code=code,resumed=False))
        run.dump(output/'campaign-status.json',dict(planned=len(jobs),outcomes=outcomes,complete=False))
        if code not in [0,2]:raise RuntimeError(f"Execution failed for {j['id']}; see the retained log")
    run.dump(output/'campaign-status.json',dict(planned=len(jobs),outcomes=outcomes,complete=True))
    print(f'Completed {len(jobs)} commands; {sum(j["exit_code"]==2 for j in outcomes)} retain numerical flags/failures.')
    return 0

if __name__=='__main__':sys.exit(main())
