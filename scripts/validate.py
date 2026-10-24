#!/usr/bin/env python3
"""Selected portability checks. Run inside a CPU allocation, not at import time."""
from pathlib import Path
import argparse,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
from results import extract, read

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);receipts=[]
def invoke(name,arguments,expected=0):
    start=time.time()
    with (out/(name+'.log')).open('w') as log:code=subprocess.run([sys.executable,'-B',*map(str,arguments)],stdout=log,stderr=subprocess.STDOUT).returncode
    receipt=dict(test=name,exit_code=code,expected=expected,seconds=time.time()-start,passed=code==expected)
    receipts.append(receipt);run.dump(out/'validation.json',receipts)
    assert receipt['passed'],(name,out/(name+'.log'))
    print(name,'passed',round(receipt['seconds'],2),flush=True)

campaign=[ROOT/'reproduce.py','--output',out/'campaign','--execute']
for key in ['schwarz/PAPER-004','controls/p2','flux/p2-axis-3e-05','radial/p16-fixed_A','diagnostics/p16']:campaign+=['--case',key]
invoke('campaign',campaign)
invoke('resume',campaign+['--resume'])
commands=[('reference-A',['baseline','PAPER-059']),('reference-B',['baseline','PAPER-060']),
    ('refinement',['baseline','EXT-N4-p8-fixed_A']),('tubes',['tubes','--seed-audit']),
    ('continuation',['continuation']),('accurate',['accurate','p8-fixed_A','--workers','3']),
    ('conditioning',['conditioning']),('radial',['radial','--p','10','--disk','fixed_A'])]
for name,args in commands:invoke(name,[ROOT/'run.py',*args,'--output',out/name])
invoke('verify-certificate',[ROOT/'run.py','tubes','--geometry','geometry-02','--certificates',out/'tubes/output/certificates','--output',out/'verified-one'])
# Make reports for the campaign separately, so no duplicated diagnostic/tube identifiers.
invoke('report',[ROOT/'report.py','--runs',out/'campaign','--output',out/'report','--plots'])
summary=read(out/'report/summary.json');assert summary['differences']==0 and summary['resolution_changes']==0,summary
expected={(r['family'],r['case'],r['quantity']):r for r in read(ROOT/'data/published.json.gz')['observations']}
checks=[]
for name in ['reference-A','reference-B','refinement','accurate','radial']:
    rows,_=extract(out/name)
    for r in rows:
        if r['quantity'] in ['L2','H1','energy','gradient','fields/paired_error/L2','fields/paired_error/H1','fields/paired_error/energy']:
            old=expected[r['family'],r['case'],r['quantity']]['value'];change=abs(r['value']/old-1)
            assert change<1e-6,(name,r,old,change)
            checks.append(dict(test=name,quantity=r['quantity'],relative_change=change,tolerance=1e-6))
run.dump(out/'norm-repeats.json',checks)
status=read(out/'campaign/radial/p16-fixed_A/output/radial-results.json')[0]
assert status['status']=='execution_failed' and status['exception']=='LinAlgError'
assert read(out/'campaign/campaign-status.json')['complete']
print('Selected portability, numerical-repeat, failure-retention and report checks passed.',flush=True)
