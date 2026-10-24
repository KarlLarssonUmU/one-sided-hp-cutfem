#!/usr/bin/env python3
"""Export fresh measurements, flags, fits and differences from published values."""
from pathlib import Path
import argparse, collections, json, math, os, sys
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.setdefault(name,'1')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from results import read, extract, write_csv, slopes


def plots(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Shared boxed legends above each row; flagged values are open crosses.
    for disk in ['fixed_A','fixed_B']:
        fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
        for ax,norm in zip(axes,['L2','H1','energy']):
            for degree in [2,4,6,8]:
                s=sorted([r for r in rows if r['family']=='baseline' and r.get('solution_kind')=='exponential' and r.get('path')==disk and r.get('p')==degree and r['quantity']==norm],key=lambda r:r['N'])
                if not s:continue
                line,=ax.loglog([2/r['N'] for r in s],[r['value'] for r in s],'.:',label=f'p={degree}')
                bad=[r for r in s if not r['resolved']]
                ax.loglog([2/r['N'] for r in bad],[r['value'] for r in bad],'x',color=line.get_color(),ms=8)
            ax.set(xlabel='h',title=norm);ax.grid(alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels()
        if handles:fig.legend(handles,labels,loc='outside upper center',ncol=4,frameon=True)
        fig.savefig(out/f'h-{disk}.pdf');plt.close(fig)
    for disk in ['fixed_A','fixed_B']:
        fig,ax=plt.subplots(figsize=(5,3),layout='constrained')
        for norm in ['L2','H1','energy']:
            s=[r for r in rows if r.get('path')==disk and r.get('solution_kind')=='radial_power' and ((r['family']=='baseline' and r['quantity']==norm) or (r['family']=='radial' and r['quantity']==('gradient' if norm=='H1' else norm)))]
            s.sort(key=lambda r:r['p'])
            if s:
                line,=ax.loglog([r['p'] for r in s],[r['value'] for r in s],'.:',label=norm)
                bad=[r for r in s if not r['resolved']];ax.loglog([r['p'] for r in bad],[r['value'] for r in bad],'x',color=line.get_color())
        ax.set(xlabel='p',ylabel='Error');ax.grid(alpha=.2)
        if ax.lines:ax.legend()
        fig.savefig(out/f'radial-{disk}.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,help='Search this campaign or single-run directory')
    p.add_argument('--published',action='store_true',help='Export published observations/tables; no new calculation')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plots',action='store_true')
    p.add_argument('--rtol',type=float,default=.01,help='Informational cross-run comparison tolerance, distinct from scientific gates')
    p.add_argument('--require-complete',action='store_true',help='Exit nonzero unless every published observation was reproduced')
    a=p.parse_args();assert (a.runs is None)!= (not a.published),'Choose exactly one of --runs or --published'
    assert 0<=a.rtol<1
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    reference=read(ROOT/'data/published.json.gz');statuses=[];rows=[];flags=[]
    if a.published:
        rows=reference['observations'];flags=reference['failed_criteria']
        tables=out/'published-tables';tables.mkdir()
        for name,table in reference['tables'].items():write_csv(tables/(name+'.csv'),table)
    else:
        paths=sorted(a.runs.resolve().rglob('request.json'))
        assert paths,'No generated runs found'
        for path in paths:
            folder=path.parent;completion=folder/'completion.json'
            if not completion.is_file():statuses.append(dict(run=str(folder),status='unfinished'));continue
            status=read(completion);statuses.append(dict(run=str(folder),**status))
            if status['exit_code'] not in [0,2]:continue
            rr,ff=extract(folder);rows.extend(rr);flags.extend(ff)
    identity=lambda r:(r['family'],r['case'],r['quantity'])
    # Diagnostics can be split across runs; duplicates from other repeated runs are ambiguous.
    keys=[identity(r) for r in rows if r['family']!='diagnostics'];assert len(set(keys))==len(keys),'Duplicate cases: report a single campaign, not mixed attempts'
    write_csv(out/'measurements.csv',rows);write_csv(out/'failed-criteria.csv',flags)
    fit=slopes(rows);write_csv(out/'fits.csv',fit)
    expected={identity(r):r for r in reference['observations']};got={identity(r):r for r in rows}
    comparisons=[]
    for key,new in got.items():
        old=expected.get(key)
        if old is None or not old['compare']:continue
        x,y=new['value'],old['value'];numeric=type(x) in [int,float] and type(y) in [int,float]
        relative=abs(x-y)/max(abs(y),1e-300) if numeric else None
        matches=relative<=a.rtol if numeric else x==y
        comparisons.append(dict(family=key[0],case=key[1],quantity=key[2],published=y,computed=x,relative_change=relative,matches=matches,
            published_resolved=old['resolved'],computed_resolved=new['resolved']))
    write_csv(out/'comparison.csv',comparisons)
    missing=[list(k) for k in expected if k not in got]
    summary=dict(source='published reference' if a.published else 'fresh run outputs',observations=len(rows),failed_criteria=len(flags),
        runs=statuses,missing_observations=missing,complete=not missing,
        comparison_rtol=a.rtol,comparisons=len(comparisons),differences=sum(not c['matches'] for c in comparisons),
        resolution_changes=sum(c['computed_resolved']!=c['published_resolved'] for c in comparisons),
        note='Comparison differences are reported, never hidden. Scientific acceptance uses each run\'s original criteria. Timings, unresolved values and tiny rounding-level residuals are platform dependent; paper figures use frozen published coordinates.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if a.plots:plots(rows,out)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['missing_observations','runs']},indent=2))
    return 2 if a.require_complete and missing else 0

if __name__=='__main__':sys.exit(main())
