"""Extract comparable observations from freshly generated result files.

No published values enter assembly, solves, quadrature or these extractors.
Floating point failures and all failed criteria are retained separately.
"""
from pathlib import Path
import csv, gzip, json, math


def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if str(p).endswith('.gz') else p.read_text())


def gate_applies(g,norm):
    if g['kind']=='stability':return False
    if g['kind'] in ['accuracy','solve_accuracy']:
        named=[n for n in ['L2','H1','energy'] if '_'+n+'_' in g['metric'] or g['metric'].endswith('_'+n)]
        return not named or norm in named
    return True


def norm_values(z):
    out={'L2':z['l2'],'H1':z['component_norms']['gradient'],'energy':z['energy']}
    out.update({k:z['component_norms'][k] for k in ['gradient','boundary','stabilization']})
    out['physical']=math.hypot(out['gradient'],out['boundary'])
    return out


def observations(family,key,raw):
    rows=[];flags=[]
    def add(quantity,value,resolved=True,compare=True,**extra):
        if isinstance(value,float) and not math.isfinite(value):value=None;resolved=False
        rows.append(dict(family=family,case=key,quantity=quantity,value=value,resolved=resolved,compare=compare,**extra))
    if family in ['baseline','schwarz','accurate']:
        gates=raw['gates'];failed=[g for g in gates if not g['passed']]
        flags=[dict(family=family,case=key,**g) for g in failed]
        add('criteria_total',len(gates));add('criteria_passed',sum(g['passed'] for g in gates))
        c=raw['case'];extra={k:c[k] for k in ['N','p','path']};extra['solution_kind']=c.get('solution_kind','exponential')
        if family=='baseline':
            label='online_q'+str(max(a['boost'] for a in raw['assembly']))+'_error'
            measurements=[norm_values(er['fields'][label]) for er in raw['errors']]
            for norm,value in measurements[-1].items():
                gate_norm={'gradient':'H1','physical':'energy','boundary':'energy','stabilization':'energy'}.get(norm,norm)
                changed=abs(measurements[0][norm]-value)/max(abs(value),1e-300)
                good=not any(gate_applies(g,gate_norm) for g in failed) and changed<=1e-4
                add(norm,value,good,**extra)
                add(norm+'_rule_change',changed,changed<=1e-4,False,**extra)
            add('minimum_fraction',raw['geometry']['minimum_positive_fraction'])
            a=raw['assembly'][-1]
            for name,fields in [('spectrum',['lambda_min','lambda_max','condition']),('threshold',['lower','upper'])]:
                for field in fields:
                    if field in a.get(name,{}):add(name+'/'+field,a[name][field],not any(g['kind']=='stability' for g in failed))
        else:
            for label,z in raw['errors'][-1]['fields'].items():
                for norm,value in norm_values(z).items():
                    add('fields/'+label+'/'+norm,value,not failed,compare=label.endswith('_error'),**extra)
            if family=='schwarz':
                for z in raw['comparisons']:
                    for name in ['condition','lambda_min','lambda_max']:add(z['label']+'/'+name,z['spectrum'][name],not failed)
                    for name in ['iterations','true_relative_residual','energy_difference_ratio','converged']:
                        add(z['label']+'/'+name,z['pcg'][name],not failed,compare=name in ['iterations','converged'])
            else:
                add('accurate_candidate_accepted',raw['accurate_candidate_accepted'])
    elif family=='flux':
        z=raw['rules'][-1]
        for label,spectrum in z['spectra'].items():
            add(label+'/resolved',spectrum['resolved'])
            for field in ['value','basis_witness_lower_bound']:
                if field in spectrum:add(label+'/'+field,spectrum[field],spectrum['resolved'],p=raw['p'],path=raw['path'],epsilon=raw['epsilon'])
    elif family=='fitted':
        z=raw['rules'][-1]
        add('flux_constant',z['spectrum'].get('value'),z['spectrum']['resolved'])
        for name in ['beta_lower','beta_upper','nominal_coercivity','kappa_J']:add(name,z[name])
    elif family=='extension':add('extension_seminorm',raw['rules'][-1]['norm'],raw['relative_change']<1e-5,p=raw['p'],path=raw['disk'])
    elif family=='radial':
        add('status',raw['status'])
        for norm,value in raw.get('errors',{}).items():
            good=raw['resolved'].get(norm,all(raw['resolved'].values()))
            add(norm,value,good,p=raw['p'],path=raw['disk'],N=raw['N'],solution_kind='radial_power')
        if raw.get('exception'):add('exception',raw['exception'])
        for norm,checks in raw.get('checks',{}).items():
            if not raw['resolved'][norm]:flags.append(dict(family=family,case=key,metric=norm,actual=checks,limit='rule <= 1e-4; field <= 0.01',passed=False,kind='accuracy'))
    elif family=='conditioning':
        for name in ['kappa_J','lambda_min_J','lambda_max_J','nominal_coercivity']:add(name,raw['rules'][-1][name],raw['resolved'])
        add('resolved',raw['resolved'])
    elif family=='continuation':
        for name,value in raw.items():
            if name!='p':add(name,float(value))
    elif family=='tubes':
        for name in ['verified','geometries','cells','reassigned','earlier_witnesses']:add(name,raw[name])
    elif family=='seed':
        for name in ['minimum_depth_over_h_lower_bound','tube_h_over_16_certified','maximum_overlap','maximum_patch_elements','independent_ray_samples']:add(name,raw[name])
    elif family=='diagnostics':
        for part in ['bubble_grams','basis','systems']:
            for z in raw[part]:
                ident=str(z.get('degree',Path(z.get('source','')).parent.name))
                for name,value in z.items():
                    if name in ['traceback','source','reason','caution','degree']:continue
                    if isinstance(value,(str,float,int,bool)) or value is None:
                        add(part+'/'+ident+'/'+name,value,compare=False)
    return rows,flags


def extract(run):
    req=read(run/'request.json');f=req['command'];result=[];flags=[]
    def take(family,key,raw):
        a,b=observations(family,key,raw);result.extend(a);flags.extend(b)
    if f in ['baseline','schwarz']:take(f,req['case'],read(run/'results.json'))
    elif f=='accurate':take(f,req['case'],read(run/req['case']/'results.json'))
    elif f=='flux':
        for z in read(run/'output/cut-results.json'):take(f,f"p{z['p']}-{z['path']}-{z['epsilon']:g}",z)
    elif f=='controls':
        for z in read(run/'output/fitted-results.json'):take('fitted',f"p{z['p']}",z)
        for z in read(run/'output/extension-results.json'):take('extension',f"p{z['p']}-{z['disk']}",z)
    elif f=='radial':
        for z in read(run/'output/radial-results.json'):take(f,f"p{z['p']}-{z['disk']}",z)
    elif f=='conditioning':
        for z in read(run/'output/results.json'):take(f,f"L{z['L']}-gamma{z['gamma']:g}",z)
    elif f=='continuation':
        for z in csv.DictReader((run/'output/continuation-sharp.csv').open()):take(f,'p'+z['p'],z)
    elif f=='tubes':
        take(f,req.get('geometry') or 'all',read(run/'output/verification.json'))
        if (run/'output/seed-assignment.json').exists():
            for z in read(run/'output/seed-assignment.json'):take('seed',z['id']+f"-L{z['layers']}",z)
    elif f=='diagnostics':take(f,'basis-and-systems',read(run/'output/diagnostics.json'))
    return result,flags


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def slopes(rows):
    # Only consecutive resolved mesh levels enter a fit; no bridging flagged points.
    import numpy as np
    output=[]
    for disk in ['fixed_A','fixed_B']:
        for degree in [2,4,6,8]:
            for norm in ['L2','H1','energy','physical','stabilization']:
                seq=sorted([r for r in rows if r['family']=='baseline' and r.get('solution_kind')=='exponential' and r.get('p')==degree and r.get('path')==disk and r['quantity']==norm],key=lambda r:r['N'])
                segments=[[]]
                for r in seq:
                    if r['resolved']:segments[-1].append(r)
                    elif segments[-1]:segments.append([])
                for i,seg in enumerate(segments):
                    if len(seg)<3:continue
                    for label,s in [(f'segment_{i+1}',seg),('finest_three',seg[-3:])]:
                        x=np.log([2/r['N'] for r in s]);y=np.log([r['value'] for r in s]);fit=np.polyfit(x,y,1)
                        output.append(dict(study='h',disk=disk,p=degree,norm=norm,window=label,case_ids=';'.join(r['case'] for r in s),slope=float(fit[0])))
        for norm in ['L2','H1','energy','physical','stabilization']:
            base=[r for r in rows if r['family']=='baseline' and r.get('solution_kind')=='radial_power' and r.get('path')==disk and r['quantity']==norm]
            ext=[dict(r,quantity='H1') if r['quantity']=='gradient' else r for r in rows if r['family']=='radial' and r.get('path')==disk]
            for label,degrees,items in [('p4-8',range(4,9),base),('p3-7',range(3,8),base),('p4-12',[4,5,6,7,8,10,12],base+ext)]:
                s=sorted([r for r in items if r.get('p') in degrees and r['quantity']==norm and r['resolved']],key=lambda r:r['p'])
                if len(s)!=len(degrees):continue # Never silently change a stated fit window.
                fit=np.polyfit(np.log([r['p'] for r in s]),np.log([r['value'] for r in s]),1)
                output.append(dict(study='radial',disk=disk,norm=norm,window=label,case_ids=';'.join(r['case'] for r in s),slope=float(fit[0])))
    return output
