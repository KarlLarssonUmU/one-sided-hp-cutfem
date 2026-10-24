#!/usr/bin/env python3
"""Bounded paper hp accuracy comparison; reference data are validation only."""
from pathlib import Path
from decimal import Decimal as D, localcontext
import json, sys, time, traceback, resource
from types import SimpleNamespace
RUN = None  # Set by the portable entry point.
import numpy as np
import workstation_case as wc
import paper_solver_2d as ordinary
import paper_energy as energy
import paper_high_precision as accurate
import extended_case_common as common
import extended_retained_geometry_2d as geometry
import saved_refinement_solver as refinement
import paired_field_export as exporter
import paired_bernstein_evaluator as evaluator


def metric(z, norm):
    return z['component_norms']['gradient'] if norm == 'gradient' else z[norm]


def one(spec):
    directory = RUN/spec['label']; directory.mkdir(exist_ok=True)
    r = dict(status='running', label=spec['label'], gates=[], errors=[], baseline=[])
    def save(): common.write(directory/'results.json', r)
    def gate(name, value, limit, kind='correctness'):
        value = float(value)
        passed = np.isfinite(value) and value <= limit
        g = dict(metric=name, actual=value if np.isfinite(value) else None,
                 limit=limit, passed=bool(passed), kind=kind)
        r['gates'].append(g)
        if kind == 'correctness' and not passed:
            save(); raise AssertionError(g)
    save(); start=time.perf_counter()
    c=wc.normalize_case(spec['case']); mesh,center,space,basis,dofs=ordinary.setup(c)
    r.update(case=c, ndof=len(dofs), active_elements=len(space['active']))
    fingerprint=common.identity(mesh,space,center,c)
    common.write(directory/'case-identity.json',fingerprint)
    inputs=RUN/'inputs'/spec['label']
    gate('reference_case_identity', int(fingerprint['sha256'] != common.read(inputs/'reference-identity.json')['sha256']),0)
    gate('strict_classifier_matches_Decimal',int(ordinary.classify(mesh,center,c['radius']) != geometry.classify(mesh,center,c['radius'])),0)
    reference=np.asarray([D(x) for x in common.read(inputs/'reference-solution.json')],dtype=object)
    r['basis_checks']=ordinary.basis_checks(mesh,space,basis,dofs)
    gate('basis_exact_conformity',r['basis_checks']['nonconforming_edges'],0)
    solutions={}
    if spec.get('resume_completed_calculations'):
        prior=inputs/'retained-calculations'; previous=common.read(prior/'results.json')
        gate('resume_case_identity',int(fingerprint!=common.read(prior/'case-identity.json')),0)
        assert all(g['passed'] for g in previous['gates'])
        r['gates'].extend(previous['gates'])
        r['resume_provenance']=dict(previous_job=spec['resume_job_id'],
            scope='Completed assembly, solves and ordinary coefficients only; all error norms measured in this attempt')
        moments=common.read(prior/'analytic-moments.json.gz')
        boundary=sorted(z['element'] for z in moments if D(z['geometry']['circle_length'])>0)
        import run_2d_disk_study as fem
        assembly=SimpleNamespace(metadata=previous['accurate_assembly'],
            patches=fem.build_patches(mesh,space['active'],boundary,c['patch_layers']))
        for order in (32,40):
            solutions[order]=np.asarray([D(x) for x in common.read(prior/f'taylor{order}-solution-Decimal.json')],dtype=object)
            r[f'taylor{order}_solve']=previous[f'taylor{order}_solve']
        r['baseline']=previous['baseline']
    else:
        assembly=accurate.Assembly(mesh,space,center,c,spec.get('workers',1),gate,directory)
        r['accurate_assembly']=assembly.metadata;save()
        retained=accurate.RetainedSolver(assembly.rows)
        for order in (32,40):
            x,info=retained.solve(assembly.rhs[order]); solutions[order]=x
            common.write(directory/f'taylor{order}-solution-Decimal.json',list(map(str,x)))
            r[f'taylor{order}_solve']=info
            gate(f'taylor{order}_residual',D(info['relative_infinity_residual']),1e-40,'solve')
            gate(f'taylor{order}_coefficient_update',D(info['relative_coefficient_change_final']),1e-35,'solve')
            save()
        del retained
    r['accurate_assembly']=assembly.metadata
    for order in (32,40):
        common.write(directory/f'taylor{order}-solution-Decimal.json',list(map(str,solutions[order])))
    save()
    paired=accurate.PairedField(mesh,space,solutions[40]); paired.save(directory/'paired-field.npz')
    r['paired_export']=paired.bounds
    worst=np.zeros(3); collapsed_worst=np.zeros(3); probes=0
    bary=np.asarray([[0,0],[1,0],[0,1],[.2,.3],[1/3,1/3],[.7,.1]])
    for eid in space['active']:
        e=mesh.elements[eid]; points=e.xy[0]+bary@e.J.T
        index=paired.positions[eid]
        truth=exporter.oracle(paired.polynomials[index],c['p'],e.xy[0],e.J,points)
        h,l=paired.evaluate(eid,points)
        worst=np.maximum(worst,np.max(abs(exporter.discrepancies(h,l,truth)),axis=1))
        h,l=evaluator.evaluate_element(paired.export,index,points,collapsed=True)
        collapsed_worst=np.maximum(collapsed_worst,np.max(abs(exporter.discrepancies(h,l,truth)),axis=1))
        probes+=len(points)
    r['online_probes']=dict(points=probes,paired_max_absolute_error=worst.tolist(),
        collapsed_max_absolute_error=collapsed_worst.tolist(),oracle_digits=100,
        scope='Six full-triangle points per active element; not an integrated online norm')
    gate('paired_online_value_gradient_probes',np.max(worst),1e-28,'online')
    baselines={}
    if spec.get('resume_completed_calculations'):
        for boost in (8,12):
            with np.load(inputs/'retained-calculations'/f'baseline-q{boost}.npz') as z:
                baselines[f'q{boost}']=z['coefficients'].copy()
        r['baseline_source']='Completed fresh ordinary q8/q12 calculations retained from failed job'
    elif (inputs/'baseline-solutions.npz').exists():
        with np.load(inputs/'baseline-solutions.npz') as z: baselines['q12']=z['q12'].copy()
        r['baseline_source']='Frozen original paper p8 q12, six corrections'
    else:
        for boost in (8,12):
            print('Ordinary assembly',spec['label'],boost,flush=True)
            tick=time.perf_counter()
            A,E,b,parts,info=ordinary.assemble(mesh,space,center,c,basis,boost)
            initial,solve=ordinary.solve(A,b)
            x,refine,last,rounding=refinement.refine(A,b,initial,6)
            baselines[f'q{boost}']=x
            np.savez_compressed(directory/f'baseline-q{boost}.npz',coefficients=x,last_update=last)
            r['baseline'].append(dict(boost=boost,seconds=time.perf_counter()-tick,
                assembly=info,solve=solve,refinement=refine))
            del A,E,b,parts
            save()
    def record(label,x,subtract):return dict(label=label,coefficients=x,subtract_exact=subtract)
    collapsed=paired.high+paired.low
    records=[record('paired_error',paired.reconstructed,True),record('reference_error',reference,True),
        record('paired_reference_difference',accurate.difference(paired.reconstructed,reference),False),
        record('series_difference',accurate.difference(solutions[32],solutions[40]),False),
        record('collapsed_error',collapsed,True),
        record('collapsed_reference_difference',accurate.difference(collapsed,reference),False)]
    for label,x in baselines.items():
        records += [record(label+'_error',x,True),record(label+'_reference_difference',accurate.difference(x,reference),False)]
    if len(baselines)==2:
        records.append(record('baseline_assembly_difference',accurate.difference(baselines['q8'],baselines['q12']),False))
    fields=accurate.RetainedFields(mesh,space,center,c,records,{},basis,gate)
    r['norm_scope']='High-precision coefficient-field error identities. Paired online arithmetic checked separately.'
    for boost in (16,20):
        print('Paper high-p norms',spec['label'],boost,flush=True)
        er=energy.measure(mesh,space,center,c,assembly.patches,fields,boost)
        r['errors'].append(er); ff=er['fields']; ref=ff['reference_error']
        gate(f'e{boost}_patch_QR',er['patch_qr_orthogonality'],1e-12)
        gate(f'e{boost}_projection_reproduction',er['patch_polynomial_relative_residual'],1e-12)
        gate(f'e{boost}_disk_area',abs(er['quadrature']['volume_measure']-np.pi*c['radius']**2)/(np.pi*c['radius']**2),1e-10)
        gate(f'e{boost}_circumference',abs(er['quadrature']['boundary_measure']-2*np.pi*c['radius'])/(2*np.pi*c['radius']),1e-12)
        old=next(z for z in common.read(inputs/'reference-results.json')['errors'] if z['independent_boost']==boost)
        for i,norm in enumerate(('l2','gradient')):
            target=old['solutions'][1][i]
            gate(f'e{boost}_{norm}_reference_repeat',abs(metric(ref,norm)-target)/target,1e-8,'measurement')
        for norm in ('l2','gradient','energy'):
            denom=metric(ref,norm)
            for label in ('paired_reference_difference','series_difference'):
                gate(f'e{boost}_{label}_{norm}',metric(ff[label],norm)/denom,.01,'accuracy')
            for label in ['collapsed_reference_difference',*[key+'_reference_difference' for key in baselines]]:
                gate(f'e{boost}_{label}_{norm}',metric(ff[label],norm)/denom,.01,'baseline_accuracy')
            if len(baselines)==2:
                gate(f'e{boost}_baseline_assembly_difference_{norm}',metric(ff['baseline_assembly_difference'],norm)/denom,.01,'baseline_accuracy')
        if (inputs/'reference-energy.json').exists():
            target=common.read(inputs/'reference-energy.json')['energy_errors'][-1]
            gate(f'e{boost}_reference_energy_repeat',abs(ref['energy']-float(target['energy']))/float(target['energy']),1e-6,'measurement')
            for key,value in ref['squared_components'].items():
                truth=float(target['squared_components'][key])
                gate(f'e{boost}_reference_{key}_repeat',abs(value-truth)/truth,1e-5,'measurement')
        save()
    low,high=r['errors']
    for label,z in high['fields'].items():
        denom=high['fields']['reference_error']
        for norm in ('l2','gradient','energy'):
            d=metric(z,norm) if label.endswith('_error') else metric(denom,norm)
            gate(f'{label}_{norm}_integration_repeat',abs(metric(low['fields'][label],norm)-metric(z,norm))/max(d,1e-300),1e-4,'integration')
        for component in ('gradient','stabilization','boundary'):
            gate(f'{label}_{component}_squared_repeat',abs(low['fields'][label]['squared_components'][component]-z['squared_components'][component])/max(z['energy_squared'] if label.endswith('_error') else denom['energy_squared'],1e-300),1e-4,'integration')
    ref=high['fields']['reference_error']
    for key in ('global_storage_full_triangle_bounds','control_export_full_triangle_bounds'):
        for i,norm in enumerate(('l2','gradient')):
            gate(key+'_'+norm,D(paired.bounds[key][i])/D(str(metric(ref,norm))),1e-4,'storage')
    r.update(status='completed',seconds=time.perf_counter()-start,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    required=[g for g in r['gates'] if g['kind']!='baseline_accuracy']
    r['accurate_candidate_accepted']=all(g['passed'] for g in required)
    r['baseline_flags']=[g for g in r['gates'] if g['kind']=='baseline_accuracy' and not g['passed']]
    save();return r['accurate_candidate_accepted']


if __name__=='__main__':
    accepted=[]
    try:
        for spec in common.read(RUN/'parameters.json')['cases']: accepted.append(one(spec))
        common.write(RUN/'results.json',dict(status='completed',accepted=accepted,all_accepted=all(accepted)))
    except Exception:
        common.write(RUN/'failure.json',dict(traceback=traceback.format_exc()))
        raise
    sys.exit(0 if all(accepted) else 1)
