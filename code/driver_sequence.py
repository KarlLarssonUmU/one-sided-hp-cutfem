#!/usr/bin/env python3
"""One cut or finite-regularity case with the frozen practical method."""
from decimal import Decimal as D,localcontext
from pathlib import Path
import csv,json,resource,subprocess,sys,time
import numpy as np
RUN=None  # Supplied by run.py.
import workstation_case as wc
import run_2d_disk_study as fem
import extended_case_common as common
import extended_retained_geometry_2d as retained_geometry
import paper_solver_2d as solver
import paper_energy as energy
import paper_stability as stability
import paper_adapter_checks as checks
import saved_refinement_solver as refinement


def difference(a,b):
    with localcontext() as ctx:
        ctx.prec=80
        def cv(x):return x if isinstance(x,D) else D.from_float(float(x))
        return np.array([cv(x)-cv(y) for x,y in zip(a,b)],dtype=object)


def export_components(result):
    """Portable long-form tables; component square roots must not be summed."""
    rows=[];local=[];c=result['case']
    for er in result['errors']:
        for label,z in er['fields'].items():
            for component,value in z['squared_components'].items():
                rawkey={'gradient':'gradient_squared','stabilization':'stabilization_squared','boundary':'boundary_l2_squared'}[component]
                rows.append(dict(case_id=result['case_id'],N=c['N'],p=c['p'],disk=c['path'],error_boost=er['boost'],
                    field=label,component=component,raw_squared=z['raw_integrals'][rawkey],weight=z['weights'][component],
                    weighted_squared=value,component_norm=z['component_norms'][component],squared_share=z['squared_shares'][component],
                    energy_squared=z['energy_squared'],energy=z['energy'],L2_error=z['l2']))
        for x in er['local_elements']:
            component='gradient' if x['region']=='volume' else 'boundary'
            weight=1. if component=='gradient' else c['p']**2/(2/c['N'])
            raw=x['gradient_squared'] if component=='gradient' else x['value_squared']
            local.append(dict(case_id=result['case_id'],error_boost=er['boost'],field=x['field'],component=component,
                entity='element',entity_id=x['element'],raw_squared=raw,weight=weight,weighted_squared=raw*weight,
                L2_squared=x['value_squared'] if component=='gradient' else ''))
        for x in er['local_patches']:
            local.append(dict(case_id=result['case_id'],error_boost=er['boost'],field=x['field'],component='stabilization',
                entity='patch',entity_id=x['patch'],raw_squared=x['raw_stabilization_squared'],weight=c['gamma1'],
                weighted_squared=x['weighted_stabilization_squared'],L2_squared=''))
    for name,data in [('energy-components.csv',rows),('energy-local.csv',local)]:
        with (RUN/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)


def main():
    spec=common.read(RUN/'parameters.json');r=dict(status='running',phase=spec['phase'],case_id=spec['case_id'],gates=[],assembly=[],errors=[])
    def save():common.write(RUN/'results.json',r)
    def gate(name,value,limit,kind='correctness'):
        value=float(value);assert np.isfinite(value),(name,value)
        g=dict(metric=name,actual=value,limit=limit,passed=value<=limit,kind=kind);r['gates'].append(g)
        if kind=='correctness' and not g['passed']:save();raise AssertionError(g)
    save()
    # Numerical adapter checks below run on every case.
    r['adapter_checks']=checks.check()
    c=wc.normalize_case(spec['case']);tick=time.perf_counter();mesh,center,space,basis,dofs=solver.setup(c)
    r.update(case=c,ndof=len(dofs),setup_seconds=time.perf_counter()-tick)
    gate('frozen_center',int(list(center)!=spec['case']['center']),0)
    gate('strict_classifier_matches_Decimal',int(solver.classify(mesh,center,c['radius'])!=retained_geometry.classify(mesh,center,c['radius'])),0)
    fingerprint=common.identity(mesh,space,center,c);common.write(RUN/'case-identity.json',fingerprint);r['case_identity']=fingerprint['sha256']
    bc=solver.basis_checks(mesh,space,basis,dofs);r['basis_checks']=bc
    gate('basis_Decimal_scaled_error',bc['Decimal_basis_scaled_error'],5e-12)
    gate('basis_linear_reproduction',bc['linear_reproduction_error'],5e-13)
    gate('basis_exact_conformity',bc['nonconforming_edges'],0)
    areas=fem.geometry_area_diagnostic(mesh,center,c['radius'],space['active'])
    r['geometry']=dict(physical_cell_areas=areas,minimum_positive_fraction=min(v/(mesh.h*mesh.h/2) for v in areas.values() if v>0),
        nonpositive_active_areas=sum(v<=0 for v in areas.values()),active_elements=space['active'],cut_elements=space['cut'])
    gate('nonpositive_active_cell_measure',r['geometry']['nonpositive_active_areas'],0)
    reference=None
    if 'reference-solution.json' in spec['inputs']:
        gate('reference_case_identity',int(fingerprint['sha256']!=common.read(RUN/'inputs/reference-identity.json')['sha256']),0)
        reference=np.array([D(x) for x in common.read(RUN/'inputs/reference-solution.json')],dtype=object)
    candidates={};last_updates={};previous_patches=None
    for boost in spec['assembly_boosts']:
        print('Paper assembly',r['case_id'],boost,flush=True);tick=time.perf_counter()
        A,E,b,parts,z=solver.assemble(mesh,space,center,c,basis,boost);z.update(boost=boost,assembly_seconds=time.perf_counter()-tick)
        gate(f'q{boost}_asymmetry',z['relative_asymmetry'],1e-13)
        gate(f'q{boost}_patch_QR_orthogonality',max(d['qr_orthogonality'] for d in z['patch_diagnostics']),1e-12)
        gate(f'q{boost}_cell_area',max(abs(z['area'][eid]-areas[eid])/(mesh.h*mesh.h/2) for eid in space['active']),1e-7)
        if previous_patches is not None:gate('assembly_same_patches',int(previous_patches!=z['patches']),0)
        previous_patches=z['patches']
        vertices=[np.vstack([mesh.elements[eid].xy for eid in patch]) for patch in z['patches']]
        z['patch_geometry']=dict(element_counts=[len(v) for v in z['patches']],
            diameters=[float(np.max(np.linalg.norm(v[:,None,:]-v[None,:,:],axis=2))) for v in vertices],
            element_overlap={str(eid):sum(eid in patch for patch in z['patches']) for eid in space['active']})
        initial,z['initial_solve']=solver.solve(A,b)
        x,z['refinement'],last,rounding=refinement.refine(A,b,initial,6)
        candidates[f'q{boost}']=x;last_updates[f'q{boost}']=last
        gate(f'q{boost}_refined_internal_backward',z['refinement']['history'][-1]['backward_error'],5e-19,'solve_accuracy')
        ld=np.longdouble;aa=A.astype(ld);xx=x.astype(ld);bb=b.astype(ld)
        backward=float(np.max(abs(bb-aa@xx))/(np.max(np.sum(abs(aa),axis=1))*np.max(abs(xx))+np.max(abs(bb))))
        z['rounded_backward_error']=backward;gate(f'q{boost}_rounded_backward',backward,1e-13,'solve_accuracy')
        np.savez(RUN/f'q{boost}-paper.npz',A=A,E=E,rhs=b,coefficients=x,initial=initial,last_update=last,rounding=rounding,**parts)
        z['spectrum']=solver.spectrum(A,E);eig=z['spectrum']
        gate(f'q{boost}_positive_coercivity',int(eig['lambda_min']<=1e-6),0,'stability')
        for name in ('whitened_metric_error','whitened_residual'):gate(f'q{boost}_{name}',eig[name],1e-6,'stability')
        z['threshold']=stability.threshold(parts,E,c['p'],mesh.h,c['gamma1']);t=z['threshold']
        gate(f'q{boost}_threshold_bracket_width',t['width']/max(1,t['upper']),1e-4,'stability')
        gate(f'q{boost}_threshold_endpoint_residual',max(v['whitened_residual'] for v in t['endpoints']),1e-6,'stability')
        lo,hi=t['endpoints'];signs=(lo['lambda_min']<=t['target']-5*lo['whitened_residual'] and hi['lambda_min']>t['target']+5*hi['whitened_residual'])
        gate(f'q{boost}_threshold_resolved_signs',int(not signs),0,'stability')
        r['assembly'].append(z);save()
        del A,E,b,parts,aa,xx,bb
    t0,t1=[z['threshold'] for z in r['assembly']]
    gate('threshold_assembly_repeat',abs((t0['lower']+t0['upper'])-(t1['lower']+t1['upper']))/max(t1['lower']+t1['upper'],1e-12),1e-3,'stability')
    records=[dict(label=key+'_error',coefficients=x,subtract_exact=True) for key,x in candidates.items()]
    records.append(dict(label='assembly_difference',coefficients=difference(candidates[f'q{spec["assembly_boosts"][0]}'],candidates[f'q{spec["assembly_boosts"][1]}']),subtract_exact=False))
    for key,x in last_updates.items():records.append(dict(label=key+'_last_update',coefficients=x,subtract_exact=False))
    if reference is not None:
        records.append(dict(label='reference_error',coefficients=reference,subtract_exact=True))
        for key,x in candidates.items():records.append(dict(label=key+'_reference_difference',coefficients=difference(x,reference),subtract_exact=False))
    np.savez(RUN/'solutions.npz',**candidates,**{k+'_last_update':v for k,v in last_updates.items()})
    tick=time.perf_counter();fields=energy.Fields(mesh,space,center,c,records,candidates,basis,gate)
    r['measurement_setup_seconds']=time.perf_counter()-tick;r['field_labels']=fields.labels
    for boost in spec['error_boosts']:
        print('Energy components',r['case_id'],boost,flush=True)
        er=energy.measure(mesh,space,center,c,previous_patches,fields,boost);r['errors'].append(er)
        gate(f'e{boost}_patch_QR',er['patch_qr_orthogonality'],1e-12)
        gate(f'e{boost}_projection_polynomial_reproduction',er['patch_polynomial_relative_residual'],1e-12)
        gate(f'e{boost}_disk_area',abs(er['quadrature']['volume_measure']-np.pi*c['radius']**2)/(np.pi*c['radius']**2),1e-10)
        gate(f'e{boost}_circumference',abs(er['quadrature']['boundary_measure']-2*np.pi*c['radius'])/(2*np.pi*c['radius']),1e-12)
        ff=er['fields'];ref=ff.get('reference_error')
        for norm,extract in [('L2',lambda z:z['l2']),('H1',lambda z:z['component_norms']['gradient']),('energy',lambda z:z['energy'])]:
            denom=min(extract(ff[key+'_error']) for key in candidates)
            gate(f'e{boost}_{norm}_assembly_difference',extract(ff['assembly_difference'])/denom,.01,'accuracy')
            for key in candidates:
                gate(f'e{boost}_{key}_{norm}_last_update',extract(ff[key+'_last_update'])/denom,1e-4,'solve_accuracy')
                gate(f'e{boost}_{key}_{norm}_online_norm_change',abs(extract(ff['online_'+key+'_error'])-extract(ff[key+'_error']))/denom,.01,'accuracy')
                if ref:
                    for label in (key+'_reference_difference','online_'+key+'_reference_difference'):
                        gate(f'e{boost}_{label}_{norm}',extract(ff[label])/extract(ref),.01,'accuracy')
        if ref:
            baseline=common.read(RUN/'inputs/reference-results.json');old=next(z for z in baseline['errors'] if z['independent_boost']==boost)
            gate(f'e{boost}_original_volume_nodes',int(er['quadrature_digests']['volume']!=old['ordered_groups_sha256']),0)
            for k,(name,value) in enumerate([('L2',ref['l2']),('H1',ref['component_norms']['gradient'])]):
                gate(f'e{boost}_{name}_reference_repeat',abs(value-old['solutions'][1][k])/old['solutions'][1][k],1e-8,'measurement')
            exact=common.read(RUN/'inputs/reference-energy.json')['energy_errors'][-1]
            for name,value in ref['squared_components'].items():
                target=float(exact['squared_components'][name]);gate(f'e{boost}_reference_{name}_squared_repeat',abs(value-target)/target,1e-5,'measurement')
            gate(f'e{boost}_reference_energy_repeat',abs(ref['energy']-float(exact['energy']))/float(exact['energy']),1e-6,'measurement')
        export_components(r);save()
        print('Selected online components',ff[f'online_q{spec["assembly_boosts"][-1]}_error'],flush=True)
    lower,upper=r['errors']
    for label in fields.labels:
        lo,hi=(row['fields'][label] for row in (lower,upper))
        reference_error=upper['fields'].get('reference_error',upper['fields'][f'q{spec["assembly_boosts"][-1]}_error'])
        is_error=label.endswith('_error') and not label.startswith('online_')
        for key in ('l2','energy'):
            denom=hi[key] if is_error else reference_error[key]
            gate(f'{label}_{key}_integration_repeat',abs(lo[key]-hi[key])/max(denom,1e-300),1e-4,'integration')
        for component in ('gradient','stabilization','boundary'):
            denom=hi['energy_squared'] if is_error else reference_error['energy_squared']
            gate(f'{label}_{component}_component_repeat',abs(lo['squared_components'][component]-hi['squared_components'][component])/max(denom,1e-300),1e-4,'integration')
    r['peak_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20;r['status']='completed';save()

if __name__=='__main__':main()
