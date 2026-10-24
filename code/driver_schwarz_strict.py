#!/usr/bin/env python3
"""Compare fixed saved q12 operators; independent field and energy validation."""
from pathlib import Path
import csv,json,resource,subprocess,sys,time
import numpy as np
RUN=None  # Supplied by run.py.
import workstation_case as wc
import run_2d_disk_study as fem
import extended_case_common as common
import paper_solver_2d as solver
import paper_energy as energy
import schwarz_strict as schwarz
from driver_sequence import difference,export_components

def main():
    spec=common.read(RUN/'parameters.json');saved=common.read(RUN/'inputs/operator-results.json')
    r=dict(status='running',phase='schwarz',case_id=spec['case_id'],case=saved['case'],gates=[],comparisons=[],errors=[])
    def save():common.write(RUN/'results.json',r)
    def gate(name,value,limit,kind='correctness'):
        value=float(value);assert np.isfinite(value),(name,value)
        g=dict(metric=name,actual=value,limit=limit,passed=value<=limit,kind=kind);r['gates'].append(g)
        if kind=='correctness' and not g['passed']:save();raise AssertionError(g)
    save()
    # Numerical adapter checks below run on every case.
    r['small_checks']=schwarz.small_checks()
    c=wc.normalize_case(saved['case']);mesh,center,space,basis,dofs=solver.setup(c)
    fingerprint=common.identity(mesh,space,center,c);common.write(RUN/'case-identity.json',fingerprint)
    gate('saved_case_identity',int(fingerprint!=common.read(RUN/'inputs/operator-identity.json')),0)
    operator=np.load(RUN/'inputs/operator.npz');A,E,b,reference=[operator[k] for k in ('A','E','rhs','coefficients')]
    gate('saved_operator_dimension',abs(len(A)-len(dofs)),0)
    scale=1/np.sqrt(np.diag(A));scaled=A*scale[:,None]*scale[None,:];rhs=scale*b
    selected=saved['errors'][-1]['fields']['online_q12_error'];r['ndof']=len(A);r['direct_errors']=selected
    boundary=sorted({z['element'] for z in saved['errors'][-1]['local_elements'] if z['region']=='boundary'})
    records=[dict(label='direct_error',coefficients=reference,subtract_exact=True)];solutions={};candidates={}
    for depth in [0,*spec['schwarz_depths']]:
        label='jacobi' if depth==0 else 'schwarz_L'+str(depth)
        print('Preconditioner',label,flush=True)
        if depth==0:B=np.eye(len(A));setup=dict(seconds=0.,regularization=0.,global_interior_solve=False,local_checks=[])
        else:
            patches=fem.build_patches(mesh,space['active'],boundary,depth)
            ids=[np.unique(np.concatenate([space['dofs'][eid] for eid in patch])) for patch in patches]
            B,setup=schwarz.build(scaled,ids)
        eig=schwarz.spectrum(scaled,B)
        gate(label+'_positive_spectrum',int(eig['lambda_min']<=0),0,'stability')
        gate(label+'_eigen_endpoint_residual',max(eig['endpoint_relative_residuals']),1e-5,'stability')
        gate(label+'_global_factorization',eig['cholesky_relative_residual'],1e-12)
        gate(label+'_local_factorizations',max([v['cholesky_relative_residual'] for v in setup['local_checks']],default=0.),1e-12)
        x,iteration=schwarz.pcg(scaled,rhs,B,A,b,scale,reference,E,selected['energy'])
        gate(label+'_PCG_converged',int(not iteration['converged']),0,'solve_accuracy')
        gate(label+'_true_residual',iteration['true_relative_residual'],1e-9,'solve_accuracy')
        gate(label+'_metric_field_difference',iteration['energy_difference_ratio'],1e-3,'solve_accuracy')
        r['comparisons'].append(dict(label=label,preconditioner_layers=depth,stabilization_layers=c['patch_layers'],setup=setup,spectrum=eig,pcg=iteration))
        solutions[label]=x;records.append(dict(label=label+'_difference',coefficients=difference(x,reference),subtract_exact=False));save()
    np.savez(RUN/'pcg-solutions.npz',**solutions)
    fields=energy.Fields(mesh,space,center,c,records,candidates,basis,gate)
    # Fields.sample permits an empty candidate list only through this stable
    # measurement path: all records here are exact stored coefficient fields.
    fields.sample=fields.stable
    patches=saved['assembly'][-1]['patches']
    for boost in (16,20):
        er=energy.measure(mesh,space,center,c,patches,fields,boost);r['errors'].append(er)
        gate(f'e{boost}_QR',er['patch_qr_orthogonality'],1e-12)
        for comparison in r['comparisons']:
            label=comparison['label'];v=er['fields'][label+'_difference']
            for norm,value,denom,limit in [('L2',v['l2'],selected['l2'],.01),('H1',v['component_norms']['gradient'],selected['component_norms']['gradient'],.01),('energy',v['energy'],selected['energy'],.001)]:
                gate(f'e{boost}_{label}_{norm}_field_difference',value/denom,limit,'accuracy')
        export_components(r);save()
    for label in fields.labels:
        a,b=[e['fields'][label] for e in r['errors']]
        for k in ('l2','energy'):gate(label+'_'+k+'_integration',abs(a[k]-b[k])/selected[k],1e-4,'integration')
    direct=r['errors'][-1]['fields']['direct_error'];old=saved['errors'][-1]['fields']['q12_error']
    for k in ('l2','energy'):gate('saved_direct_'+k+'_repeat',abs(direct[k]-old[k])/old[k],1e-8,'measurement')
    r['peak_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20;r['status']='completed';save()
if __name__=='__main__':main()
