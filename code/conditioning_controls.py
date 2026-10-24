"""Paired cached-matrix control: vary stabilization weight and patch depth."""
from pathlib import Path
import json, sys, time, platform, getpass, os
import numpy as np
import scipy
from scipy import linalg as la

run=Path(sys.argv[1]).resolve()
# Run through the portable entry point; no host-specific requirement.
dump=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
spec=json.loads((run/'parameters.json').read_text())
dump(run/'environment.json',dict(host=platform.node(),user=getpass.getuser(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,threads=int(os.environ.get('OPENBLAS_NUM_THREADS','1')),device='CPU',dtype='float64; longdouble residuals',seed=None))
start=time.time();status=1;rows=[]
try:
    for layers in [2,4]:
        for gamma in [.1,1.]:
            row=dict(p=6,N=6,path='axis',epsilon=3e-5,L=layers,gamma=gamma,beta=16.,rules=[])
            for boost in [8,12]:
                with np.load(run/'inputs'/f'L{layers}-q{boost}.npz') as f:
                    A=f['A']+(gamma-.1)*f['S1']; E=f['E']+(gamma-.1)*f['S1']
                    assert np.max(abs(A-A.T)) < 1e-10*max(1.,np.max(abs(A)))
                    if gamma==.1:assert np.array_equal(A,f['A'])
                A=(A+A.T)/2;E=(E+E.T)/2
                d=1/np.sqrt(np.diag(A));J=A*d[:,None]*d
                values,vectors=la.eigh(J,check_finite=False)
                assert values[0]>0
                ld=np.longdouble;JJ=J.astype(ld)
                residuals=[]
                for i in [0,-1]:
                    v=vectors[:,i].astype(ld);r=JJ@v-ld(values[i])*v
                    residuals.append(float(np.sqrt(r@r)/abs(values[i])))
                assert max(residuals)<1e-5,residuals
                d=1/np.sqrt(np.diag(E));G=E*d[:,None]*d
                L=la.cholesky(G,lower=True);Q=la.solve_triangular(L,np.eye(len(L)),lower=True)
                whitening=float(np.linalg.norm(Q@G@Q.T-np.eye(len(L)),ord=np.inf))
                assert whitening<1e-5,whitening
                W=Q@(A*d[:,None]*d)@Q.T;W=(W+W.T)/2
                eig=la.eigvalsh(W,subset_by_index=[0,0],check_finite=False)[0]
                assert eig>0
                row['rules'].append(dict(boost=boost,kappa_J=float(values[-1]/values[0]),lambda_min_J=float(values[0]),lambda_max_J=float(values[-1]),relative_endpoint_residuals=residuals,nominal_coercivity=float(eig),metric_whitening_error=whitening))
            for key in ['kappa_J','nominal_coercivity']:
                change=abs(row['rules'][0][key]/row['rules'][1][key]-1)
                assert change<1e-4,(layers,gamma,key,change)
                row[key+'_paired_relative_change']=change
            row['resolved']=True;rows.append(row);dump(run/'output/results.json',rows)
    assert len(rows)==4
    # Independent agreement with the already published L2/Jacobi comparison.
    baseline=rows[0]['rules'][-1]['kappa_J']
    assert abs(baseline/spec['previous_L2_kappa_J']-1)<1e-4
    status=0
finally:
    dump(run/'completion.json',dict(exit_code=status,job_id=os.environ.get('SLURM_JOB_ID'),seconds=time.time()-start,started=start,finished=time.time(),completed_cases=len(rows)))
