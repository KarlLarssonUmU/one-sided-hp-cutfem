"""Unshifted scaled additive Schwarz and explicitly verified PCG comparisons."""
import time
import numpy as np

def build(A,patch_dofs):
    start=time.perf_counter();n=len(A);coverage=np.zeros(n,dtype=int);seen=set();domains=[]
    for ids in patch_dofs:
        ids=np.unique(ids);key=tuple(int(i) for i in ids)
        if key not in seen:seen.add(key);domains.append(ids);coverage[ids]+=1
    interior=np.flatnonzero(coverage==0)
    if len(interior):domains.append(interior);coverage[interior]+=1
    assert np.all(coverage>0)
    B=np.zeros_like(A);checks=[]
    for ids in domains:
        local=A[np.ix_(ids,ids)];d=1/np.sqrt(np.diag(local));scaled=local*d[:,None]*d[None,:]
        L=np.linalg.cholesky(scaled)
        invL=np.linalg.solve(L,np.eye(len(ids)));inverse=(invL.T@invL)*d[:,None]*d[None,:]
        weights=1/np.sqrt(coverage[ids]);B[np.ix_(ids,ids)]+=inverse*weights[:,None]*weights[None,:]
        checks.append(dict(size=len(ids),shift=0.,cholesky_relative_residual=float(np.linalg.norm(L@L.T-scaled,np.inf)/np.linalg.norm(scaled,np.inf))))
    B=(B+B.T)/2
    return B,dict(seconds=time.perf_counter()-start,subdomains=len(domains),uncovered_interior_size=len(interior),
        maximum_overlap=int(max(coverage)),maximum_local_size=max(len(v) for v in domains),local_checks=checks,
        regularization=0.,global_interior_solve=bool(len(interior)))

def spectrum(A,B):
    start=time.perf_counter();L=np.linalg.cholesky(B);H=L.T@A@L;H=(H+H.T)/2
    eigen,V=np.linalg.eigh(H);checks=[];ld=np.longdouble
    # Validate the endpoint eigenpairs against the product, avoiding an O(n^3)
    # extended-precision matrix product solely for checking the transform.
    ll=L.astype(ld);aa=A.astype(ld)
    for k in (0,-1):
        v=V[:,k].astype(ld);res=ll.T@(aa@(ll@v))-ld(eigen[k])*v
        checks.append(float(np.linalg.norm(res)))
    return dict(seconds=time.perf_counter()-start,lambda_min=float(eigen[0]),lambda_max=float(eigen[-1]),
        condition=float(eigen[-1]/eigen[0]),endpoint_absolute_residuals=checks,
        endpoint_relative_residuals=[checks[0]/abs(float(eigen[0])),checks[1]/abs(float(eigen[-1]))],
        cholesky_relative_residual=float(np.linalg.norm(L@L.T-B,np.inf)/np.linalg.norm(B,np.inf)),shift=0.)

def pcg(A,b,B,original_A,original_b,scale,reference,E,discretization_energy,maxiter=3000):
    start=time.perf_counter();x=np.zeros_like(b);r=b.copy();apply_seconds=0.;applies=0
    def apply(v):
        nonlocal apply_seconds,applies
        t=time.perf_counter();out=B@v;apply_seconds+=time.perf_counter()-t;applies+=1;return out
    z=apply(r);direction=z.copy();rz=float(r@z);bscale=np.linalg.norm(b)
    ld=np.longdouble;oa=original_A.astype(ld);ob=original_b.astype(ld);ee=E.astype(ld)
    reference=reference.astype(ld);normb=np.linalg.norm(ob);history=[];first=None;breakdown=None;converged=False
    for iteration in range(1,maxiter+1):
        Ad=A@direction;curvature=float(direction@Ad)
        if not np.isfinite([curvature,rz]).all() or min(curvature,rz)<=0:
            breakdown='nonpositive_or_nonfinite_curvature';break
        alpha=rz/curvature;x+=alpha*direction;r-=alpha*Ad
        recursive=float(np.linalg.norm(r)/bscale)
        if iteration%10==0 or recursive<=1e-9 or iteration==maxiter:
            physical=scale*x;xx=physical.astype(ld);true=float(np.linalg.norm(ob-oa@xx)/normb)
            diff=xx-reference;squared=diff@(ee@diff);assert squared>=0
            ratio=float(np.sqrt(squared)/discretization_energy)
            checkpoint=dict(iteration=iteration,true_relative_residual=true,recursive_relative_residual=recursive,energy_difference_ratio=ratio)
            history.append(checkpoint)
            if true<=1e-9 and first is None:first=dict(checkpoint)
            if true<=1e-9 and ratio<=1e-3:converged=True;break
            # Replace drifting recursive residuals; retain a regular CG step
            # otherwise. No direct-solution correction is used.
            fresh=b-A@x
            if np.linalg.norm(fresh-r)>max(.1*np.linalg.norm(fresh),1e-30):
                r=fresh;z=apply(r);direction=z.copy();rz=float(r@z);continue
        z=apply(r);new=float(r@z);direction=z+(new/rz)*direction;rz=new
    physical=scale*x;xx=physical.astype(ld);diff=xx-reference
    final=dict(iterations=iteration,true_relative_residual=float(np.linalg.norm(ob-oa@xx)/normb),
        energy_difference_ratio=float(np.sqrt(diff@(ee@diff))/discretization_energy),converged=converged,breakdown=breakdown,
        first_residual_1e9=first,history=history,seconds=time.perf_counter()-start,apply_seconds=apply_seconds,applications=applies,
        stop_policy='True original-system relative residual <=1e-9 AND saved-metric field discrepancy/direct discretization energy <=1e-3. The latter is a validation stopping rule requiring a reference solution.')
    return physical,final

def small_checks():
    A=np.array([[2.,-1.],[-1.,2.]])
    B,z=build(A,[np.array([0,1])]);assert np.max(abs(B@A-np.eye(2)))<1e-14
    s=spectrum(A,B);assert abs(s['condition']-1)<1e-13
    x,z=pcg(A,np.array([1.,0.]),B,A,np.array([1.,0.]),np.ones(2),np.array([2/3,1/3]),A,1.)
    assert z['converged'] and z['iterations']==1 and np.max(abs(x-[2/3,1/3]))<1e-14
    return dict(single_domain_inverse=True,identity_preconditioned_spectrum=True,one_step_exact_pcg=True)
