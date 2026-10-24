"""Experimental gamma0=0 gradient stabilization using sampled QR residuals.

For weighted FE gradient samples F and patch-polynomial gradient samples G,
Q=qr(G) gives S1=(F-Q(Q.T F)).T(F-Q(Q.T F)). This is the same H1-projection
gradient residual in exact arithmetic. It avoids expanding the squared norm
as a subtraction of Gram matrices. The original PSD cleanup is retained.
The mean fixes only the constant mode and therefore does not affect S1.
This helper does NOT implement the optional L2 residual stabilization.
"""
import numpy as np
import run_2d_disk_study as fem
import workstation_case as wc

def _gradient_stabilization(mesh,p,patches,element_dofs,element_basis,ndof,quadrature_boost=0,projection_basis='h1_orthonormal'):
    S1=np.zeros((ndof,ndof));patch_dofs_all=[];diagnostics=[]
    r,s,wref=fem.triangle_quadrature(max(6,p+3+quadrature_boost))
    indices=[(i,total-i) for total in range(p+1) for i in range(total+1)]
    cache={}
    for patch in patches:
        dofs=np.unique(np.concatenate([element_dofs[eid] for eid in patch]))
        patch_dofs_all.append(dofs);local_pos={int(g):i for i,g in enumerate(dofs)}
        vertices=np.vstack([mesh.elements[eid].xy for eid in patch])
        center=.5*(vertices.min(axis=0)+vertices.max(axis=0))
        scale=max(.5*float(np.max(vertices.max(axis=0)-vertices.min(axis=0))),.5*mesh.h)
        nrows=2*len(r)*len(patch)
        F=np.zeros((nrows,len(dofs)));G=np.zeros((nrows,len(indices)-1));offset=0
        for eid in patch:
            if eid not in cache:
                e=mesh.elements[eid]
                points=e.xy[0][None,:]+(e.J@np.vstack((r,s))).T
                _,grads=fem.physical_basis(e,element_basis[eid],points)
                cache[eid]=(points,grads,np.sqrt(e.detJ*wref))
            points,grads,sqrtw=cache[eid]
            _,qgrads=fem.eval_patch_polynomials(indices,points,center,scale)
            loc=np.asarray([local_pos[int(g)] for g in element_dofs[eid]])
            for axis in (0,1):
                end=offset+len(r)
                F[offset:end,loc]=(grads[:,axis,:]*sqrtw).T
                G[offset:end,:]=(qgrads[1:,axis,:]*sqrtw).T
                offset=end
        Q,R=np.linalg.qr(G,mode='reduced')
        residual=F-Q@(Q.T@F)
        raw=residual.T@residual
        cleanup={};local=fem.psd_cleanup(raw,cleanup)
        S1[np.ix_(dofs,dofs)]+=local
        diagnostics.append({'elements':len(patch),'dofs':len(dofs),
                            'sample_qr_orthogonality':float(np.linalg.norm(Q.T@Q-np.eye(Q.shape[1]),2)),
                            'gradient_R_condition':float(np.linalg.cond(R)),
                            's1_trace':float(np.trace(local)),
                            **{'s1_'+k:v for k,v in cleanup.items()}})
    return S1,np.zeros_like(S1),patch_dofs_all,diagnostics

def assemble(case,center):
    assert case['gamma0']==0.,'This experimental formulation supports gamma0=0 only.'
    original=fem.assemble_patch_stabilization
    try:
        fem.assemble_patch_stabilization=_gradient_stabilization
        problem=wc.assemble(case,center,'h1_orthonormal')
    finally:fem.assemble_patch_stabilization=original
    problem['stabilization_implementation']='sampled_qr_residual_gram_gamma0_zero'
    return problem
