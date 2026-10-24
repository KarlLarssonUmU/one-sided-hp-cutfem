"""Fixed sampled residuals: separate Gram, cleanup, and accumulation controls.

All samples, QR and projection residuals retain the original float64 arithmetic.
Extended means NumPy longdouble (64 mantissa bits on this host). Eigenpairs and
the 2e-12 cleanup threshold remain float64. Uncleaned controls change the
finite-precision operator and are diagnostics, not validated replacements.
"""
from decimal import Decimal, localcontext
import time
import numpy as np
import run_2d_disk_study as fem
import decimal_retained_system as ds

VARIANTS=('baseline','gram_extended','reconstruction_extended',
          'accumulation_extended','uncleaned64','uncleaned_extended')

def gram_extended(residual):
    # Contiguous rows make the inner products contiguous; no BLAS cast to f64.
    transposed=np.ascontiguousarray(residual.T,dtype=np.longdouble)
    return transposed@transposed.T

def cleanup_parts(raw):
    symmetric=.5*(raw+raw.T)
    eigenvalues,vectors=np.linalg.eigh(symmetric)
    scale=max(1.,float(np.max(np.abs(eigenvalues))))
    discarded=eigenvalues[eigenvalues<2e-12*scale]
    diag=dict(raw_min_eigenvalue=float(eigenvalues[0]),raw_spectral_scale=scale,
              clipped_modes=int(discarded.size),clip_threshold=2e-12*scale,
              discarded_absolute_trace=float(np.sum(np.abs(discarded))))
    eigenvalues[eigenvalues<2e-12*scale]=0.
    ordinary=(vectors*eigenvalues)@vectors.T
    v=vectors.astype(np.longdouble);w=eigenvalues.astype(np.longdouble)
    extended=(v*w)@v.T
    return ordinary,extended,diag

def assemble(mesh,p,patches,element_dofs,element_basis,ndof,quadrature_boost):
    matrices={name:np.zeros((ndof,ndof),dtype=np.longdouble if name in
              ('accumulation_extended','uncleaned_extended') else np.float64)
              for name in VARIANTS}
    r,s,wref=fem.triangle_quadrature(max(6,p+3+quadrature_boost))
    indices=[(i,total-i) for total in range(p+1) for i in range(total+1)]
    cache={};diagnostics=[]
    for patch_index,patch in enumerate(patches):
        tick=time.perf_counter()
        dofs=np.unique(np.concatenate([element_dofs[eid] for eid in patch]))
        local_pos={int(g):i for i,g in enumerate(dofs)}
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
        raw_ext=gram_extended(residual)
        ordinary,reconstructed,diag=cleanup_parts(raw)
        original_diag={};original=fem.psd_cleanup(raw,original_diag)
        assert np.array_equal(original,ordinary) and diag==original_diag
        gram_clean=fem.psd_cleanup(np.asarray(raw_ext,dtype=float))
        local=dict(baseline=ordinary,gram_extended=gram_clean,
                   reconstruction_extended=np.asarray(reconstructed,dtype=float),
                   accumulation_extended=ordinary,uncleaned64=raw,
                   uncleaned_extended=raw_ext)
        ix=np.ix_(dofs,dofs)
        for name,value in local.items():matrices[name][ix]+=value
        norm=float(np.linalg.norm(raw,'fro'))
        diag.update(patch=patch_index,elements=len(patch),dofs=len(dofs),rows=nrows,
                    qr_orthogonality=float(np.linalg.norm(Q.T@Q-np.eye(Q.shape[1]),2)),
                    Gram_change_relative=float(np.linalg.norm(raw_ext-raw,'fro')/norm),
                    cleanup_change_relative=float(np.linalg.norm(ordinary-raw,'fro')/norm),
                    reconstruction_change_relative=float(np.linalg.norm(reconstructed-ordinary,'fro')/norm),
                    seconds=time.perf_counter()-tick)
        diagnostics.append(diag)
        if patch_index%6==0:print('Stabilization patch',patch_index,'seconds',diag['seconds'],flush=True)
    return matrices,diagnostics

def self_check():
    """Decimal inner products and an analytic cleanup threshold boundary."""
    R=np.array([[1.,1e-8,-.7],[.3,-2.,.2],[1e-9,.1,3.],[.2,.9,-.4]])
    with localcontext() as ctx:
        ctx.prec=80
        exact=np.array([[str(sum((ds.exact_decimal(R[k,i])*ds.exact_decimal(R[k,j])
                                 for k in range(len(R))),Decimal(0)))
                         for j in range(3)] for i in range(3)],dtype=np.longdouble)
    error=float(np.max(np.abs(gram_extended(R)-exact))/np.max(np.abs(exact)))
    assert error<2e-18
    raw=np.diag([-1e-15,1e-13,.4,2.])
    ordinary,extended,diag=cleanup_parts(raw)
    expected=np.diag([0.,0.,.4,2.])
    assert np.array_equal(ordinary,expected) and np.array_equal(extended,expected)
    assert diag['clipped_modes']==2 and diag['clip_threshold']==4e-12
    return dict(Decimal_Gram_scaled_error=error,analytic_cleanup_threshold=True,
                float64_mantissa_bits=np.finfo(float).nmant+1,
                longdouble_mantissa_bits=np.finfo(np.longdouble).nmant+1)
