"""Precision controls on identical stored quadrature groups and frozen basis.

The contractions control promotes the candidate's rounded float64 physical
samples and loads before all local products/sums. The samples control retains
Horner values, gradient transforms and exp(x+y) in longdouble as well. Reference
coordinates, geometry, quadrature nodes/weights and basis coefficients stay
float64. Neither control evaluates or modifies the stabilization matrix.
"""
from decimal import Decimal as D, localcontext
import hashlib
import time
import numpy as np
import run_2d_disk_study as fem
import precision_eval_2d as pe

COMPONENTS=('K','M','Mb','Cn','fvol','fbc_cons','fbc_pen')
MODES=('contractions','samples')

def volume_kernel(v,g,w,f):
    v,g,w,f=(np.asarray(a,dtype=np.longdouble) for a in (v,g,w,f))
    return {'K':np.einsum('iaq,q,jaq->ij',g,w,g),
            'M':(v*w)@v.T,'fvol':v@(w*f)}

def boundary_kernel(v,g,n,w,u):
    v,g,n,w,u=(np.asarray(a,dtype=np.longdouble) for a in (v,g,n,w,u))
    dn=np.einsum('iaq,qa->iq',g,n)
    return {'Mb':(v*w)@v.T,'Cn':(dn*w)@v.T,
            'fbc_cons':-dn@(w*u),'fbc_pen':v@(w*u)}

def samples(problem,eid,points):
    e=problem['mesh'].elements[eid]
    r,s=fem.reference_coordinates(e,points)  # Same binary64 coordinates.
    v,gref=pe.horner(problem['element_basis'][eid],r,s,np.longdouble)
    v64=np.asarray(v,dtype=float)
    g64=np.einsum('ab,ibq->iaq',e.invJT,np.asarray(gref,dtype=float))
    g=np.einsum('ab,ibq->iaq',e.invJT.astype(np.longdouble),gref)
    u64,_,f64=fem.manufactured_solution_values(points,'exponential')
    u=np.exp(points[:,0].astype(np.longdouble)+points[:,1].astype(np.longdouble))
    return {'contractions':(v64,g64,u64,f64),'samples':(v,g,u,-2*u)}

def groups_digest(problem):
    h=hashlib.sha256()
    for key in ('volume_groups','boundary_groups'):
        h.update(key.encode())
        for eid,*arrays in problem[key]:
            h.update(str(eid).encode()+b':')
            for a in arrays:
                h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def reassemble(problem,space):
    assert problem['solution_kind']=='exponential'
    blocks={mode:{k:np.zeros_like(problem[k],dtype=np.longdouble) for k in COMPONENTS} for mode in MODES}
    before=groups_digest(problem);checked=set();maximum_sample_difference=0.
    tick=time.perf_counter()
    for phase,key in [('volume','volume_groups'),('boundary','boundary_groups')]:
        start=time.perf_counter()
        for group in problem[key]:
            eid,points=group[:2];idx=problem['element_dofs'][eid]
            vals=samples(problem,eid,points)
            if (phase,eid) not in checked:
                # Check the actual frozen candidate, including the gradient
                # transform and rounding, once for every participating element.
                with pe.evaluator(space,'horner_extended'):
                    v,g=fem.physical_basis(problem['mesh'].elements[eid],problem['element_basis'][eid],points)
                difference=max(float(np.max(np.abs(v-vals['contractions'][0]))),float(np.max(np.abs(g-vals['contractions'][1]))))
                maximum_sample_difference=max(maximum_sample_difference,difference)
                assert difference==0
                checked.add((phase,eid))
            for mode,(v,g,u,f) in vals.items():
                local=(volume_kernel(v,g,group[2],f) if phase=='volume' else boundary_kernel(v,g,group[2],group[3],u))
                for name,value in local.items():
                    blocks[mode][name][np.ix_(idx,idx) if value.ndim==2 else idx]+=value
        print('Extended local',phase,'seconds',time.perf_counter()-start,flush=True)
    assert groups_digest(problem)==before
    return blocks,dict(seconds=time.perf_counter()-tick,quadrature_sha256=before,
                       candidate_sample_checks=len(checked),sample_max_difference=maximum_sample_difference)

def kernel_check():
    """Independent 80-digit scalar products of the exact binary64 test inputs."""
    v=np.array([[1.,.1,-.3,2.,1e-4],[.7,-1.2,.4,.5,-.2],[1.1,.9,-.8,.3,.6]])
    g=np.stack((v*np.array([1.,-2.,3.,.5,1.]),v[::-1]*.7),axis=1)
    w=np.array([.031,.29,.7,.13,.002]);u=np.array([1.2,.8,1.5,.6,2.]);f=-2*u
    n=np.array([[.6,.8],[-.8,.6],[.3,.7],[-.6,.8],[1.,0.]])
    observed={**volume_kernel(v,g,w,f),**boundary_kernel(v,g,n,w,u)}
    with localcontext() as context:
        context.prec=80
        cv=lambda a:np.asarray([D.from_float(float(x)) for x in a.ravel()],dtype=object).reshape(a.shape)
        V,G,W,U,F,N=(cv(a) for a in (v,g,w,u,f,n));dn=np.empty(v.shape,dtype=object)
        for i in range(3):
            for q in range(5):dn[i,q]=sum(G[i,a,q]*N[q,a] for a in range(2))
        reference={k:np.empty(a.shape,dtype=object) for k,a in observed.items()}
        for i in range(3):
            for j in range(3):
                reference['K'][i,j]=sum(G[i,a,q]*W[q]*G[j,a,q] for a in range(2) for q in range(5))
                reference['M'][i,j]=reference['Mb'][i,j]=sum(V[i,q]*W[q]*V[j,q] for q in range(5))
                reference['Cn'][i,j]=sum(dn[i,q]*W[q]*V[j,q] for q in range(5))
            reference['fvol'][i]=sum(V[i,q]*W[q]*F[q] for q in range(5))
            reference['fbc_cons'][i]=-sum(dn[i,q]*W[q]*U[q] for q in range(5))
            reference['fbc_pen'][i]=sum(V[i,q]*W[q]*U[q] for q in range(5))
        errors={}
        for k,a in observed.items():
            ref=np.asarray([np.longdouble(str(x)) for x in reference[k].ravel()]).reshape(a.shape)
            errors[k]=float(np.max(np.abs(a-ref))/max(np.longdouble(1),np.max(np.abs(ref))))
    return dict(decimal_digits=80,max_scaled_error=max(errors.values()),component_errors=errors)
