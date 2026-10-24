"""Controlled evaluation of fixed 2D power coefficients; no basis rebuilding."""
from contextlib import contextmanager
from decimal import Decimal as D,localcontext
import hashlib
import json
import numpy as np
import run_2d_disk_study as fem
import residual_gram_2d as candidate

_COEFFICIENTS={}

def coefficients(polys,dtype):
    key=(id(polys),np.dtype(dtype).str)
    if key not in _COEFFICIENTS:
        degree=max(i+j for poly in polys for i,j in poly)
        a=np.zeros((len(polys),degree+1,degree+1),dtype=dtype)
        for k,poly in enumerate(polys):
            for (i,j),c in poly.items():a[k,i,j]=c
        _COEFFICIENTS[key]=(polys,a) # retain the list, preventing id reuse
    return _COEFFICIENTS[key][1]

def horner(polys,r,s,dtype=np.float64):
    a=coefficients(polys,dtype);r=np.asarray(r,dtype=dtype);s=np.asarray(s,dtype=dtype)
    values=np.zeros((len(polys),r.size),dtype=dtype)
    dr=np.zeros_like(values);ds=np.zeros_like(values)
    degree=a.shape[1]-1
    for i in range(degree,-1,-1):
        row=np.zeros_like(values);drow=np.zeros_like(values)
        for j in range(degree-i,-1,-1):
            drow=drow*s+row
            row=row*s+a[:,i,j,None]
        dr=dr*r+values
        values=values*r+row
        ds=ds*r+drow
    return values,np.stack((dr,ds),axis=1)

def point_check(p=8):
    polys=fem.reference_basis(p,[0,1,2])
    r=np.array([.13,.71,.499,.001,.998]);s=np.array([.27,.2,.499,.998,.001])
    reference=np.zeros((3,len(polys),len(r)),dtype=np.longdouble)
    with localcontext() as context:
        context.prec=80
        for k,poly in enumerate(polys):
            for q,(x,y) in enumerate(zip(r,s)):
                x=D.from_float(float(x));y=D.from_float(float(y))
                v=dr=ds=D(0)
                for (i,j),c in poly.items():
                    c=D.from_float(c);v+=c*x**i*y**j
                    if i:dr+=D(i)*c*x**(i-1)*y**j
                    if j:ds+=D(j)*c*x**i*y**(j-1)
                reference[:,k,q]=[np.longdouble(str(z)) for z in (v,dr,ds)]
    scales=np.maximum(1,np.max(np.abs(reference),axis=2))[:,:,None]
    errors={}
    for name,fn in [('original',fem.eval_polys),('horner64',lambda p,r,s:horner(p,r,s,np.float64)),('horner_extended',lambda p,r,s:horner(p,r,s,np.longdouble))]:
        v,g=fn(polys,r,s);observed=np.stack((v,g[:,0],g[:,1]))
        errors[name]=float(np.max(np.abs(observed-reference)/scales))
    return dict(degree=p,decimal_precision=80,errors=errors,
                float64_mantissa_bits=np.finfo(np.float64).nmant+1,
                longdouble_mantissa_bits=np.finfo(np.longdouble).nmant+1,
                longdouble_epsilon=str(np.finfo(np.longdouble).eps))

def frozen_basis(mesh,center,radius,p):
    active,_,_=fem.classify_elements(mesh,center,radius)
    _,dofs,basis=fem.build_dofs(mesh,active,p)
    patterns={}
    for eid in active:
        v=mesh.elements[eid].vids;key=(v[0]<v[1],v[1]<v[2],v[2]<v[0])
        patterns[key]=basis[eid]
    return dict(active=active,dofs=dofs,basis=basis,patterns=patterns,p=p)

@contextmanager
def evaluator(space,mode):
    old_basis=fem.reference_basis;old_eval=fem.eval_polys
    def fixed(p,vids):
        assert p==space['p']
        key=(vids[0]<vids[1],vids[1]<vids[2],vids[2]<vids[0])
        return space['patterns'][key]
    try:
        fem.reference_basis=fixed
        if mode!='original':
            assert mode in ('horner64','horner_extended')
            dtype=np.float64 if mode=='horner64' else np.longdouble
            def evaluate(polys,r,s):
                v,g=horner(polys,r,s,dtype)
                # Only polynomial evaluation precision changes; physical
                # gradient transforms, local integrals and QR remain float64.
                return np.asarray(v,dtype=float),np.asarray(g,dtype=float)
            fem.eval_polys=evaluate
        yield
    finally:fem.reference_basis=old_basis;fem.eval_polys=old_eval

def assemble(case,center,space,mode):
    with evaluator(space,mode):problem=candidate.assemble(case,center)
    assert problem['active']==space['active']
    for eid in space['active']:
        np.testing.assert_array_equal(problem['element_dofs'][eid],space['dofs'][eid])
        assert problem['element_basis'][eid]==space['basis'][eid]
    return problem

def refined_solve(A,b,steps=5):
    x=np.linalg.solve(A,b);extended=x.astype(np.longdouble)
    Aext=A.astype(np.longdouble);bext=b.astype(np.longdouble)
    history=[]
    for i in range(steps):
        residual=bext-Aext@extended
        history.append(float(np.sqrt(np.sum(residual**2))/np.sqrt(np.sum(bext**2))))
        correction=np.linalg.solve(A,np.asarray(residual,dtype=float))
        extended+=correction.astype(np.longdouble)
    residual=bext-Aext@extended
    history.append(float(np.sqrt(np.sum(residual**2))/np.sqrt(np.sum(bext**2))))
    return x,extended,history

def measure_extended(problem,coefficients,groups,pairs):
    """Common extended-precision evaluation of the frozen coefficient basis.

    Quadrature nodes/weights and reference coordinates remain float64; all
    polynomial values, coefficient contractions and error sums use longdouble.
    """
    C=np.asarray(coefficients,dtype=np.longdouble)
    sums=np.zeros((len(C)+len(pairs),2),dtype=np.longdouble)
    for eid,points,w in groups:
        e=problem['mesh'].elements[eid];r,s=fem.reference_coordinates(e,points)
        v,gref=horner(problem['element_basis'][eid],r,s,np.longdouble)
        g=np.einsum('ab,ibq->iaq',e.invJT.astype(np.longdouble),gref)
        local=C[:,problem['element_dofs'][eid]]
        values=local@v;grad=np.einsum('ki,iaq->kqa',local,g)
        exact=np.exp(points[:,0].astype(np.longdouble)+points[:,1].astype(np.longdouble))
        weight=w.astype(np.longdouble)
        sums[:len(C),0]+=((values-exact)**2)@weight
        sums[:len(C),1]+=np.sum((grad-exact[:,None])**2,axis=2)@weight
        for k,(a,b) in enumerate(pairs):
            delta=local[a]-local[b];dv=delta@v;dg=np.einsum('i,iaq->qa',delta,g)
            sums[len(C)+k,0]+=np.dot(weight,dv**2)
            sums[len(C)+k,1]+=np.dot(weight,np.sum(dg**2,axis=1))
    return np.asarray(np.sqrt(sums),dtype=float)
