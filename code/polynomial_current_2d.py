"""P8 manufactured loads using the current extended samples and fixed operators.

Only the manufactured values change. The exponential control repeats the saved
load components with identical sample, contraction and global summation order.
Existing polynomial representation and field integration are reused unchanged;
the exact-data callback uses longdouble division for every Taylor coefficient.
"""
from contextlib import contextmanager
from decimal import Decimal as D, localcontext
from math import factorial
import numpy as np
import decimal_retained_system as ds
import local_integrals_2d as li
import polynomial_probe_2d as poly

def taylor(t,degree):
    t=np.asarray(t,dtype=np.longdouble)
    value=np.full_like(t,np.longdouble(1)/factorial(degree))
    for k in range(degree-1,-1,-1):value=value*t+np.longdouble(1)/factorial(k)
    return value

def data(points,*unused,**kwargs):
    points=np.asarray(points,dtype=np.longdouble);t=points[:,0]+points[:,1]
    value=taylor(t,8);gradient=taylor(t,7)
    return value,np.column_stack((gradient,gradient)),-2*taylor(t,6)

def formula_check():
    points=np.array([[.13,.27],[.71,.2],[-.6,.1],[0.,0.],[.4,-.2],[-.9,-.3]],dtype=np.longdouble)
    u,g,f=data(points);actual=np.column_stack((u,g[:,0],f))
    with localcontext() as ctx:
        ctx.prec=80;reference=[]
        for x,y in points:
            t=ds.exact_decimal(x)+ds.exact_decimal(y)
            def p(n):return sum(((t**k if k else D(1))/D(factorial(k)) for k in range(n+1)),D(0))
            reference.append([str(v) for v in (p(8),p(7),-2*p(6))])
    expected=np.asarray(reference,dtype=np.longdouble)
    error=float(np.max(np.abs(actual-expected)/np.maximum(1,np.abs(expected))))
    assert error<=5e-18
    return dict(decimal_digits=80,max_scaled_error=error,points=len(points),all_Taylor_coefficients_divided_in_longdouble=True)

def load_components(problem):
    """Same three load contractions as local_integrals_2d, shared samples."""
    n=1+max(int(np.max(d)) for d in problem['element_dofs'].values())
    blocks={mode:{key:np.zeros(n,dtype=np.longdouble) for key in ('fvol','fbc_cons','fbc_pen')} for mode in ('exponential','polynomial')}
    before=li.groups_digest(problem)
    for phase,key in [('volume','volume_groups'),('boundary','boundary_groups')]:
        for group in problem[key]:
            eid,points=group[:2];idx=problem['element_dofs'][eid]
            v,g,u,f=li.samples(problem,eid,points)['samples']
            pu,_,pf=data(points)
            if phase=='volume':
                w=np.asarray(group[2],dtype=np.longdouble)
                for mode,force in [('exponential',f),('polynomial',pf)]:blocks[mode]['fvol'][idx]+=v@(w*force)
            else:
                normals=np.asarray(group[2],dtype=np.longdouble);w=np.asarray(group[3],dtype=np.longdouble)
                dn=np.einsum('iaq,qa->iq',g,normals)
                for mode,value in [('exponential',u),('polynomial',pu)]:
                    blocks[mode]['fbc_cons'][idx]+=-dn@(w*value)
                    blocks[mode]['fbc_pen'][idx]+=v@(w*value)
    assert li.groups_digest(problem)==before
    return blocks,before

@contextmanager
def polynomial_measurement():
    previous=poly.data
    try:
        poly.data=data
        yield
    finally:poly.data=previous

def reference_residual(A,b,x):
    """80-digit residual of the exact stored A,b and rounded reference field."""
    with localcontext() as ctx:
        ctx.prec=80;X=[ds.exact_decimal(v) for v in x];B=[ds.exact_decimal(v) for v in b]
        residual=[sum((ds.exact_decimal(row[j])*X[j] for j in np.flatnonzero(row)),D(0))-B[i] for i,row in enumerate(A)]
        ratio=max(map(abs,residual))/max(map(abs,B))
    return dict(decimal_digits=80,relative_infinity_residual=str(ratio),residual=[str(v) for v in residual])
