"""Naive and Kahan global assembly from exactly the same local volume blocks."""
from decimal import Decimal as D,localcontext
import numpy as np
import decimal_retained_system as ds
import repaired_precision_2d as control
import precision_eval_2d as pe
import polynomial_current_2d as pc
import local_integrals_2d as li

def add(total,correction,index,value):
    previous=total[index];increment=value-correction[index];updated=previous+increment
    correction[index]=(updated-previous)-increment;total[index]=updated

def assemble(problem,data=pc.data):
    n=1+max(int(np.max(x)) for x in problem['element_dofs'].values())
    naive={k:np.zeros((n,n) if k in ('K','M') else n,dtype=np.longdouble) for k in ('K','M','fvol')}
    corrected={k:v.copy() for k,v in naive.items()};correction={k:v.copy() for k,v in naive.items()}
    digest=li.groups_digest(problem);checks=dict(coordinates=0.,inverse_transpose=0.);seen=set()
    for eid,points,weights in problem['volume_groups']:
        e=problem['mesh'].elements[eid];idx=problem['element_dofs'][eid]
        if eid not in seen:
            sampled=points[np.unique(np.linspace(0,len(points)-1,min(3,len(points)),dtype=int))]
            for name,value in control.point_check(e,sampled).items():checks[name]=max(checks[name],value)
            seen.add(eid)
        r,s,inv=control.geometry(e,points);v,gref=pe.horner(problem['element_basis'][eid],r,s,np.longdouble)
        g=np.einsum('ab,ibq->iaq',inv,gref);_,_,force=data(points)
        local=li.volume_kernel(v,g,weights,force)
        for name,value in local.items():
            ix=np.ix_(idx,idx) if value.ndim==2 else idx
            naive[name][ix]+=value;add(corrected[name],correction[name],ix,value)
    assert digest==li.groups_digest(problem)
    return naive,corrected,dict(quadrature_sha256=digest,Decimal80_geometry_scaled_errors=checks,checked_elements=len(seen),component_maximum_changes={k:float(np.max(np.abs(corrected[k]-naive[k]))) for k in naive})

def self_check():
    # Exact representable increments are lost by naive accumulation at unity.
    total=np.zeros((2,2),dtype=np.longdouble);correction=total.copy();naive=total.copy();ix=np.ix_([0,1],[0,1])
    small=np.longdouble(2)**-65
    values=[np.ones((2,2),dtype=np.longdouble)]+[np.full((2,2),small,dtype=np.longdouble) for _ in range(8)]+[-np.ones((2,2),dtype=np.longdouble)]
    for value in values:naive[ix]+=value;add(total,correction,ix,value)
    assert np.all(total==8*small) and np.all(naive==0)
    with localcontext() as ctx:
        ctx.prec=80;exact=sum((ds.exact_decimal(v[0,0]) for v in values),D(0));assert ds.exact_decimal(total[0,0])==exact
    return dict(exact_small_increments_retained=True,naive_loses_increments=True)
