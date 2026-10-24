"""Precision-controlled evaluation of the unchanged analytic local operator.

The input moments retain their own precision. Raising contraction precision
alone makes no claim to regenerate more accurate geometric moments.
"""
from contextlib import contextmanager
from decimal import Decimal as D, localcontext
from math import factorial
import time
import numpy as np
import analytic_cut_operator_2d as original
import decimal_retained_system as ds
_DIGITS=100

def contract(data,digits):
    e,basis,center,V,M,degree=data
    with localcontext() as ctx:
        ctx.prec=digits
        C,monos=original.physical_polynomials(e,basis,degree);pos={ij:i for i,ij in enumerate(monos)}
        gx=np.full(C.shape,D(0),dtype=object);gy=gx.copy()
        for k,(i,j) in enumerate(monos):
            if i:gx[:,pos[i-1,j]]+=i*C[:,k]
            if j:gy[:,pos[i,j-1]]+=j*C[:,k]
        cx,cy,R=map(ds.exact_decimal,center);normal=-(cx*gx+cy*gy)/R
        for k,(i,j) in enumerate(monos):
            if i+j<degree:normal[:,pos[i+1,j]]+=gx[:,k]/R;normal[:,pos[i,j+1]]+=gy[:,k]/R
        HV=np.array([[V[i+k,j+l] for k,l in monos] for i,j in monos],dtype=object)
        HB=np.array([[M[i+k,j+l] for k,l in monos] for i,j in monos],dtype=object)
        C_HV=C@HV;C_HB=C@HB;N_HB=normal@HB
        K=(gx@HV)@gx.T+(gy@HV)@gy.T;Mb=C_HB@C.T;Cn=N_HB@C.T
        u=np.array([D(1)/D(factorial(i)*factorial(j)) for i,j in monos],dtype=object)
        force=np.array([-D(2)/D(factorial(i)*factorial(j)) if i+j<=degree-2 else D(0) for i,j in monos],dtype=object)
        out=dict(K=K,Mb=Mb,Cn=Cn,fvol=C_HV@force,fbc_pen=C_HB@u,fbc_cons=-(N_HB@u))
        scale=max(D(1),max(map(abs,K.ravel())),max(map(abs,Mb.ravel())))
        asymmetries={key:str(max(map(abs,(a-a.T).ravel()))/scale) for key,a in [('K',K),('Mb',Mb)]}
        asym=max(map(D,asymmetries.values()));out['K']=(K+K.T)/2;out['Mb']=(Mb+Mb.T)/2
    return out,dict(digits=digits,relative_asymmetry=str(asym),component_asymmetry=asymmetries)

def element_operator(eid):
    tick=time.perf_counter();out,info=contract(original._DATA[eid],_DIGITS)
    return eid,out,dict(info,element=eid,seconds=time.perf_counter()-tick)

@contextmanager
def contraction_precision(digits):
    global _DIGITS
    assert digits in (80,100,120,140)
    old=original.element_operator;old_digits=_DIGITS;_DIGITS=digits;original.element_operator=element_operator
    try:yield
    finally:original.element_operator=old;_DIGITS=old_digits
