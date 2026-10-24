"""Decimal residual refinement of exact stored binary floating-point systems.

Only the residual accumulation and coefficient updates change. Corrections use
NumPy's existing float64 solve. Sparse rows omit exact zeros, not small entries.
No operator reassembly, regularization, or alteration of the stored A,b occurs.
"""
from decimal import Decimal as D, localcontext
import time
import numpy as np

_POW5={}
def exact_decimal(value):
    numerator,denominator=value.as_integer_ratio()
    if not numerator:return D(0)
    exponent=denominator.bit_length()-1
    assert denominator==1<<exponent
    if exponent not in _POW5:_POW5[exponent]=5**exponent
    integer=D(abs(numerator)*_POW5[exponent])
    return D((int(numerator<0),integer.as_tuple().digits,-exponent))

def refine(A,b,initial,steps=6):
    assert A.dtype==b.dtype==initial.dtype==np.longdouble and steps==6
    tick=time.perf_counter();rows=[]
    for row in A:
        indices=np.flatnonzero(row)
        rows.append([(int(j),exact_decimal(row[j])) for j in indices])
    rhs=[exact_decimal(v) for v in b];x=[exact_decimal(v) for v in initial]
    build_seconds=time.perf_counter()-tick;rounded=np.asarray(A,dtype=float)
    def residual():return [rhs[i]-sum((value*x[j] for j,value in row),D(0)) for i,row in enumerate(rows)]
    history=[];five=None
    with localcontext() as ctx:
        ctx.prec=60;scale=max(abs(v) for v in rhs)
        for step in range(steps):
            r=residual();history.append(str(max(abs(v) for v in r)/scale))
            correction=np.linalg.solve(rounded,np.asarray([float(v) for v in r]))
            x=[v+D.from_float(float(delta)) for v,delta in zip(x,correction)]
            if step==4:five=list(x)
        r=residual();history.append(str(max(abs(v) for v in r)/scale))
        change=max(abs(a-b) for a,b in zip(x,five))/max(abs(v) for v in x)
        ctx.prec=80;r80=residual();check80=max(abs(v) for v in r80)/scale
    converted=np.asarray([str(v) for v in x],dtype=np.longdouble)
    converted5=np.asarray([str(v) for v in five],dtype=np.longdouble)
    return converted,dict(decimal_digits=60,verification_digits=80,steps=steps,
        nonzero_entries=sum(map(len,rows)),conversion_seconds=build_seconds,seconds=time.perf_counter()-tick,
        relative_infinity_residual_history=history,relative_infinity_residual_80=str(check80),
        relative_coefficient_change_step6=str(change),rounded_coefficient_change_step6=float(np.max(np.abs(converted-converted5))),
        decimal_coefficients=[str(v) for v in x])
