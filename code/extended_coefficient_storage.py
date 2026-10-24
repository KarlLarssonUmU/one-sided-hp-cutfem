"""Store a retained coefficient as two binary64 components.

This tests storage, not ordinary floating-point field evaluation. The two
components must remain separate; adding them in binary64 loses the low part.
No universal106-bit accuracy claim is made near underflow or overflow.
"""
from decimal import Decimal as D,localcontext
import numpy as np
import decimal_retained_system as ds

def split(values):
    high=np.array([float(v) for v in values],dtype=np.float64)
    assert np.all(np.isfinite(high))
    #1100 digits preserve exact binary64 components and provide ample guard
    #digits for the normal-range retained coefficients tested here. This is
    #not a universal exact-summation guarantee for arbitrarily separated exponents.
    with localcontext() as ctx:
        ctx.prec=1100
        low=np.array([float(v-ds.exact_decimal(h)) for v,h in zip(values,high,strict=True)],dtype=np.float64)
        assert np.all(np.isfinite(low))
        reconstructed=[ds.exact_decimal(h)+ds.exact_decimal(l) for h,l in zip(high,low,strict=True)]
    return high,low,reconstructed

def self_check():
    with localcontext() as ctx:
        ctx.prec=1100
        exact=D(1)+D(2)**-54
        values=[exact,D('1.123456789012345678901234567890123456789'),D('-0.000000000000000000012345678901234567890123456789')]
        high,low,restored=split(values)
        assert high[0]==1 and low[0]==2.**-54 and restored[0]==exact
        assert np.float64(high[0]+low[0])==1
        relative=max(abs(a-b)/abs(a) for a,b in zip(values,restored))
        assert relative<D('1e-30')
        return dict(normal_range_relative_error=str(relative),small_increment_exact=True,binary64_sum_loses_low_component=True,components=2,component_mantissa_bits=53)
