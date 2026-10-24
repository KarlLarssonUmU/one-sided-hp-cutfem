"""Full-triangle polynomial norms with the physical inverse-transpose map."""
from decimal import Decimal as D,localcontext
from math import factorial
import precise_polynomial_identity as identity

def squared_norms(element,polynomial,digits=100):
    with localcontext() as ctx:
        ctx.prec=digits
        a,b,c,d=map(identity.exact_value,element.J.ravel());det=a*d-b*c
        assert det>0
        inverse_transpose=((d/det,-c/det),(-b/det,a/det))
        deriv=[{(i-1,j):i*v for (i,j),v in polynomial.items() if i},{(i,j-1):j*v for (i,j),v in polynomial.items() if j}]
        physical=[]
        for row in inverse_transpose:
            field={}
            for factor,gradient in zip(row,deriv):
                for ij,v in gradient.items():field[ij]=field.get(ij,D(0))+factor*v
            physical.append(field)
        moments={}
        def integrate_square(field):
            value=D(0)
            for (i,j),u in field.items():
                for (k,l),v in field.items():
                    ij=(i+k,j+l)
                    if ij not in moments:moments[ij]=D(factorial(ij[0])*factorial(ij[1]))/D(factorial(sum(ij)+2))
                    value+=u*v*moments[ij]
            assert value>=0
            return det*value
        return integrate_square(polynomial),sum((integrate_square(z) for z in physical),D(0))

def self_check():
    from types import SimpleNamespace
    import numpy as np
    rows=[]
    with localcontext() as ctx:
        ctx.prec=100
        for J in (np.array([[2.,1.],[0.,3.]]),np.array([[3.,0.],[1.,2.]])):
            e=SimpleNamespace(J=J);a,b,c,d=map(D.from_float,J.ravel());area=(a*d-b*c)/2
            for label,poly,expected in [('x',{(1,0):a,(0,1):b},area),('y',{(1,0):c,(0,1):d},area),('x+y',{(1,0):a+c,(0,1):b+d},2*area)]:
                observed=squared_norms(e,poly,100)[1];error=abs(observed-expected);assert error<D('1e-95')
                rows.append(dict(J=J.tolist(),field=label,observed=str(observed),expected=str(expected),error=str(error)))
        e=SimpleNamespace(J=np.array([[2.,1.],[0.,3.]]));l2,h1=squared_norms(e,{(1,1):D(6),(0,2):D(3)},100)
        assert abs(l2-D(24)/5)<D('1e-95') and abs(h1-8)<D('1e-95')
        rows.append(dict(field='xy',L2_squared=str(l2),H1_squared=str(h1),expected_L2_squared='4.8',expected_H1_squared='8'))
    return rows
