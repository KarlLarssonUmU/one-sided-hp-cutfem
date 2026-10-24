"""Evaluate a repaired Pp field minus the exact polynomial before sampling.

The coefficient identity is formed with Decimal arithmetic. This avoids the
subtraction of two order-one sampled functions when their difference is tiny.
It does not change the coefficients, basis, mesh, operator or linear solve.
"""
from decimal import Decimal as D,localcontext
from fractions import Fraction
from math import factorial
import numpy as np
import decimal_retained_system as ds

def exact_value(x):
    return x if isinstance(x,D) else ds.exact_decimal(x)

def decimal_value(x):
    if isinstance(x,Fraction):return D(x.numerator)/D(x.denominator)
    return exact_value(x)

def power(x,k):return x**k if k else D(1)

def error_polynomial(element,basis,coefficients,digits=80):
    """Reference-element coefficients of sum c_i phi_i - Pp(x+y)."""
    with localcontext() as context:
        context.prec=digits
        degree=max(i+j for poly in basis for i,j in poly)
        exact={}
        for coefficient,polynomial in zip(coefficients,basis,strict=True):
            c=exact_value(coefficient)
            for ij,value in polynomial.items():exact[ij]=exact.get(ij,D(0))+c*decimal_value(value)
        a=sum((exact_value(v) for v in element.xy[0]),D(0))
        b=sum((exact_value(v) for v in element.J[:,0]),D(0))
        c=sum((exact_value(v) for v in element.J[:,1]),D(0))
        error={}
        for total in range(degree+1):
            for i in range(total+1):
                j=total-i
                target=power(b,i)*power(c,j)/D(factorial(i)*factorial(j))*sum((power(a,k)/D(factorial(k)) for k in range(degree+1-total)),D(0))
                error[i,j]=exact.get((i,j),D(0))-target
        assert all(i+j<=degree for i,j in exact)
    return error

def evaluate_decimal(polynomial,r,s):
    value=dr=ds_=D(0)
    for (i,j),c in polynomial.items():
        value+=c*power(r,i)*power(s,j)
        if i:dr+=i*c*power(r,i-1)*power(s,j)
        if j:ds_+=j*c*power(r,i)*power(s,j-1)
    return value,dr,ds_

def decimal_geometry(element,point):
    """Independent pivoted elimination, followed by inverse-column solves."""
    J=[[exact_value(x) for x in row] for row in element.J]
    def solve(rhs):
        m=[row[:] for row in J];v=list(rhs)
        if abs(m[1][0])>abs(m[0][0]):m.reverse();v.reverse()
        factor=m[1][0]/m[0][0]
        y=(v[1]-factor*v[0])/(m[1][1]-factor*m[0][1]);x=(v[0]-m[0][1]*y)/m[0][0]
        return x,y
    r,s=solve([exact_value(point[k])-exact_value(element.xy[0,k]) for k in range(2)])
    invT=[solve([D(int(i==j)) for i in range(2)]) for j in range(2)]
    return r,s,invT

def exact_point_error(element,basis,coefficients,point,digits=80):
    """Direct field and exact Pp subtraction, independent of Taylor expansion."""
    with localcontext() as context:
        context.prec=digits;r,s,inv=decimal_geometry(element,point);v=dr=ds_=D(0)
        for coefficient,poly in zip(coefficients,basis,strict=True):
            values=evaluate_decimal({ij:decimal_value(x) for ij,x in poly.items()},r,s)
            c=exact_value(coefficient);v+=c*values[0];dr+=c*values[1];ds_+=c*values[2]
        degree=max(i+j for poly in basis for i,j in poly)
        t=sum((exact_value(x) for x in point),D(0))
        P=lambda n:sum((power(t,k)/D(factorial(k)) for k in range(n+1)),D(0))
        return [v-P(degree),*[row[0]*dr+row[1]*ds_-P(degree-1) for row in inv]]

def identity_point_error(element,polynomial,point,digits=80):
    with localcontext() as context:
        context.prec=digits;r,s,inv=decimal_geometry(element,point);v,dr,ds_=evaluate_decimal(polynomial,r,s)
        return [v,*[row[0]*dr+row[1]*ds_ for row in inv]]

def rounded_polynomial(polynomial):
    return {ij:np.longdouble(str(c)) for ij,c in polynomial.items()}

def sample_error(element,polynomial,points,horner,geometry):
    r,s,inv=geometry(element,points)
    v,g=horner([polynomial],r,s,np.longdouble)
    physical=np.einsum('ab,ibq->iaq',inv,g)
    return np.column_stack((v[0],physical[0].T))

def triangle_squared_norms(element,polynomial,digits=80):
    """Analytic full-element L2/gradient norms, bounding a clipped subdomain.

    Integral r^a s^b dr ds = a! b! / (a+b+2)! on the unit triangle.
    Decimal arithmetic is used for coefficients, moments and cancellation.
    """
    with localcontext() as context:
        context.prec=digits
        J=[[exact_value(x) for x in row] for row in element.J]
        det=J[0][0]*J[1][1]-J[0][1]*J[1][0];assert det>0
        inv=[[J[1][1]/det,-J[1][0]/det],[-J[0][1]/det,J[0][0]/det]]
        deriv=[{(i-1,j):i*c for (i,j),c in polynomial.items() if i},{(i,j-1):j*c for (i,j),c in polynomial.items() if j}]
        physical=[]
        for row in inv:
            p={}
            for multiplier,d in zip(row,deriv):
                for ij,c in d.items():p[ij]=p.get(ij,D(0))+multiplier*c
            physical.append(p)
        moments={}
        def squared(p):
            value=D(0)
            for (i,j),a in p.items():
                for (k,l),b in p.items():
                    ij=(i+k,j+l)
                    if ij not in moments:moments[ij]=D(factorial(ij[0])*factorial(ij[1]))/D(factorial(sum(ij)+2))
                    value+=a*b*moments[ij]
            assert value>=0
            return det*value
        return squared(polynomial),sum((squared(p) for p in physical),D(0))
