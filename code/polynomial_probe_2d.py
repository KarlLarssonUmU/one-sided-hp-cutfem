"""Manufactured P8(x+y) data and an independent Decimal coefficient reference."""
from contextlib import contextmanager
from decimal import Decimal as D, localcontext
from math import factorial
import numpy as np
import run_2d_disk_study as fem
import precision_eval_2d as pe


def taylor(t,degree):
    t=np.asarray(t)
    value=np.full_like(t,1/factorial(degree))
    for k in range(degree-1,-1,-1):
        value=value*t+np.asarray(1,dtype=t.dtype)/factorial(k)
    return value


def data(points,*unused,**kwargs):
    t=points[:,0]+points[:,1]
    value=taylor(t,8);derivative=taylor(t,7)
    return value,np.column_stack((derivative,derivative)),-2*taylor(t,6)


@contextmanager
def manufactured_data():
    old=fem.manufactured_solution_values
    try:
        fem.manufactured_solution_values=data
        yield
    finally:
        fem.manufactured_solution_values=old


def decimal_lu(matrix):
    n=len(matrix);a=[row[:] for row in matrix];perm=list(range(n))
    for k in range(n):
        pivot=max(range(k,n),key=lambda i:abs(a[i][k]))
        assert a[pivot][k]!=0
        a[k],a[pivot]=a[pivot],a[k];perm[k],perm[pivot]=perm[pivot],perm[k]
        for i in range(k+1,n):
            a[i][k]/=a[k][k]
            for j in range(k+1,n):a[i][j]-=a[i][k]*a[k][j]
    return a,perm


def decimal_solve(lu,rhs):
    a,perm=lu;n=len(a);x=[rhs[i] for i in perm]
    for i in range(n):x[i]-=sum((a[i][j]*x[j] for j in range(i)),D(0))
    for i in range(n-1,-1,-1):
        x[i]=(x[i]-sum((a[i][j]*x[j] for j in range(i+1,n)),D(0)))/a[i][i]
    return x


def representation(mesh,space,precision=80):
    """Solve stored-basis polynomial coefficient identities, not sample fitting.

    LU factors and affine P8 coefficients are computed at 80 Decimal digits.
    Stored float64 basis/geometry entries are converted exactly. Shared DOF
    discrepancies are recorded before averaging and conversion to longdouble.
    """
    monomials=[(i,total-i) for total in range(9) for i in range(total+1)]
    n=1+max(int(np.max(v)) for v in space['dofs'].values())
    cache={};all_values=[[] for _ in range(n)];max_identity=D(0)
    with localcontext() as context:
        context.prec=precision
        for eid in space['active']:
            polys=space['basis'][eid]
            key=tuple(tuple(sorted(p.items())) for p in polys)
            matrix=[[D.from_float(float(p.get(ij,0.))) for p in polys] for ij in monomials]
            if key not in cache:cache[key]=decimal_lu(matrix)
            e=mesh.elements[eid]
            a=D.from_float(float(e.xy[0,0]))+D.from_float(float(e.xy[0,1]))
            b=D.from_float(float(e.J[0,0]))+D.from_float(float(e.J[1,0]))
            c=D.from_float(float(e.J[0,1]))+D.from_float(float(e.J[1,1]))
            def power(x,k):return x**k if k else D(1)
            rhs=[power(b,i)*power(c,j)/D(factorial(i)*factorial(j))*sum((power(a,k)/D(factorial(k)) for k in range(9-i-j)),D(0)) for i,j in monomials]
            x=decimal_solve(cache[key],rhs)
            max_identity=max(max_identity,max(abs(sum((row[j]*x[j] for j in range(len(x))),D(0))-target) for row,target in zip(matrix,rhs)))
            for dof,value in zip(space['dofs'][eid],x):all_values[int(dof)].append(value)
        discrepancy=max(max(v)-min(v) for v in all_values)
        coefficients=np.array([np.longdouble(str(sum(v,D(0))/len(v))) for v in all_values],dtype=np.longdouble)
    return coefficients,dict(decimal_digits=precision,unique_basis_patterns=len(cache),
        maximum_decimal_coefficient_identity_residual=str(max_identity),
        maximum_shared_dof_coefficient_discrepancy=float(discrepancy),
        note='Stored basis and affine geometry are unchanged. Averaged shared coefficients are converted to longdouble. This is an independent representation of one polynomial, not a higher-precision assembled method.')


def formula_check():
    points=np.array([[.13,.27],[.71,.2],[-.6,.1],[0.,0.],[.4,-.2]],dtype=np.longdouble)
    actual=data(points);reference=[]
    with localcontext() as context:
        context.prec=80
        for point in points:
            t=sum((D.from_float(float(v)) for v in point),D(0))
            def p(n):return sum(((t**k if k else D(1))/D(factorial(k)) for k in range(n+1)),D(0))
            reference.append([np.longdouble(str(v)) for v in (p(8),p(7),-2*p(6))])
    expected=np.array(reference,dtype=np.longdouble)
    observed=np.column_stack((actual[0],actual[1][:,0],actual[2]))
    return float(np.max(np.abs(observed-expected)/np.maximum(1,np.abs(expected))))


def measure_polynomial(problem,coefficients,groups,pairs):
    """Common extended-precision evaluation of the frozen coefficient basis.

    Quadrature nodes/weights and reference coordinates remain float64; all
    polynomial values, coefficient contractions and error sums use longdouble.
    """
    C=np.asarray(coefficients,dtype=np.longdouble)
    sums=np.zeros((len(C)+len(pairs),2),dtype=np.longdouble)
    for eid,points,w in groups:
        e=problem['mesh'].elements[eid];r,s=fem.reference_coordinates(e,points)
        v,gref=pe.horner(problem['element_basis'][eid],r,s,np.longdouble)
        g=np.einsum('ab,ibq->iaq',e.invJT.astype(np.longdouble),gref)
        local=C[:,problem['element_dofs'][eid]]
        values=local@v;grad=np.einsum('ki,iaq->kqa',local,g)
        exact,exact_gradient,_=data(points.astype(np.longdouble))
        weight=w.astype(np.longdouble)
        sums[:len(C),0]+=((values-exact)**2)@weight
        sums[:len(C),1]+=np.sum((grad-exact_gradient)**2,axis=2)@weight
        for k,(a,b) in enumerate(pairs):
            delta=local[a]-local[b];dv=delta@v;dg=np.einsum('i,iaq->qa',delta,g)
            sums[len(C)+k,0]+=np.dot(weight,dv**2)
            sums[len(C)+k,1]+=np.dot(weight,np.sum(dg**2,axis=1))
    return np.asarray(np.sqrt(sums),dtype=float)
