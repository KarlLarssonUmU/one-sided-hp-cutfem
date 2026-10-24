"""Preserve the r*s*(1-r-s) factor instead of storing rounded expanded bubbles."""
from contextlib import contextmanager
from decimal import Decimal as D,localcontext
from fractions import Fraction as F
from math import factorial
import numpy as np
import precision_eval_2d as pe
import polynomial_probe_2d as poly
import decimal_retained_system as ds
_FACTORS={}

def fraction(x):
    if isinstance(x,F):return x
    n,d=x.as_integer_ratio();return F(n,d)

def repair_space(space):
    repaired={};cache={};p=space['p'];head_count=3+3*(p-1)
    for eid,polys in space['basis'].items():
        key=tuple(tuple(sorted(a.items())) for a in polys)
        if key not in cache:
            head=[dict(x) for x in polys[:head_count]];quotients=[];expanded=[]
            for bubble in polys[head_count:]:
                assert all(i>=1 and j>=1 for i,j in bubble)
                degree=max(i+j for i,j in bubble);q={}
                # Triangular exact division, preserving coefficients below
                # the highest homogeneous degree of the expanded bubble.
                for total in range(degree-2):
                    for i in range(total+1):
                        j=total-i
                        q[i,j]=fraction(bubble.get((i+1,j+1),0.))+q.get((i-1,j),F(0))+q.get((i,j-1),F(0))
                with localcontext() as ctx:
                    ctx.prec=80
                    qld={ij:np.longdouble(str(D(v.numerator)/D(v.denominator))) for ij,v in q.items() if v}
                exact={ij:fraction(v) for ij,v in qld.items()};out={}
                for (i,j),v in exact.items():
                    for ij,multiplier in [((i+1,j+1),1),((i+2,j+1),-1),((i+1,j+2),-1)]:out[ij]=out.get(ij,F(0))+multiplier*v
                quotients.append(qld);expanded.append({ij:v for ij,v in out.items() if v})
            basis=head+expanded;cache[key]=basis
            _FACTORS[id(basis)]=(basis,head,quotients)
        repaired[eid]=cache[key]
    patterns={}
    for eid,basis in repaired.items():
        old=space['basis'][eid]
        for key,pattern in space['patterns'].items():
            if old==pattern:patterns[key]=basis;break
    assert len(patterns)==len(space['patterns'])
    return dict(space,basis=repaired,patterns=patterns)

@contextmanager
def evaluation():
    original=pe.horner
    def factored(polys,r,s,dtype=np.float64):
        if id(polys) not in _FACTORS:return original(polys,r,s,dtype)
        _,head,q=_FACTORS[id(polys)];r=np.asarray(r,dtype=dtype);s=np.asarray(s,dtype=dtype)
        h,hg=original(head,r,s,dtype);v,g=original(q,r,s,dtype)
        bubble=r*s*(1-r-s);dr=s*(1-2*r-s);ds_=r*(1-r-2*s)
        values=v*bubble;grad=g*bubble[None,None,:]+v[:,None,:]*np.stack((dr,ds_))[None,:,:]
        return np.concatenate((h,values)),np.concatenate((hg,grad))
    try:
        pe.horner=factored
        yield
    finally:pe.horner=original

def representation(mesh,space,digits=80):
    p=space['p'];monomials=[(i,total-i) for total in range(p+1) for i in range(total+1)]
    n=1+max(int(np.max(x)) for x in space['dofs'].values());values=[[] for _ in range(n)];cache={};maximum=D(0)
    with localcontext() as ctx:
        ctx.prec=digits
        cv=lambda x:D(fraction(x).numerator)/D(fraction(x).denominator)
        for eid,polys in space['basis'].items():
            key=id(polys);matrix=[[cv(poly_.get(ij,0.)) for poly_ in polys] for ij in monomials]
            if key not in cache:cache[key]=poly.decimal_lu(matrix)
            e=mesh.elements[eid];a=sum((ds.exact_decimal(v) for v in e.xy[0]),D(0));b=sum((ds.exact_decimal(v) for v in e.J[:,0]),D(0));c=sum((ds.exact_decimal(v) for v in e.J[:,1]),D(0))
            power=lambda x,k:x**k if k else D(1)
            rhs=[power(b,i)*power(c,j)/D(factorial(i)*factorial(j))*sum((power(a,k)/D(factorial(k)) for k in range(p+1-i-j)),D(0)) for i,j in monomials]
            x=poly.decimal_solve(cache[key],rhs)
            maximum=max(maximum,max(abs(sum((row[j]*x[j] for j in range(len(x))),D(0))-target) for row,target in zip(matrix,rhs)))
            for idx,v in zip(space['dofs'][eid],x):values[int(idx)].append(v)
        discrepancy=max(max(v)-min(v) for v in values)
        result=np.asarray([str(sum(v,D(0))/len(v)) for v in values],dtype=np.longdouble)
    return result,dict(decimal_digits=digits,maximum_identity_residual=str(maximum),maximum_shared_coefficient_discrepancy=float(discrepancy),unique_basis_patterns=len(cache))

def self_check(space):
    # Use exact Fraction restriction, without the old helper's binary64 cast.
    from math import comb
    nonzero=0;head=3+3*(space['p']-1)
    checked=set();max_error=0.;edge_value=0.
    r=np.asarray([.13,.71,.499,.001,.998],dtype=np.longdouble);s=np.asarray([.27,.2,.499,.998,.001],dtype=np.longdouble)
    with evaluation():
        for polys in space['basis'].values():
            if id(polys) in checked:continue
            checked.add(id(polys))
            for bubble in polys[head:]:
                trace=[F(0)]*(space['p']+1)
                for (i,j),v in bubble.items():
                    assert i>=1 and j>=1
                    for k in range(j+1):trace[i+k]+=v*comb(j,k)*(-1)**k
                nonzero+=int(any(trace))
            observed_v,observed_g=pe.horner(polys,r,s,np.longdouble)
            with localcontext() as ctx:
                ctx.prec=80;reference=np.zeros((3,len(polys),len(r)),dtype=np.longdouble)
                for k,poly_ in enumerate(polys):
                    for q,(rr,ss) in enumerate(zip(r,s)):
                        rr,ss=ds.exact_decimal(rr),ds.exact_decimal(ss);v=dr=dss=D(0)
                        power=lambda x,n:x**n if n else D(1)
                        for (i,j),value in poly_.items():
                            f=fraction(value);c=D(f.numerator)/D(f.denominator)
                            v+=c*power(rr,i)*power(ss,j)
                            if i:dr+=i*c*power(rr,i-1)*power(ss,j)
                            if j:dss+=j*c*power(rr,i)*power(ss,j-1)
                        reference[:,k,q]=[np.longdouble(str(x)) for x in (v,dr,dss)]
            observed=np.stack((observed_v,observed_g[:,0],observed_g[:,1]))
            error=float(np.max(np.abs(observed-reference)/np.maximum(1,np.max(np.abs(reference),axis=2)[:,:,None])))
            max_error=max(max_error,error)
            t=np.array([0.,.125,.25,.5,.75,.875,1.],dtype=np.longdouble)
            for rr,ss in ((t,0*t),(0*t,t),(t,1-t)):
                values,_=pe.horner(polys,rr,ss,np.longdouble);edge_value=max(edge_value,float(np.max(np.abs(values[head:]))))
    assert nonzero==0 and edge_value==0 and max_error<=2e-14
    return dict(exact_nonzero_bubble_edge_traces=nonzero,sampled_bubble_edge_max=edge_value,Decimal80_point_scaled_error=max_error,point_tolerance=2e-14,basis_patterns=len(checked))
