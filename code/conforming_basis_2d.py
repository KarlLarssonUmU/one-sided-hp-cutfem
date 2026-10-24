"""Exact edge polynomials and explicit edge/bubble factors for the Pp basis."""
from contextlib import contextmanager
from fractions import Fraction as F
from math import factorial
import numpy as np
import precision_eval_2d as pe
import conforming_bubbles_2d as bubbles
_FULL={}
LAMBDAS=({(0,0):F(1),(1,0):F(-1),(0,1):F(-1)}, {(1,0):F(1)}, {(0,1):F(1)})
def add(a,b,scale=F(1)):
    out=dict(a)
    for ij,x in b.items():out[ij]=out.get(ij,F(0))+scale*x
    return {ij:x for ij,x in out.items() if x}
def mul(a,b):
    out={}
    for (i,j),x in a.items():
        for (k,l),y in b.items():out[i+k,j+l]=out.get((i+k,j+l),F(0))+x*y
    return {ij:x for ij,x in out.items() if x}
def power(a,n):
    out={(0,0):F(1)}
    for _ in range(n):out=mul(out,a)
    return out
def edge_quotient(n,lo,hi):
    # L_n(z)=(z^2-1) P'_{n-1}(z)/(n(n-1)).
    m=n-1;S=add(lo,hi);D=add(hi,lo,F(-1));out={}
    for j in range(m//2+1):
        k=m-2*j
        if k==0:continue
        coefficient=F((-1)**j*factorial(2*m-2*j),2**m*factorial(j)*factorial(m-j)*factorial(k))
        coefficient*=F(-4*k,n*(n-1))
        term=mul(power(D,k-1),power(S,n-1-k));out=add(out,term,coefficient)
    return out

def repair_space(space):
    bubble_space=bubbles.repair_space(space);new={};cache={};p=space['p'];head=3+3*(p-1)
    for eid,old in bubble_space['basis'].items():
        if id(old) not in cache:
            edge_polys=[];quotients=[];factors=[]
            # Infer each original edge's orientation from its odd degree-three
            # trace coefficient. Use the frozen vertex-order pattern directly.
            pattern=next(key for key,value in bubble_space['patterns'].items() if value is old)
            for edge,(a,b) in enumerate(((0,1),(1,2),(2,0))):
                lo,hi=(a,b) if pattern[edge] else (b,a)
                for degree in range(2,p+1):
                    q=edge_quotient(degree,LAMBDAS[lo],LAMBDAS[hi]);quotients.append(q);factors.append((lo,hi))
                    edge_polys.append(mul(mul(LAMBDAS[lo],LAMBDAS[hi]),q))
            basis=[dict(x) for x in LAMBDAS]+edge_polys+old[head:]
            cache[id(old)]=basis
            bq=bubbles._FACTORS[id(old)][2]
            _FULL[id(basis)]=(basis,[dict(x) for x in LAMBDAS],quotients,factors,bq)
        new[eid]=cache[id(old)]
    patterns={key:cache[id(value)] for key,value in bubble_space['patterns'].items()}
    return dict(space,basis=new,patterns=patterns)

@contextmanager
def evaluation():
    original=pe.horner
    def factored(polys,r,s,dtype=np.float64):
        if id(polys) not in _FULL:return original(polys,r,s,dtype)
        _,vertices,eq,factors,bq=_FULL[id(polys)];r=np.asarray(r,dtype=dtype);s=np.asarray(s,dtype=dtype)
        vv,vg=original(vertices,r,s,dtype);q,qg=original(eq,r,s,dtype)
        lo=np.asarray([a for a,b in factors]);hi=np.asarray([b for a,b in factors])
        ef=vv[lo]*vv[hi];efg=vg[lo]*vv[hi,None,:]+vg[hi]*vv[lo,None,:]
        ev=q*ef;eg=qg*ef[:,None,:]+q[:,None,:]*efg
        q,qg=original(bq,r,s,dtype);bf=r*s*(1-r-s);bfg=np.stack((s*(1-2*r-s),r*(1-r-2*s)))
        bv=q*bf;bg=qg*bf[None,None,:]+q[:,None,:]*bfg[None,:,:]
        return np.concatenate((vv,ev,bv)),np.concatenate((vg,eg,bg))
    try:
        pe.horner=factored
        yield
    finally:pe.horner=original

representation=bubbles.representation

def restrict(poly,a,b):
    degree=max(i+j for i,j in poly);out=[F(0)]*(degree+1)
    def univariate_power(x,y,n):
        values=[F(1)]
        for _ in range(n):
            nxt=[F(0)]*(len(values)+1)
            for k,v in enumerate(values):nxt[k]+=v*x;nxt[k+1]+=v*y
            values=nxt
        return values
    for (i,j),c in poly.items():
        rr=univariate_power(a[0],b[0]-a[0],i);ss=univariate_power(a[1],b[1]-a[1],j)
        for k,x in enumerate(rr):
            for l,y in enumerate(ss):out[k+l]+=bubbles.fraction(c)*x*y
    while out and out[-1]==0:out.pop()
    return tuple(out)

def self_check(space,mesh=None):
    # Nested context lets the existing independent Decimal coefficient
    # reference check evaluate this factor-preserving implementation.
    with evaluation():result=bubbles.self_check(space)
    refs=((F(0),F(0)),(F(1),F(0)),(F(0),F(1)));p=space['p']
    base=next(iter(space['basis'].values()));traces=[restrict(poly,refs[0],refs[1]) for poly in base[3:3+p-1]]
    for i,a in enumerate(traces):
        assert not a or a[0]==0
        assert sum(a)==0
        da=[k*x for k,x in enumerate(a)][1:]
        for j,b in enumerate(traces):
            db=[k*x for k,x in enumerate(b)][1:]
            product=sum((x*y/F(k+l+1) for k,x in enumerate(da) for l,y in enumerate(db)),F(0))
            assert product==(F(4,2*(i+2)-1) if i==j else F(0))
    mismatch=0;shared=0
    if mesh is not None:
        edges={}
        for eid,polys in space['basis'].items():
            e=mesh.elements[eid]
            for a,b in ((0,1),(1,2),(2,0)):
                lo,hi=(a,b) if e.vids[a]<e.vids[b] else (b,a)
                values={int(idx):restrict(poly,refs[lo],refs[hi]) for idx,poly in zip(space['dofs'][eid],polys)}
                values={idx:v for idx,v in values.items() if v}
                key=tuple(sorted((e.vids[a],e.vids[b])))
                if key in edges:shared+=1;mismatch+=int(values!=edges[key])
                else:edges[key]=values
        assert mismatch==0
    result.update(exact_edge_derivative_Gram=True,shared_edges_checked=shared,nonconforming_shared_edges=mismatch)
    return result
