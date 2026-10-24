"""Exact polynomial element moments and Decimal80 H1 projection complements.

The patch projector is basis invariant: use global monomials (excluding the
constant gradient) instead of sampled Legendre columns. S=K-B G^{-1} B.T is
the same H1 projection-residual bilinear form in exact arithmetic.
"""
from decimal import Decimal as D,localcontext
from fractions import Fraction
from math import factorial
import multiprocessing as mp
import time
import numpy as np
import decimal_retained_system as ds
import polynomial_probe_2d as poly
_DATA=None

def dec(v):
    if isinstance(v,Fraction):return D(v.numerator)/D(v.denominator)
    if isinstance(v,int):return D(v)
    return ds.exact_decimal(v)

def multiply(a,b):
    out={}
    for (i,j),x in a.items():
        for (k,l),y in b.items():out[i+k,j+l]=out.get((i+k,j+l),D(0))+x*y
    return out

def reference_matrices(basis,degree):
    derivatives=[[{(i-1,j):D(i)*dec(v) for (i,j),v in p.items() if i} for p in basis], [{(i,j-1):D(j)*dec(v) for (i,j),v in p.items() if j} for p in basis]]
    moments={(i,j):D(factorial(i)*factorial(j))/D(factorial(i+j+2)) for i in range(2*degree-1) for j in range(2*degree-1-i)}
    def inner(a,b):return sum((x*y*moments[i+k,j+l] for (i,j),x in a.items() for (k,l),y in b.items()),D(0))
    matrices=[]
    for da,db in ((0,0),(0,1),(1,1)):
        matrices.append(np.array([[inner(a,b) for b in derivatives[db]] for a in derivatives[da]],dtype=object))
    return matrices

def element_data(mesh,space):
    degree=space['p'];monos=[(i,t-i) for t in range(degree+1) for i in range(t+1)];cache={};data={};shared={};worst=D(0)
    with localcontext() as ctx:
        ctx.prec=80
        for eid in space['active']:
            basis=space['basis'][eid];e=mesh.elements[eid];key=id(basis)
            if key not in cache:
                matrix=[[dec(p.get(ij,0)) for p in basis] for ij in monos]
                cache[key]=(matrix,poly.decimal_lu(matrix),reference_matrices(basis,degree))
            matrix,lu,(rr,rs,ss)=cache[key]
            a,b,c,d=[ds.exact_decimal(v) for v in e.J.ravel()];det=a*d-b*c;assert det>0
            inv=np.array([[d,-c],[-b,a]],dtype=object)/det;metric=inv.T@inv
            K=det*(metric[0,0]*rr+metric[0,1]*(rs+rs.T)+metric[1,1]*ss)
            X={(0,0):ds.exact_decimal(e.xy[0,0]),(1,0):a,(0,1):b};Y={(0,0):ds.exact_decimal(e.xy[0,1]),(1,0):c,(0,1):d}
            xp=[{(0,0):D(1)}];yp=[{(0,0):D(1)}]
            for _ in range(degree):xp.append(multiply(xp[-1],X));yp.append(multiply(yp[-1],Y))
            columns=[]
            for i,j in monos:
                target=multiply(xp[i],yp[j]);rhs=[target.get(ij,D(0)) for ij in monos];values=poly.decimal_solve(lu,rhs)
                worst=max(worst,max(abs(sum((v*x for v,x in zip(row,values)),D(0))-z) for row,z in zip(matrix,rhs)));columns.append(values)
            P=np.array(columns,dtype=object).T;B=K@P[:,1:];G=P[:,1:].T@B
            for dof,row in zip(space['dofs'][eid],P):shared.setdefault(int(dof),[]).append(row)
            data[eid]=dict(dofs=space['dofs'][eid],K=K,P=P,B=B,G=G)
        mismatch=max(abs(v-z) for values in shared.values() for row in values[1:] for v,z in zip(row,values[0]))
        globalP=np.array([shared[i][0] for i in range(len(shared))],dtype=object)
    return data,globalP,dict(coefficient_identity_maximum=str(worst),shared_coefficient_maximum=str(mismatch),basis_patterns=len(cache),elements=len(data))

def patch_result(item):
    index,patch=item;tick=time.perf_counter()
    with localcontext() as ctx:
        ctx.prec=80;dofs=np.unique(np.concatenate([_DATA[e]['dofs'] for e in patch]));pos={int(v):i for i,v in enumerate(dofs)};n=len(dofs)
        modes=_DATA[patch[0]]['B'].shape[1]
        K=np.full((n,n),D(0),dtype=object);B=np.full((n,modes),D(0),dtype=object);G=np.full((modes,modes),D(0),dtype=object);P=np.full((n,modes+1),D(0),dtype=object)
        for eid in patch:
            z=_DATA[eid];ix=np.array([pos[int(v)] for v in z['dofs']]);K[np.ix_(ix,ix)]+=z['K'];B[ix]+=z['B'];G+=z['G'];P[ix]=z['P']
        lu=poly.decimal_lu(G.tolist());inverse=np.array([poly.decimal_solve(lu,[D(int(i==j)) for i in range(modes)]) for j in range(modes)],dtype=object).T
        inverse_error=max(abs(x-D(int(i==j))) for i,row in enumerate(G@inverse) for j,x in enumerate(row))
        S=K-(B@inverse)@B.T;scale=max(map(abs,K.ravel()));asym=max(map(abs,(S-S.T).ravel()))/scale;S=(S+S.T)/2
        kernel=max(map(abs,(S@P).ravel()))/scale
        numeric=np.asarray(S,dtype=float);eigen=np.linalg.eigvalsh((numeric+numeric.T)/2);es=max(1e-300,float(np.max(np.abs(eigen))))
        metrics=dict(index=index,elements=len(patch),dofs=n,Decimal_inverse_error=str(inverse_error),relative_asymmetry=str(asym),relative_polynomial_kernel=str(kernel),float64_minimum_eigenvalue=float(eigen[0]),float64_relative_negative=max(0.,-float(eigen[0])/es),seconds=time.perf_counter()-tick)
    return dofs,S,metrics

def assemble(mesh,space,patches,workers):
    global _DATA
    _DATA,P,checks=element_data(mesh,space)
    # On one full triangle the FE gradient space equals the projector space.
    dofs,zero,unit=patch_result((-1,[space['active'][0]]))
    scale=max(map(abs,_DATA[space['active'][0]]['K'].ravel()))
    with localcontext() as ctx:
        ctx.prec=80;unit['relative_zero_matrix']=str(max(map(abs,zero.ravel()))/scale)
    assert D(unit['relative_zero_matrix'])<D('1e-40')
    n=len(P);globalS=np.full((n,n),D(0),dtype=object);diagnostics=[]
    with localcontext() as ctx:
        ctx.prec=80
        with mp.get_context('fork').Pool(workers) as pool:
            for dofs,S,metrics in pool.imap(patch_result,enumerate(patches)):
                globalS[np.ix_(dofs,dofs)]+=S;diagnostics.append(metrics);print('Analytic patch',metrics['index'],'dofs',len(dofs),'seconds',metrics['seconds'],flush=True)
        scale=max(map(abs,globalS.ravel()));global_kernel=max(map(abs,(globalS@P).ravel()))/scale
    checks.update(single_triangle=unit,patches=diagnostics,global_relative_polynomial_kernel=str(global_kernel),workers=workers)
    return globalS,checks

def combine(base_rows,rhs,S,gamma1=.1):
    with localcontext() as ctx:
        ctx.prec=80;gamma=ds.exact_decimal(np.float64(gamma1));rows=[]
        for i,base in enumerate(base_rows):
            row=dict(base)
            for j,value in enumerate(S[i]):
                if value:row[j]=row.get(j,D(0))+gamma*value
            rows.append([(j,v) for j,v in sorted(row.items()) if v])
    return rows,rhs
