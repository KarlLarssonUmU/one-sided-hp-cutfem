"""Decimal finite-element contractions from analytic cut-cell moments."""
from decimal import Decimal as D,localcontext
from math import factorial
import multiprocessing as mp
import time
import numpy as np
import precise_projection_2d as projection
import decimal_retained_system as ds
_DATA=None

def physical_polynomials(element,basis,degree):
    a,b,c,d=map(ds.exact_decimal,element.J.ravel());det=a*d-b*c;assert det>0
    x,y=map(ds.exact_decimal,element.xy[0]);r={(0,0):(-d*x+b*y)/det,(1,0):d/det,(0,1):-b/det};s={(0,0):(c*x-a*y)/det,(1,0):-c/det,(0,1):a/det}
    rp=[{(0,0):D(1)}];sp=[{(0,0):D(1)}]
    for _ in range(degree):rp.append(projection.multiply(rp[-1],r));sp.append(projection.multiply(sp[-1],s))
    monos=[(i,total-i) for total in range(degree+1) for i in range(total+1)];pos={ij:i for i,ij in enumerate(monos)}
    composed={ij:projection.multiply(rp[ij[0]],sp[ij[1]]) for ij in monos};C=np.full((len(basis),len(monos)),D(0),dtype=object)
    for k,poly in enumerate(basis):
        for ij,v in poly.items():
            coefficient=projection.dec(v)
            for kl,value in composed[ij].items():C[k,pos[kl]]+=coefficient*value
    return C,monos

def element_operator(eid):
    tick=time.perf_counter();e,basis,center,V,M,degree=_DATA[eid]
    with localcontext() as ctx:
        ctx.prec=80;C,monos=physical_polynomials(e,basis,degree);pos={ij:i for i,ij in enumerate(monos)};gx=np.full(C.shape,D(0),dtype=object);gy=gx.copy()
        for k,(i,j) in enumerate(monos):
            if i:gx[:,pos[i-1,j]]+=i*C[:,k]
            if j:gy[:,pos[i,j-1]]+=j*C[:,k]
        cx,cy,R=map(ds.exact_decimal,center);normal=-(cx*gx+cy*gy)/R
        for k,(i,j) in enumerate(monos):
            if i+j<degree:normal[:,pos[i+1,j]]+=gx[:,k]/R;normal[:,pos[i,j+1]]+=gy[:,k]/R
        HV=np.array([[V[i+k,j+l] for k,l in monos] for i,j in monos],dtype=object);HB=np.array([[M[i+k,j+l] for k,l in monos] for i,j in monos],dtype=object)
        C_HV=C@HV;C_HB=C@HB;N_HB=normal@HB
        K=(gx@HV)@gx.T+(gy@HV)@gy.T;Mb=C_HB@C.T;Cn=N_HB@C.T
        u=np.array([D(1)/D(factorial(i)*factorial(j)) for i,j in monos],dtype=object)
        force=np.array([-D(2)/D(factorial(i)*factorial(j)) if i+j<=degree-2 else D(0) for i,j in monos],dtype=object)
        out=dict(K=K,Mb=Mb,Cn=Cn,fvol=C_HV@force,fbc_pen=C_HB@u,fbc_cons=-(N_HB@u))
        scale=max(D(1),max(map(abs,K.ravel())),max(map(abs,Mb.ravel())))
        asym=max(max(map(abs,(K-K.T).ravel())),max(map(abs,(Mb-Mb.T).ravel())))/scale
        out['K']=(K+K.T)/2;out['Mb']=(Mb+Mb.T)/2
    return eid,out,dict(element=eid,relative_asymmetry=str(asym),seconds=time.perf_counter()-tick)

def assemble(mesh,space,center,radius,moments,workers):
    global _DATA
    _DATA={}
    for row in moments:
        eid=row['element'];V={(i,j):D(v) for i,j,v in row['volume']};M={(i,j):D(v) for i,j,v in row['boundary']}
        _DATA[eid]=(mesh.elements[eid],space['basis'][eid],[*center,radius],V,M,space['p'])
    assert set(_DATA)==set(space['active']);n=1+max(int(np.max(v)) for v in space['dofs'].values());out={k:np.full((n,n) if k in ('K','Mb','Cn') else n,D(0),dtype=object) for k in ('K','Mb','Cn','fvol','fbc_pen','fbc_cons')};checks=[]
    with localcontext() as ctx:
        ctx.prec=80
        with mp.get_context('fork').Pool(workers) as pool:
            for eid,local,info in pool.imap(element_operator,space['active']):
                ix=space['dofs'][eid]
                for key,value in local.items():out[key][np.ix_(ix,ix) if value.ndim==2 else ix]+=value
                checks.append(info);print('Analytic cut operator element',eid,'seconds',info['seconds'],flush=True)
    return out,checks

def combine(blocks,stabilization,h,p,beta0=16.,gamma1=.1):
    with localcontext() as ctx:
        ctx.prec=80;alpha=ds.exact_decimal(np.float64(beta0))*D(p)**2/ds.exact_decimal(h);gamma=ds.exact_decimal(np.float64(gamma1));n=len(blocks['K']);rows=[]
        assert len(stabilization)==n
        for i in range(n):
            S={j:D(v) for j,v in stabilization[i]};row=[]
            for j in range(n):
                v=blocks['K'][i,j]-blocks['Cn'][i,j]-blocks['Cn'][j,i]+alpha*blocks['Mb'][i,j]+gamma*S.get(j,D(0))
                if v:row.append((j,v))
            rows.append(row)
        rhs=[v+c+alpha*b for v,c,b in zip(blocks['fvol'],blocks['fbc_cons'],blocks['fbc_pen'])]
    return rows,rhs

def self_check():
    from types import SimpleNamespace
    import circle_moments_2d as cm
    global _DATA
    with localcontext() as ctx:
        ctx.prec=80
        element=SimpleNamespace(J=np.array([[2.,1.],[0.,3.]]),xy=np.array([[1.,2.],[3.,2.],[2.,5.]]))
        basis=[{(0,0):1,(1,0):-1,(0,1):-1},{(1,0):1},{(0,1):1}]
        V,M,_=cm.moments(element.xy,[2.,3.],10.,2);_DATA={0:(element,basis,[2.,3.,10.],V,M,1)}
        _,out,info=element_operator(0);expected=np.array([[D(5)/6,-D(2)/3,-D(1)/6],[-D(2)/3,D(5)/6,-D(1)/6],[-D(1)/6,-D(1)/6,D(1)/3]],dtype=object)
        error=max(map(abs,(out['K']-expected).ravel()));assert error<D('1e-65')
        assert all(v==0 for key in ('Mb','Cn','fvol','fbc_pen','fbc_cons') for v in out[key].ravel())
    return dict(affine_linear_stiffness_error=str(error),no_circle_boundary_inside_full_triangle=True)
