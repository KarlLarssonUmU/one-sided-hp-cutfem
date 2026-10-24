"""Ordinary exponential loads from analytic moments and bounded Taylor series."""
from decimal import Decimal as D,localcontext
from math import factorial
import multiprocessing as mp,time
import numpy as np
import circle_moments_2d as moments
import analytic_cut_operator_2d as operator
import decimal_retained_system as ds
_DATA=None

def element_load(eid):
    tick=time.perf_counter();e,basis,center,radius,old,degree,orders=_DATA[eid]
    with localcontext() as ctx:
        ctx.prec=80
        x0=list(map(ds.exact_decimal,e.xy[0]));J=[[ds.exact_decimal(x) for x in row] for row in e.J];vertices=[x0,[x0[i]+J[i][0] for i in (0,1)],[x0[i]+J[i][1] for i in (0,1)]]
        V,M,info=moments.moments(vertices,center,radius,degree+max(orders));C,monos=operator.physical_polynomials(e,basis,degree);pos={ij:k for k,ij in enumerate(monos)}
        gx=np.full(C.shape,D(0),dtype=object);gy=gx.copy()
        for k,(i,j) in enumerate(monos):
            if i:gx[:,pos[i-1,j]]+=i*C[:,k]
            if j:gy[:,pos[i,j-1]]+=j*C[:,k]
        cx,cy=map(ds.exact_decimal,center);R=ds.exact_decimal(radius);normal=-(cx*gx+cy*gy)/R
        for k,(i,j) in enumerate(monos):
            if i+j<degree:normal[:,pos[i+1,j]]+=gx[:,k]/R;normal[:,pos[i,j+1]]+=gy[:,k]/R
        repeat=max(abs(table[i,j]-D(value)) for label,table in [('volume',V),('boundary',M)] for i,j,value in old[label])
        needed=sorted(set([degree-2,degree,*orders]));integrated={}
        for order in needed:
            coefficients=[(i,t-i,D(1)/D(factorial(i)*factorial(t-i))) for t in range(order+1) for i in range(t+1)]
            integrated[order]=[np.array([sum((c*table[i+k,j+l] for k,l,c in coefficients),D(0)) for i,j in monos],dtype=object) for table in (V,M)]
        bound_z=abs(cx+cy)+D(2).sqrt()*R;X=abs(cx)+R;Y=abs(cy)+R;powers=np.array([moments.power(X,i)*moments.power(Y,j) for i,j in monos],dtype=object)
        basis_bound=np.abs(C)@powers;normal_bound=np.abs(normal)@powers;out={}
        for order in orders:
            v,b=integrated[order];tail=bound_z.exp()*bound_z**(order+1)/D(factorial(order+1))
            out[order]=dict(fvol=-2*(C@v),fbc_pen=C@b,fbc_cons=-(normal@b),volume_tail_bound=2*V[0,0]*basis_bound*tail,penalty_tail_bound=M[0,0]*basis_bound*tail,normal_tail_bound=M[0,0]*normal_bound*tail,pointwise_tail_bound=str(tail))
        polynomial=dict(fvol=-2*(C@integrated[degree-2][0]),fbc_pen=C@integrated[degree][1],fbc_cons=-(normal@integrated[degree][1]))
    return eid,out,polynomial,V,M,dict(info,element=eid,lower_moment_repeat=str(repeat),seconds=time.perf_counter()-tick)

def assemble(mesh,space,center,radius,old,orders,workers,beta0=16.):
    global _DATA
    _DATA={z['element']:(mesh.elements[z['element']],space['basis'][z['element']],center,radius,z,space['p'],orders) for z in old};n=1+max(int(np.max(v)) for v in space['dofs'].values())
    rhs={order:np.full(n,D(0),dtype=object) for order in orders};bounds={order:np.full(n,D(0),dtype=object) for order in orders};poly=np.full(n,D(0),dtype=object);saved=[];checks=[]
    with localcontext() as ctx:
        ctx.prec=80;alpha=ds.exact_decimal(np.float64(beta0))*D(space['p'])**2/ds.exact_decimal(mesh.h)
        with mp.get_context('fork').Pool(workers) as pool:
            for eid,local,P,V,M,info in pool.imap(element_load,space['active']):
                ix=space['dofs'][eid]
                for order,a in local.items():
                    rhs[order][ix]+=a['fvol']+a['fbc_cons']+alpha*a['fbc_pen']
                    bounds[order][ix]+=a['volume_tail_bound']+a['normal_tail_bound']+alpha*a['penalty_tail_bound']
                poly[ix]+=P['fvol']+P['fbc_cons']+alpha*P['fbc_pen'];checks.append(info)
                saved.append(dict(element=eid,volume=[[i,j,str(v)] for (i,j),v in V.items()],boundary=[[i,j,str(v)] for (i,j),v in M.items()],geometry=info))
                print('Analytic exponential moments',eid,'seconds',info['seconds'],flush=True)
    return rhs,bounds,poly,saved,checks
