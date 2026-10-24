"""Stable exponential error identities accepting retained Decimal coefficients."""
from decimal import Decimal as D,localcontext
from math import factorial
import hashlib,time
import numpy as np
import decimal_retained_system as ds
import repaired_precision_2d as control
import precision_eval_2d as pe
import precise_quadrature_2d as pq
import precise_volume_2d as cv
import decimal_coefficient_identity as identity
import run_2d_disk_study as fem

def loads(mesh,space,center,boost,gate,radius=.62,beta0=16.):
    degree=space["p"]
    n=1+max(int(np.max(v)) for v in space['dofs'].values());vol=np.zeros(n,dtype=np.longdouble);correction=vol.copy()
    groups,area,info=pq.volume(mesh,center,radius,degree,space['active'],boost)
    info['volume_sha256']=pq.canonical_digest(groups);checks=[]
    for eid,points,weights in groups:
        r,s,inv=control.geometry(mesh.elements[eid],points);v,_=pe.horner(space['basis'][eid],r,s,np.longdouble)
        u=np.exp(points[:,0]+points[:,1]);local=v@(-2*u*weights);cv.add(vol,correction,space['dofs'][eid],local)
        if len(checks)<33:
            with localcontext() as ctx:
                ctx.prec=100;truth=(ds.exact_decimal(points[0,0])+ds.exact_decimal(points[0,1])).exp()
                checks.append(float(abs(ds.exact_decimal(u[0])-truth)/truth))
    del groups
    boundary,bc=pq.boundary(mesh,center,radius,degree,space['active'],boost);info['boundary_sha256']=pq.canonical_digest(boundary)
    with localcontext() as ctx:
        ctx.prec=80;pen=[D(0) for _ in range(n)];normal=pen[:];alpha=ds.exact_decimal(np.float64(beta0))*D(degree)**2/ds.exact_decimal(mesh.h)
        for eid,points,normals,weights in boundary:
            r,s,inv=control.geometry(mesh.elements[eid],points);v,gref=pe.horner(space['basis'][eid],r,s,np.longdouble)
            g=np.einsum('ab,ibq->iaq',inv,gref);dn=np.einsum('iaq,qa->iq',g,normals);u=np.exp(points[:,0]+points[:,1])
            wu=[ds.exact_decimal(w)*ds.exact_decimal(x) for w,x in zip(weights,u)]
            for i,a in enumerate(space['dofs'][eid]):
                a=int(a);pen[a]+=sum((ds.exact_decimal(x)*w for x,w in zip(v[i],wu)),D(0));normal[a]-=sum((ds.exact_decimal(x)*w for x,w in zip(dn[i],wu)),D(0))
        rhs=[ds.exact_decimal(a)+b+alpha*c for a,b,c in zip(vol,normal,pen)]
    gate(f'q{boost}_exponential_sample_reference',max(checks),5e-18)
    return rhs,dict(quadrature=info,exponential_sample_checks=len(checks))

def expansion(e,basis,C,degree):
    with localcontext() as ctx:
        ctx.prec=80
        a=sum((ds.exact_decimal(v) for v in e.xy[0]),D(0));b=sum((ds.exact_decimal(v) for v in e.J[:,0]),D(0));c=sum((ds.exact_decimal(v) for v in e.J[:,1]),D(0))
        z0=a+(b+c)/3;shift=a-z0;E=z0.exp();target={}
        for total in range(degree+1):
            for i in range(total+1):
                j=total-i
                target[i,j]=E*identity.power(b,i)*identity.power(c,j)/D(factorial(i)*factorial(j))*sum((identity.power(shift,k)/D(factorial(k)) for k in range(degree+1-total)),D(0))
        out=[]
        for coefficients in C:
            p={ij:-v for ij,v in target.items()}
            for coefficient,poly in zip(coefficients,basis):
                value=identity.exact_value(coefficient)
                for ij,v in poly.items():p[ij]+=value*identity.decimal_value(v)
            out.append(p)
    return out,(z0,E)

def tail(t,start):
    term=t**start/np.longdouble(factorial(start));out=term.copy()
    for k in range(start+1,50):term=term*t/k;out+=term
    return out

def polynomials(mesh,space,C,gate):
    output={};degree=space["p"]
    for eid in space['active']:
        e=mesh.elements[eid];local=C[:,space['dofs'][eid]];expanded,(z0,E)=expansion(e,space['basis'][eid],local,degree)
        point=e.xy[0]+e.J@np.array([.2,.3]);rounded=[identity.rounded_polynomial(v) for v in expanded]
        with localcontext() as ctx:
            ctx.prec=100;r,s,inv=identity.decimal_geometry(e,point);t=sum((ds.exact_decimal(v) for v in point),D(0));u=t.exp()
            tau=t-z0;T=lambda n:sum((identity.power(tau,k)/D(factorial(k)) for k in range(n+1)),D(0))
            for k,coefficients in enumerate(local):
                value=dr=dss=D(0)
                for coefficient,poly in zip(coefficients,space['basis'][eid]):
                    v=identity.evaluate_decimal({ij:identity.decimal_value(x) for ij,x in poly.items()},r,s);c=identity.exact_value(coefficient);value+=c*v[0];dr+=c*v[1];dss+=c*v[2]
                direct=[value-u,*[row[0]*dr+row[1]*dss-u for row in inv]]
                v=identity.identity_point_error(e,expanded[k],point,100)
                indirect=[v[0]-E*(tau.exp()-T(degree)),*[x-E*(tau.exp()-T(degree-1)) for x in v[1:]]]
                gate(f'e{eid}_field{k}_exponential_identity',float(max(abs(a-b) for a,b in zip(direct,indirect))),1e-60)
                sampled=identity.sample_error(e,rounded[k],point[None,:],pe.horner,control.geometry)[0]
                tt=np.sum(point.astype(np.longdouble))-np.longdouble(str(z0));ee=np.longdouble(str(E))
                sampled-=ee*np.array([tail(np.array(tt),degree+1),tail(np.array(tt),degree),tail(np.array(tt),degree)])
                expected=np.array([str(v) for v in direct],dtype=np.longdouble)
                gate(f'e{eid}_field{k}_stable_sample',np.max(np.abs(sampled-expected))/(1e-32+np.max(np.abs(expected))),1e-10)
        with localcontext() as ctx:
            ctx.prec=80;difference={ij:expanded[0][ij]-expanded[1][ij] for ij in expanded[0]}
        output[eid]=(rounded+[identity.rounded_polynomial(difference)],np.longdouble(str(z0)),np.longdouble(str(E)))
    return output

def measure(mesh,center,space,polys,boost,radius=.62):
    degree=space["p"]
    tick=time.perf_counter();groups,area=fem.volume_quadrature_groups(mesh,center,radius,degree,space['active'],boost);batches={e:([],[]) for e in space['active']};digest=hashlib.sha256()
    for eid,points,weights in groups:
        digest.update(str(eid).encode()+b':');digest.update(points.tobytes());digest.update(weights.tobytes());batches[eid][0].append(points);batches[eid][1].append(weights)
    del groups;sums=np.zeros((3,2),dtype=np.longdouble)
    for eid,(points,weights) in batches.items():
        points=np.concatenate(points).astype(np.longdouble);weights=np.concatenate(weights).astype(np.longdouble);p,z0,E=polys[eid]
        r,s,inv=control.geometry(mesh.elements[eid],points);v,gref=pe.horner(p,r,s,np.longdouble);g=np.einsum('ab,ibq->iaq',inv,gref);t=points[:,0]+points[:,1]-z0
        v[:2]-=E*tail(t,degree+1);g[:2]-=E*tail(t,degree)[None,None,:]
        sums[:,0]+=(v*v)@weights;sums[:,1]+=np.sum(g*g,axis=1)@weights
    norms=np.sqrt(sums).astype(float)
    return dict(independent_boost=boost,solutions=norms[:2].tolist(),differences=norms[2:].tolist(),ordered_groups_sha256=digest.hexdigest(),seconds=time.perf_counter()-tick),area
