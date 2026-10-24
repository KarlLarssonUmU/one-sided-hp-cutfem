"""80-digit Gauss construction and longdouble polar quadrature arithmetic.

The initial control keeps the existing angular partition endpoints. Node and
weight construction, trigonometry, radial intersections and Cartesian mapping
retain longdouble instead of rounding to binary64 or15 decimal places.
"""
from decimal import Decimal as D,localcontext
import math
import numpy as np
import run_2d_disk_study as fem
import decimal_retained_system as ds
_RULES={}

def pi_decimal():
    with localcontext() as ctx:
        ctx.prec=90
        def atan(x):
            term=x;total=x;k=1
            while abs(term)>D('1e-90'):
                term*=-x*x;total+=term/(2*k+1);k+=1
            return total
        return 4*(4*atan(D(1)/5)-atan(D(1)/239))
PI=np.longdouble(str(pi_decimal()));TAU=2*PI

def gauss(n):
    if n in _RULES:return _RULES[n]
    with localcontext() as ctx:
        ctx.prec=80;roots=[];weights=[];maximum=D(0)
        def legendre(x):
            p0=D(1);p1=x
            for k in range(2,n+1):p0,p1=p1,((2*k-1)*x*p1-(k-1)*p0)/k
            return p1,D(n)*(x*p1-p0)/(x*x-1)
        for seed in np.polynomial.legendre.leggauss(n)[0]:
            x=D.from_float(float(seed))
            for _ in range(6):p,derivative=legendre(x);x-=p/derivative
            p,derivative=legendre(x);maximum=max(maximum,abs(p));roots.append(x);weights.append(D(2)/((1-x*x)*derivative*derivative))
        moments=max(abs(sum((w*(x**k if k else D(1)) for x,w in zip(roots,weights)),D(0))-(D(2)/(k+1) if k%2==0 else D(0))) for k in range(2*n))
        assert maximum<D('1e-70') and moments<D('1e-70')
        rule=(np.array([str(x) for x in roots],dtype=np.longdouble),np.array([str(x) for x in weights],dtype=np.longdouble),dict(nodes=n,Decimal80_root_residual=str(maximum),Decimal80_maximum_moment_error=str(moments)))
    _RULES[n]=rule;return rule

def interval(n,a,b):
    x,w,_=gauss(n);a=np.longdouble(a);b=np.longdouble(b)
    return a+(b-a)*(x+1)/2,(b-a)*w/2

def radii(mesh,center,radius,direction):
    d=np.longdouble;a=mesh.edges[:,0,:].astype(d);e=mesh.edges[:,1,:].astype(d)-a;q=a-np.asarray(center,dtype=d)
    den=direction[0]*e[:,1]-direction[1]*e[:,0];good=np.abs(den)>d('1e-13')
    rho=np.full(den.shape,np.nan,dtype=d);t=rho.copy()
    rho[good]=(q[good,0]*e[good,1]-q[good,1]*e[good,0])/den[good]
    t[good]=(q[good,0]*direction[1]-q[good,1]*direction[0])/den[good]
    mask=good&(rho>d('1e-12'))&(rho<d(radius)-d('1e-12'))&(t>-d('1e-11'))&(t<1+d('1e-11'))
    return np.unique(np.concatenate((np.array([0,d(radius)],dtype=d),rho[mask])))

def angular_intervals(angles):
    for i,a in enumerate(angles):
        a=np.longdouble(a);b=np.longdouble(angles[i+1]) if i+1<len(angles) else np.longdouble(angles[0])+TAU
        if b-a>np.longdouble('1e-12'):yield a,b

def volume(mesh,center,radius,p,active,boost):
    active=set(active);d=np.longdouble;center=np.asarray(center,dtype=d);angles=fem.angular_partition(mesh,np.asarray(center,dtype=float),radius,max(1,1+boost//2))
    q=max(5,p+3+boost);groups=[];area={eid:d(0) for eid in active};trace=dict(rays=0,skipped_intervals=0,skipped_length=0.,changed_element_sequence_rays=0)
    for aa,bb in angular_intervals(angles):
        theta,wt=interval(q,aa,bb)
        for t,w in zip(theta,wt):
            direction=np.array([np.cos(t),np.sin(t)],dtype=d);rs=radii(mesh,center,radius,direction);ids=[]
            for ra,rb in zip(rs[:-1],rs[1:]):
                if rb-ra<d('2e-12'):
                    trace['skipped_intervals']+=1;trace['skipped_length']+=float(rb-ra);continue
                eid=mesh.locate(center+(ra+rb)/2*direction)
                if eid not in active:continue
                rho,wr=interval(q,ra,rb);points=center+rho[:,None]*direction;weights=w*wr*rho
                groups.append((eid,points,weights));area[eid]+=weights.sum();ids.append(eid)
            old=fem.ray_edge_radii(mesh,np.asarray(center,dtype=float),radius,np.asarray(direction,dtype=float))
            oldids=[mesh.locate(center+(a+b)/2*direction) for a,b in zip(old[:-1],old[1:]) if b-a>=2e-12 and mesh.locate(center+(a+b)/2*direction) in active]
            trace['rays']+=1;trace['changed_element_sequence_rays']+=int(ids!=oldids)
    return groups,area,dict(trace=trace,gauss=gauss(q)[2],angular_endpoints='saved binary64 partition promoted; final endpoint uses longdouble 2*pi',dtype='longdouble',radial_rounding='none')

def boundary(mesh,center,radius,p,active,boost):
    d=np.longdouble;active=set(active);center=np.asarray(center,dtype=d);q=max(6,p+4+boost)
    angles=fem.boundary_partition(mesh,np.asarray(center,dtype=float),radius,max(1,1+boost//2));groups=[]
    for aa,bb in angular_intervals(angles):
        mid=(aa+bb)/2;eid=mesh.locate(center+d(radius)*np.array([np.cos(mid),np.sin(mid)]))
        if eid not in active:continue
        theta,w=interval(q,aa,bb);normals=np.column_stack((np.cos(theta),np.sin(theta)))
        groups.append((eid,center+d(radius)*normals,normals,d(radius)*w))
    return groups,dict(gauss=gauss(q)[2],angular_endpoints='saved binary64 partition promoted; final endpoint uses longdouble 2*pi')

def self_check():
    checks={}
    for n in (23,24,27,28):
        x,w,info=gauss(n);error=max(abs(np.sum(w*x**k)-(np.longdouble(2)/(k+1) if k%2==0 else 0)) for k in range(2*n))
        assert error<np.longdouble('2e-18');checks[str(n)]=dict(info,longdouble_moment_error=float(error))
    angles=np.array([0.,.17,1.1,3.,6.2],dtype=np.longdouble);maximum=D(0)
    with localcontext() as ctx:
        ctx.prec=80
        for a in angles:
            x=ds.exact_decimal(a);s=x;c=D(1);st=x;ct=D(1)
            for k in range(1,90):st*=-x*x/D((2*k)*(2*k+1));ct*=-x*x/D((2*k-1)*(2*k));s+=st;c+=ct
            maximum=max(maximum,abs(ds.exact_decimal(np.sin(a))-s),abs(ds.exact_decimal(np.cos(a))-c))
    assert maximum<D('2e-18')
    checks['trigonometry_absolute_error']=str(maximum)
    checks['longdouble_mantissa_bits']=np.finfo(np.longdouble).nmant+1
    return checks

def canonical_digest(groups):
    import hashlib
    h=hashlib.sha256();assert np.little_endian and np.finfo(np.longdouble).nmant==63
    for eid,*arrays in groups:
        h.update(str(eid).encode()+b':')
        for array in arrays:
            a=np.ascontiguousarray(array);h.update(str(a.shape).encode())
            # x87 stores80 meaningful bits in16 bytes; exclude padding bytes.
            h.update(a.view(np.uint8).reshape(-1,16)[:,:10].tobytes() if a.dtype==np.longdouble else a.tobytes())
    return h.hexdigest()

def disk_moments(groups,center,radius,degree=8):
    from math import factorial
    points=np.concatenate([g[1] for g in groups]);w=np.concatenate([g[2] for g in groups]);delta=points-np.asarray(center,dtype=np.longdouble)
    xp=[np.ones_like(w)];yp=[xp[0]]
    for k in range(degree):xp.append(xp[-1]*delta[:,0]);yp.append(yp[-1]*delta[:,1])
    checks=[]
    with localcontext() as ctx:
        ctx.prec=80;R=ds.exact_decimal(np.float64(radius));pi=pi_decimal();area=pi*R*R
        for total in range(degree+1):
            for i in range(total+1):
                j=total-i;exact=D(0)
                if i%2==0 and j%2==0:
                    a,b=i//2,j//2;exact=pi*R**(i+j+2)*D(factorial(i)*factorial(j))/D(4**(a+b)*factorial(a)*factorial(b)*factorial(a+b+1))
                observed=np.sum(w*xp[i]*yp[j]);error=abs(ds.exact_decimal(observed)-exact)/(area*R**total)
                checks.append(dict(i=i,j=j,scaled_error=str(error)))
    return checks
