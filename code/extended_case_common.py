"""Shared records and geometry checks for individually frozen analytic cases."""
from decimal import Decimal as D,localcontext
from math import comb,factorial
import gzip,hashlib,json,multiprocessing as mp
import numpy as np
import circle_moments_2d as cm
import decimal_retained_system as ds

def write(path,value):
    with (gzip.open(path,'wt') if str(path).endswith('.gz') else path.open('w')) as f:json.dump(value,f,indent=None if str(path).endswith('.gz') else 2,allow_nan=False)

def read(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else path.open()) as f:return json.load(f)

def sparse(a):return [[[j,str(v)] for j,v in enumerate(row) if v] for row in a]

def identity(mesh,space,center,case):
    records=[]
    for eid in space['active']:
        e=mesh.elements[eid]
        records.append(dict(element=eid,vertices=[[float(v).hex() for v in row] for row in e.xy],J=[[float(v).hex() for v in row] for row in e.J],dofs=list(map(int,space['dofs'][eid])),basis=[[[i,j,str(v)] for (i,j),v in sorted(poly.items())] for poly in space['basis'][eid]]))
    value=dict(N=case['N'],p=case['p'],center=[float(v).hex() for v in center],radius=float(case['radius']).hex(),patch_layers=case['patch_layers'],beta0=float(case['beta0']).hex(),gamma1=float(case['gamma1']).hex(),gamma0=float(case['gamma0']).hex(),elements=records)
    return dict(sha256=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest(),definition=value)

def element_moments(args):
    eid,vertices,center,radius,degree=args;V,M,info=cm.moments(vertices,center,radius,degree)
    reflected=[[z[1],z[0]] for z in (vertices[0],vertices[2],vertices[1])];W,T,_=cm.moments(reflected,[center[1],center[0]],radius,degree)
    with localcontext() as ctx:
        ctx.prec=80;reflection=max([abs(V[i,j]-W[j,i]) for i,j in V]+[abs(M[i,j]-T[j,i]) for i,j in M])
    return eid,V,M,dict(info,reflection_maximum=str(reflection))

def moments(mesh,space,center,radius,workers,gate):
    degree=2*space['p'];jobs=[]
    for z in cm.self_check(degree):
        for key in ('maximum_volume_error','maximum_boundary_error'):gate(z['case']+'_'+key,D(z[key]),1e-60)
    with localcontext() as ctx:
        ctx.prec=80
        for eid in space['active']:
            e=mesh.elements[eid];x0=list(map(ds.exact_decimal,e.xy[0]));J=[[ds.exact_decimal(v) for v in row] for row in e.J]
            vertices=[x0,[x0[i]+J[i][0] for i in (0,1)],[x0[i]+J[i][1] for i in (0,1)]]
            gate(f'e{eid}_stored_affine_vertex_identity',max(abs(vertices[i][j]-ds.exact_decimal(e.xy[i,j])) for i in range(3) for j in (0,1)),0)
            jobs.append((eid,vertices,list(map(ds.exact_decimal,center)),ds.exact_decimal(radius),degree))
    with mp.get_context('fork').Pool(workers) as pool:values=list(pool.imap(element_moments,jobs))
    saved=[];checks=[];fractions=[]
    with localcontext() as ctx:
        ctx.prec=80;indices=list(values[0][1]);totalV={ij:D(0) for ij in indices};totalM=totalV.copy();cx,cy=map(ds.exact_decimal,center);R=ds.exact_decimal(radius)
        for eid,V,M,info in values:
            gate(f'e{eid}_reflection',D(info['reflection_maximum']),1e-60)
            a,b,c,d=map(ds.exact_decimal,mesh.elements[eid].J.ravel());full=(a*d-b*c)/2;area=V[0,0]
            gate(f'e{eid}_negative_area',max(D(0),-area),1e-65);gate(f'e{eid}_excess_area',max(D(0),area-full),1e-65)
            fractions.append(dict(element=eid,fraction=str(area/full),circle_length=str(M[0,0])))
            for ij in indices:totalV[ij]+=V[ij];totalM[ij]+=M[ij]
            saved.append(dict(element=eid,volume=[[i,j,str(v)] for (i,j),v in V.items()],boundary=[[i,j,str(v)] for (i,j),v in M.items()],geometry=info))
        def centered(i,j,boundary):
            if i%2 or j%2:return D(0)
            a,b=i//2,j//2;angular=2*cm.PI*D(factorial(i)*factorial(j))/D(4**(a+b)*factorial(a)*factorial(b)*factorial(a+b))
            return angular*R**(i+j+1) if boundary else angular*R**(i+j+2)/D(i+j+2)
        for i,j in indices:
            for label,observed,boundary in [('volume',totalV[i,j],False),('boundary',totalM[i,j],True)]:
                exact=sum((D(comb(i,k)*comb(j,l))*cm.power(cx,i-k)*cm.power(cy,j-l)*centered(k,l,boundary) for k in range(i+1) for l in range(j+1)),D(0));error=abs(observed-exact)
                gate(f'global_{label}_{i}_{j}',error,1e-60);checks.append(dict(kind=label,i=i,j=j,value=str(observed),exact=str(exact),absolute_error=str(error)))
    return saved,dict(global_moments=checks,cell_fractions=fractions,minimum_cell_fraction=str(min(D(z['fraction']) for z in fractions)))

def integration(records,label,gate):
    assert len(records)==2
    for key in ('solutions','differences'):
        lo,hi=(np.array(row[key]) for row in records)
        for i in range(len(hi)):
            for k,norm in enumerate(('L2','H1')):gate(f'{label}_{key}{i}_{norm}_integration',abs(lo[i,k]-hi[i,k])/max(hi[i,k],1e-300),1e-4,'integration_accuracy')
