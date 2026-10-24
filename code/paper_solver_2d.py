"""Paper adapter: practical assembly, low-degree support and ordinary data.

Reference-table preparation accepts exact stored polynomials. All element
evaluation, quadrature, residual QR, matrix accumulation and solves use float64.
The caller supplies the repaired space; this module never reads a reference
solution or uses Decimal arithmetic.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import comb
import time
import numpy as np
import run_2d_disk_study as fem
import conforming_basis_2d as repair
import radial_rounding_2d as radial
import workstation_case as wc


def exact(v):
    return v if isinstance(v,F) else F(*v.as_integer_ratio()) if hasattr(v,'as_integer_ratio') else F(v)


@lru_cache(maxsize=24)
def indices(p):return [(i,t-i) for t in range(p+1) for i in range(t+1)]


def controls(polys,p):
    out=[]
    for poly in polys:
        values={ij:exact(v) for ij,v in poly.items()}
        out.append([float(sum((v*F(comb(i,k)*comb(j,l),comb(p,k+l)*comb(k+l,k))
                               for (k,l),v in values.items() if k<=i and l<=j),F(0)))
                    for i,j in indices(p)])
    return np.asarray(out,dtype=float)


def prepare(polys):
    p=max(i+j for poly in polys for i,j in poly)
    deriv=[{(i-1,j):i*exact(v) for (i,j),v in poly.items() if i} for poly in polys]
    deriv += [{(i,j-1):j*exact(v) for (i,j),v in poly.items() if j} for poly in polys]
    return p,controls(polys,p),controls(deriv,p-1) if p else None


def weights(p,r,s):
    a=1-r-s
    powers=[[np.ones_like(r)] for _ in range(3)]
    for x,pw in zip((r,s,a),powers):
        for _ in range(p):pw.append(pw[-1]*x)
    return np.asarray([comb(p,i+j)*comb(i+j,i)*powers[0][i]*powers[1][j]*powers[2][p-i-j]
                       for i,j in indices(p)])


def evaluate_polys(table,r,s):
    p,C,G=table;v=C@weights(p,r,s)
    if p:
        dg=G@weights(p-1,r,s);g=np.stack((dg[:len(C)],dg[len(C):]),axis=1)
    else:g=np.zeros((len(C),2,len(r)))
    return v,g


class Basis:
    def __init__(self,space):
        self.tables={}
        self.element={}
        for eid,polys in space['basis'].items():
            key=id(polys)
            if key not in self.tables:
                _,_,eq,factors,bq=repair._FULL[key]
                self.tables[key]=(prepare(eq) if eq else None,np.asarray(factors,dtype=int).reshape(-1,2),prepare(bq) if bq else None)
            self.element[eid]=key

    def reference(self,eid,r,s):
        r=np.asarray(r,dtype=float);s=np.asarray(s,dtype=float)
        eq,factors,bq=self.tables[self.element[eid]]
        v=np.array([1-r-s,r,s]);g=np.broadcast_to(np.array([[-1.,-1.],[1.,0.],[0.,1.]])[:,:,None],(3,2,len(r)))
        values=[v];grads=[g]
        if eq is not None:
            q,dq=evaluate_polys(eq,r,s);lo,hi=factors.T
            ef=v[lo]*v[hi];df=g[lo]*v[hi,None,:]+g[hi]*v[lo,None,:]
            values.append(q*ef);grads.append(dq*ef[:,None,:]+q[:,None,:]*df)
        if bq is not None:
            q,dq=evaluate_polys(bq,r,s);f=r*s*(1-r-s)
            df=np.stack((s*(1-2*r-s),r*(1-r-2*s)))
            values.append(q*f);grads.append(dq*f[None,None,:]+q[:,None,:]*df[None,:,:])
        return np.concatenate(values),np.concatenate(grads)

    def physical(self,e,points):
        # Explicit 2x2 map avoids a generic solve for every point batch.
        a,b,c,d=e.J.ravel();det=a*d-b*c;delta=np.asarray(points,dtype=float)-e.xy[0]
        r=(d*delta[:,0]-b*delta[:,1])/det;s=(-c*delta[:,0]+a*delta[:,1])/det
        v,g=self.reference(e.eid,r,s)
        invT=np.array([[d,-c],[-b,a]])/det
        return v,np.einsum('ab,ibq->iaq',invT,g)


def classify(mesh,center,radius):
    """Strict binary64 distance comparison, without an artificial radius gap."""
    active=[];interior=[];cut=[]
    for e in mesh.elements:
        edges=np.roll(e.xy,-1,axis=0)-e.xy;offset=center-e.xy
        inside=np.all(edges[:,0]*offset[:,1]-edges[:,1]*offset[:,0]>=0)
        t=np.clip(np.sum(offset*edges,axis=1)/np.sum(edges*edges,axis=1),0,1)
        closest=offset-t[:,None]*edges
        distance=0. if inside else np.min(np.sum(closest*closest,axis=1))
        if distance<radius*radius:
            active.append(e.eid)
            (interior if np.all(np.sum(offset*offset,axis=1)<radius*radius) else cut).append(e.eid)
    return active,interior,cut


def setup(case):
    """Strict classification drives both the DOF inventory and repaired space."""
    original=fem.classify_elements
    try:
        fem.classify_elements=classify
        mesh,center,ndof=wc.geometry(case)
    finally:fem.classify_elements=original
    active,interior,cut=classify(mesh,center,case['radius'])
    dof_map,dofs,basis=fem.build_dofs(mesh,active,case['p']);patterns={}
    for eid in active:
        v=mesh.elements[eid].vids
        patterns[(v[0]<v[1],v[1]<v[2],v[2]<v[0])]=basis[eid]
    space=dict(active=active,interior=interior,cut=cut,dofs=dofs,basis=basis,patterns=patterns,p=case['p'])
    if case['p']>=3:
        space=repair.repair_space(space)
    else:
        # At p1/p2 some orientation patterns are equal polynomials. Avoid the
        # legacy repair's one-to-one reverse lookup of those duplicate lists.
        fixed={}
        for pattern in patterns:
            polys=[dict(v) for v in repair.LAMBDAS];quotients=[];factors=[]
            for edge,(a,b) in enumerate(((0,1),(1,2),(2,0))):
                lo,hi=(a,b) if pattern[edge] else (b,a)
                for degree in range(2,case['p']+1):
                    q=repair.edge_quotient(degree,repair.LAMBDAS[lo],repair.LAMBDAS[hi])
                    quotients.append(q);factors.append((lo,hi))
                    polys.append(repair.mul(repair.mul(repair.LAMBDAS[lo],repair.LAMBDAS[hi]),q))
            repair._FULL[id(polys)]=(polys,polys[:3],quotients,factors,[])
            fixed[pattern]=polys
        def pattern(eid):
            v=mesh.elements[eid].vids;return (v[0]<v[1],v[1]<v[2],v[2]<v[0])
        space.update(basis={eid:fixed[pattern(eid)] for eid in active},patterns=fixed)
    assert len(dof_map)==ndof==1+max(int(max(v)) for v in dofs.values())
    return mesh,center,space,Basis(space),dof_map


def data(points,case,center):
    """Ordinary manufactured solution/gradient/Poisson load in input dtype."""
    points=np.asarray(points);kind=case.get('solution_kind','exponential')
    if kind=='exponential':
        u=np.exp(points[:,0]+points[:,1]);return u,np.column_stack((u,u)),-2*u
    assert kind=='radial_power'
    alpha=case.get('singularity_alpha',2.5);assert alpha>1
    delta=points-np.asarray(center,dtype=points.dtype)
    radius=np.sqrt(np.sum(delta*delta,axis=1));positive=radius>0
    factor=np.zeros_like(radius);factor[positive]=alpha*radius[positive]**(alpha-2)
    # alpha<2 is singular at the center: quadrature must avoid that node.
    assert alpha>=2 or np.all(positive),'Singular forcing sampled at the center'
    return radius**alpha,delta*factor[:,None],-(alpha*alpha)*radius**(alpha-2)


def basis_checks(mesh,space,basis,dof_map):
    """Independent exact traces and linear reproduction, including p1/p2."""
    from decimal import Decimal as D,localcontext
    import decimal_coefficient_identity as identity
    refs=((F(0),F(0)),(F(1),F(0)),(F(0),F(1)))
    edges={};shared=0;worst=0.;linear_error=0.
    c=np.zeros(len(dof_map))
    for key,idx in dof_map.items():
        if key[0]=='v':
            x,y=mesh.vertices[key[1]];c[idx]=1+x-2*y
    for eid,polys in space['basis'].items():
        e=mesh.elements[eid]
        for a,b in ((0,1),(1,2),(2,0)):
            lo,hi=(a,b) if e.vids[a]<e.vids[b] else (b,a)
            values={int(idx):repair.restrict(poly,refs[lo],refs[hi]) for idx,poly in zip(space['dofs'][eid],polys)}
            values={idx:v for idx,v in values.items() if v};key=tuple(sorted((e.vids[a],e.vids[b])))
            if key in edges:assert values==edges[key];shared+=1
            else:edges[key]=values
        rs=np.array([[.125,.375],[.499,.499],[0.,0.],[1.,0.],[0.,1.]])
        points=e.xy[0]+rs@e.J.T;v,g=basis.physical(e,points)
        local=c[space['dofs'][eid]]
        linear_error=max(linear_error,float(np.max(abs(local@v-(1+points[:,0]-2*points[:,1])))),
                         float(np.max(abs(np.einsum('i,iaq->aq',local,g)-np.array([1.,-2.])[:,None]))))
        with localcontext() as ctx:
            ctx.prec=80
            for q,point in enumerate(points):
                rr,ss,inv=identity.decimal_geometry(e,point)
                for k,poly in enumerate(polys):
                    vv,dr,ds=identity.evaluate_decimal({ij:identity.decimal_value(x) for ij,x in poly.items()},rr,ss)
                    truth=[vv,*[row[0]*dr+row[1]*ds for row in inv]]
                    for got,expected in zip((v[k,q],g[k,0,q],g[k,1,q]),truth):
                        worst=max(worst,float(abs(D.from_float(float(got))-expected)/max(D(1),abs(expected))))
    return dict(shared_edges=shared,nonconforming_edges=0,Decimal_basis_scaled_error=worst,linear_reproduction_error=linear_error)


def add(total,correction,value):
    y=value-correction;t=total+y;correction[:]=(t-total)-y;total[:]=t


def assemble(mesh,space,center,case,basis,boost):
    tick=time.perf_counter();p=case['p'];n=1+max(int(max(v)) for v in space['dofs'].values())
    old=fem.ray_edge_radii
    try:
        fem.ray_edge_radii=lambda m,c,R,d:radial.radii(m,c,R,d,15)
        volume,area=fem.volume_quadrature_groups(mesh,center,case['radius'],p,space['active'],boost)
    finally:fem.ray_edge_radii=old
    boundary=fem.boundary_quadrature_groups(mesh,center,case['radius'],p,space['active'],boost)
    boundary_elements=sorted(set(z[0] for z in boundary))
    timings=dict(quadrature=time.perf_counter()-tick)
    result={key:np.zeros((n,n)) for key in ('K','Mb','Cn')}
    result.update({key:np.zeros(n) for key in ('fvol','fbc_cons','fbc_pen')})
    count={};worst_outside=0.
    for kind,groups in (('volume',volume),('boundary',boundary)):
        tick=time.perf_counter();batches={eid:[] for eid in space['active']}
        for z in groups:batches[z[0]].append(z[1:])
        count[kind]=sum(len(z[-1]) for z in groups)
        for eid,parts in batches.items():
            if not parts:continue
            e=mesh.elements[eid];dofs=space['dofs'][eid];size=len(dofs)
            arrays=[np.concatenate([z[k] for z in parts]) for k in range(len(parts[0]))]
            matrices=('K',) if kind=='volume' else ('Mb','Cn')
            vectors=('fvol',) if kind=='volume' else ('fbc_cons','fbc_pen')
            local={name:np.zeros((size,size) if name in matrices else size) for name in matrices+vectors}
            corrections={name:np.zeros_like(v) for name,v in local.items()}
            for start in range(0,len(arrays[0]),2048):
                point=arrays[0][start:start+2048];w=arrays[-1][start:start+2048]
                assert np.all(w>0)
                v,g=basis.physical(e,point);u,_,forcing=data(point,case,center)
                if kind=='volume':
                    values=dict(K=(g[:,0]*w)@g[:,0].T+(g[:,1]*w)@g[:,1].T,fvol=v@(forcing*w))
                else:
                    normals=arrays[1][start:start+2048];dn=np.einsum('iaq,qa->iq',g,normals)
                    values=dict(Mb=(v*w)@v.T,Cn=(dn*w)@v.T,fbc_cons=-dn@(w*u),fbc_pen=v@(w*u))
                for name,value in values.items():add(local[name],corrections[name],value)
            for name,value in local.items():result[name][np.ix_(dofs,dofs) if name in matrices else dofs]+=value
        timings[kind+'_integration']=time.perf_counter()-tick
    del volume,boundary
    patches=fem.build_patches(mesh,space['active'],boundary_elements,case['patch_layers'])
    tick=time.perf_counter();S,diagnostics=stabilization(mesh,space,basis,patches,boost,n)
    timings['stabilization']=time.perf_counter()-tick
    result['S1']=S
    alpha=p*p/mesh.h;base=result['K']+case['gamma1']*S-result['Cn']-result['Cn'].T
    A=base+case['beta0']*alpha*result['Mb'];E=result['K']+case['gamma1']*S+alpha*result['Mb']
    asymmetry=float(np.max(abs(A-A.T))/np.max(abs(A)))
    A=(A+A.T)/2;E=(E+E.T)/2
    rhs=result['fvol']+result['fbc_cons']+case['beta0']*alpha*result['fbc_pen']
    assert all(a.dtype==np.float64 for a in (A,E,rhs,*result.values()))
    return A,E,rhs,result,dict(timings=timings,point_counts=count,area=area,patches=patches,
                              patch_diagnostics=diagnostics,relative_asymmetry=asymmetry)


def stabilization(mesh,space,basis,patches,boost,n):
    p=space['p'];r,s,w=fem.triangle_quadrature(max(6,p+3+boost));monomials=indices(p)
    cache={};S=np.zeros((n,n));diagnostics=[]
    for ix,patch in enumerate(patches):
        dofs=np.unique(np.concatenate([space['dofs'][eid] for eid in patch]));positions={int(g):k for k,g in enumerate(dofs)}
        vertices=np.vstack([mesh.elements[eid].xy for eid in patch]);center=(vertices.min(0)+vertices.max(0))/2
        scale=max(float(np.max(vertices.max(0)-vertices.min(0)))/2,mesh.h/2)
        Fm=np.zeros((2*len(r)*len(patch),len(dofs)));G=np.zeros((len(Fm),len(monomials)-1));offset=0
        for eid in patch:
            if eid not in cache:
                e=mesh.elements[eid];points=e.xy[0]+np.column_stack((r,s))@e.J.T
                # Reference points are already available on full triangles.
                _,gr=basis.reference(eid,r,s);a,b,c,d=e.J.ravel();det=a*d-b*c
                grad=np.einsum('ab,ibq->iaq',np.array([[d,-c],[-b,a]])/det,gr)
                cache[eid]=(points,grad,np.sqrt(det*w))
            points,grad,sqrtw=cache[eid];_,qg=fem.eval_patch_polynomials(monomials,points,center,scale)
            loc=[positions[int(g)] for g in space['dofs'][eid]]
            for axis in (0,1):
                end=offset+len(r);Fm[offset:end,loc]=(grad[:,axis]*sqrtw).T;G[offset:end]=(qg[1:,axis]*sqrtw).T;offset=end
        Q,R=np.linalg.qr(G,mode='reduced');residual=Fm-Q@(Q.T@Fm)
        raw=residual.T@residual;S[np.ix_(dofs,dofs)]+=raw
        diagnostics.append(dict(patch=ix,rows=len(Fm),dofs=len(dofs),
            qr_orthogonality=float(np.max(abs(Q.T@Q-np.eye(Q.shape[1])))),
            gradient_R_condition=float(np.linalg.cond(R))))
    return S,diagnostics


def solve(A,b):
    tick=time.perf_counter();assert np.all(np.diag(A)>0)
    scale=1/np.sqrt(np.diag(A));scaled=scale[:,None]*A*scale[None,:]
    x=scale*np.linalg.solve(scaled,scale*b)
    assert x.dtype==np.float64 and np.all(np.isfinite(x))
    return x,dict(seconds=time.perf_counter()-tick,scaling='diagonal',solver='numpy.linalg.solve',refinement_steps=0)


def spectrum(A,E):
    tick=time.perf_counter();assert np.all(np.diag(E)>0)
    d=1/np.sqrt(np.diag(E));es=E*d[:,None]*d[None,:];L=np.linalg.cholesky(es)
    Q=np.linalg.solve(L,np.diag(d));H=Q@A@Q.T;H=(H+H.T)/2
    values,vectors=np.linalg.eigh(H);x=Q.T@vectors[:,0]
    # Extended arithmetic checks the float64 eigentransform only; it never
    # supplies the practical operator, solve, coefficients, or eigenvectors.
    ld=np.longdouble;aa=A.astype(ld);ee=E.astype(ld);xx=x.astype(ld);qq=Q.astype(ld)
    energy=xx@(ee@xx);residual=aa@xx-ld(values[0])*(ee@xx)
    white=qq@residual
    return dict(lambda_min=float(values[0]),lambda_max=float(values[-1]),
                metric='unregularized',shift=0.,seconds=time.perf_counter()-tick,
                whitened_metric_error=float(np.max(np.sum(abs(qq@ee@qq.T-np.eye(len(A),dtype=ld)),axis=1))),
                whitened_residual=float(np.sqrt(white@white/energy)),
                Rayleigh=float((xx@(aa@xx))/energy))
