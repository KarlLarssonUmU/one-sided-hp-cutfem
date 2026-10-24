"""Paired-rule flux, fitted-polygon and extension-seminorm experiments.

No shifted metrics. Flux H is the normal-derivative Gram, not boundary mass.
All writes are relative to the run directory supplied by the batch script.
"""
from pathlib import Path
import csv,json,time,sys,platform,getpass,os,traceback
import numpy as np
import scipy
from scipy import linalg as la
from scipy.spatial import Delaunay
import paper_solver_2d as solver
import run_2d_disk_study as fem
import workstation_case as wc

dump=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sym=lambda a:(a+a.T)/2

def flux(mesh,space,basis,groups):
    n=1+max(int(max(v)) for v in space['dofs'].values());H=np.zeros((n,n))
    for eid,xy,normals,w in groups:
        _,g=basis.physical(mesh.elements[eid],xy)
        dn=np.einsum('iaq,qa->iq',g,normals);ii=space['dofs'][eid]
        H[np.ix_(ii,ii)]+=(dn*w)@dn.T
    return sym(H)

def extremal(H,G,pin):
    """Constant quotient, Jacobi equilibration, no regularization or clipping."""
    keep=np.delete(np.arange(len(G)),pin);g=G[np.ix_(keep,keep)];h=H[np.ix_(keep,keep)]
    result=dict(resolved=False,shift=0.,quotient='one vertex coefficient pinned')
    if np.min(np.diag(g))<=0:return dict(result,reason='nonpositive metric diagonal')
    d=1/np.sqrt(np.diag(g));g=sym(g*d[:,None]*d);h=sym(h*d[:,None]*d)
    try:
        L=la.cholesky(g,lower=True,check_finite=False)
        Q=la.solve_triangular(L,np.eye(len(g)),lower=True,check_finite=False)
        W=sym(Q@h@Q.T)
        val,y=la.eigh(W,subset_by_index=[len(g)-1,len(g)-1],check_finite=False)
        z=Q.T@y[:,0];lam=float(val[0]);ld=np.longdouble
        gg=g.astype(ld);hh=h.astype(ld);zz=z.astype(ld)
        residual=hh@zz-ld(lam)*(gg@zz)
        rr=Q.astype(ld)@residual
        rho=float(np.sqrt(rr@rr)/max(abs(lam),1.))
        # Check the whitening itself; ill conditioning is not silently repaired.
        metric=float(np.max(np.sum(abs(Q@g@Q.T-np.eye(len(g))),axis=1)))
        ray=float((zz@(hh@zz))/(zz@(gg@zz)))
        resolved=bool(lam>0 and metric<1e-5 and rho<1e-5 and abs(ray-lam)<1e-5*max(1.,abs(lam)))
        x=np.zeros(len(G));x[keep]=d*z
        return dict(result,resolved=resolved,value=lam,rayleigh=ray,relative_residual=rho,
                    whitening_error=metric,reason='' if resolved else 'eigensystem accuracy gate'),x
    except la.LinAlgError as exc:
        return dict(result,reason=str(exc)),None

def spectrum_record(H,K,S,case,dof_map):
    # An interior vertex is a well-scaled representative of the constant quotient.
    vertex=[i for key,i in dof_map.items() if key[0]=='v'];pin=max(vertex,key=lambda i:K[i,i])
    records={};vectors={}
    for name,gamma in [('unit',1.),('computational',case['gamma1']),('unstabilized',0.)]:
        G=K+gamma*S
        got=extremal(H*case['h']/case['p']**2,G,pin)
        result,x=got if isinstance(got,tuple) else (got,None)
        # Each diagonal is a positive quadrature integral; this is a lower bound
        # even when the full unstabilized metric cannot be reliably inverted.
        result['basis_witness_lower_bound']=float(np.max(case['h']*np.diag(H)/(case['p']**2*np.diag(G))))
        records[name]=result
        if x is not None:vectors[name]=x
    return records,vectors

def linear_checks(mesh,dof_map,H,K,area,perimeter,center,R):
    x=np.zeros(len(H));constant=np.zeros(len(H))
    for key,i in dof_map.items():
        if key[0]=='v':constant[i]=1.;x[i]=mesh.vertices[key[1],0]
    checks=dict(volume_linear_relative=abs(float(x@K@x)/area-1),
        constant_K=float(np.linalg.norm(K@constant)/max(np.linalg.norm(K),1.)),
        constant_H=float(np.linalg.norm(H@constant)/max(np.linalg.norm(H),1.)))
    if R is not None:checks['circle_flux_linear_relative']=abs(float(x@H@x)/(np.pi*R)-1)
    assert max(checks.values())<2e-10,checks
    return checks

def cut_study(run, degrees=range(1,9), path=None, epsilon=None):
    """Exactly 96 cases: 8 degrees x 3 paths x 4 offsets; two quadrature rules."""
    rows=list(csv.DictReader((run/'inputs/cut-cases.csv').open()))
    geom=[r for r in rows if r['p']=='2' and r['patch_layers']=='2']
    assert len(geom)==12
    geom=[r for r in geom if (path is None or r['disk']==path) and (epsilon is None or float(r['epsilon'])==epsilon)]
    assert geom
    records=[];original_volume=fem.volume_quadrature_groups;original_boundary=fem.boundary_quadrature_groups
    for row in geom:
        base=wc.normalize_case(dict(N=6,p=8,path=row['disk'],epsilon=float(row['epsilon']),patch_layers=2,
            center=[float(row['center_x']),float(row['center_y'])]))
        mesh,center,_,_,_=solver.setup(base)
        quadratures={}
        for boost in (8,12):
            old=fem.ray_edge_radii
            try:
                fem.ray_edge_radii=lambda m,c,R,d:solver.radial.radii(m,c,R,d,15)
                volume,area=original_volume(mesh,center,base['radius'],8,solver.classify(mesh,center,base['radius'])[0],boost)
            finally:fem.ray_edge_radii=old
            boundary=original_boundary(mesh,center,base['radius'],8,solver.classify(mesh,center,base['radius'])[0],boost)
            quadratures[boost]=(volume,area,boundary)
        for p in degrees:
            start=time.perf_counter();case=dict(base,p=p);mesh,center,space,basis,dm=solver.setup(case)
            record=dict(path=case['path'],epsilon=case['epsilon'],p=p,N=6,L=2,center=center.tolist(),
                ndof=len(dm),h=mesh.h,eta=float(row['minimum_fraction']),rules=[])
            key=f"{case['path']}-e{case['epsilon']:.0e}-p{p}";out=run/'output'/key;out.mkdir()
            for boost in (8,12):
                volume,area,boundary=quadratures[boost]
                try:
                    fem.volume_quadrature_groups=lambda *a,**kw:(volume,area)
                    fem.boundary_quadrature_groups=lambda *a,**kw:boundary
                    A,E,b,parts,meta=solver.assemble(mesh,space,center,case,basis,boost)
                finally:
                    fem.volume_quadrature_groups=original_volume;fem.boundary_quadrature_groups=original_boundary
                H=flux(mesh,space,basis,boundary)
                checks=linear_checks(mesh,dm,H,parts['K'],np.pi*case['radius']**2,None,center,case['radius'])
                result,vectors=spectrum_record(H,parts['K'],parts['S1'],dict(case,h=mesh.h),dm)
                record['rules'].append(dict(boost=boost,spectra=result,linear_checks=checks,assembly_seconds=meta['timings']))
                np.savez_compressed(out/f'q{boost}.npz',H=H,K=parts['K'],S=parts['S1'],**vectors)
            for name in ('unit','computational','unstabilized'):
                a,b=[r['spectra'][name] for r in record['rules']]
                if 'value' in a and 'value' in b:
                    discrepancy=abs(a['value']-b['value'])/max(abs(b['value']),1.)
                    b['paired_relative_change']=discrepancy;b['resolved']=bool(a['resolved'] and b['resolved'] and discrepancy<1e-4)
                else:b['resolved']=False
            # The stabilized theorem test must pass; unstabilized failures are evidence.
            assert all(record['rules'][-1]['spectra'][k]['resolved'] for k in ['unit','computational']),record
            record['seconds']=time.perf_counter()-start;dump(out/'result.json',record);records.append(record)
            dump(run/'output/cut-results.json',records);print(key,record['rules'][-1]['spectra']['unit']['value'],record['seconds'],flush=True)
        del quadratures
    assert len(records)==len(geom)*len(degrees)

def fitted_mesh():
    R=.62;h=1/3
    interior=np.array([(x,y) for y in np.arange(-1,1.01,h) for x in np.arange(-1,1.01,h) if x*x+y*y<(R-.45*h)**2])
    theta=np.arange(20)*2*np.pi/20;vertices=np.vstack((interior,R*np.column_stack((np.cos(theta),np.sin(theta)))))
    triangles=Delaunay(vertices).simplices;elements=[];edges={}
    for ids in triangles:
        xy=vertices[ids];J=np.column_stack((xy[1]-xy[0],xy[2]-xy[0]))
        if np.linalg.det(J)<0:ids=ids[[0,2,1]];xy=vertices[ids];J=np.column_stack((xy[1]-xy[0],xy[2]-xy[0]))
        eid=len(elements);elements.append(fem.Element(eid,tuple(int(v) for v in ids),xy,J,float(np.linalg.det(J)),np.linalg.inv(J).T))
        for a,b in fem.EDGE_PAIRS:edges.setdefault(tuple(sorted((ids[a],ids[b]))),[]).append((eid,ids[a],ids[b]))
    mesh=fem.Mesh(6,vertices,elements,np.array([[vertices[a],vertices[b]] for a,b in edges]),h)
    bd=[v[0] for v in edges.values() if len(v)==1]
    assert len(bd)==20 and len(elements)>20
    return mesh,bd

def fitted_setup(case,mesh):
    bm=fem.build_mesh;cl=solver.classify
    try:
        fem.build_mesh=lambda N:mesh
        solver.classify=lambda *a:([e.eid for e in mesh.elements],[e.eid for e in mesh.elements],[])
        return solver.setup(dict(case,center=[0.,0.]))
    finally:fem.build_mesh=bm;solver.classify=cl

def fitted_assemble(mesh,bd,space,basis,order):
    n=1+max(int(max(v)) for v in space['dofs'].values());M={k:np.zeros((n,n)) for k in ['K','H','Mb','Cn']}
    r,s,w=fem.triangle_quadrature(order)
    for e in mesh.elements:
        pts=e.xy[0]+np.column_stack((r,s))@e.J.T;v,g=basis.physical(e,pts);ww=w*e.detJ;ii=space['dofs'][e.eid]
        M['K'][np.ix_(ii,ii)]+=(g[:,0]*ww)@g[:,0].T+(g[:,1]*ww)@g[:,1].T
    t,wt=fem.gauss_legendre_interval(order,0,1)
    for eid,a,b in bd:
        edge=mesh.vertices[b]-mesh.vertices[a];length=np.linalg.norm(edge);normal=np.array([edge[1],-edge[0]])/length
        pts=mesh.vertices[a]+t[:,None]*edge;v,g=basis.physical(mesh.elements[eid],pts);dn=np.einsum('iaq,a->iq',g,normal);ww=wt*length;ii=space['dofs'][eid]
        for name,value in dict(H=(dn*ww)@dn.T,Mb=(v*ww)@v.T,Cn=(dn*ww)@v.T).items():M[name][np.ix_(ii,ii)]+=value
    return M

def threshold_condition(M,p,h):
    K,H,Mb,Cn=[M[k] for k in ['K','H','Mb','Cn']];alpha=p*p/h;E=sym(K+alpha*Mb);base=sym(K-Cn-Cn.T)
    d=1/np.sqrt(np.diag(E));L=la.cholesky(E*d[:,None]*d,lower=True);Q=la.solve_triangular(L,np.diag(d),lower=True)
    B=sym(Q@base@Q.T);T=sym(Q@(alpha*Mb)@Q.T)
    lam=lambda beta:float(la.eigh(B+beta*T,subset_by_index=[0,0],eigvals_only=True,check_finite=False)[0])
    lo=0.;hi=16.;assert lam(hi)>1e-8
    while hi-lo>1e-4:
        mid=(lo+hi)/2
        if lam(mid)>1e-8:hi=mid
        else:lo=mid
    A=base+16*alpha*Mb;d=1/np.sqrt(np.diag(A));AJ=sym(A*d[:,None]*d)
    ev=la.eigvalsh(AJ,check_finite=False);assert ev[0]>0
    return dict(beta_lower=lo,beta_upper=hi,nominal_coercivity=lam(16.),kappa_J=float(ev[-1]/ev[0]))

def extension_norm(mesh,space,center,case,patches,boost):
    p=space['p'];r,s,w=fem.triangle_quadrature(p+3+boost);indices=solver.indices(p);total=np.longdouble(0);orth=0.
    for patch in patches:
        vertices=np.vstack([mesh.elements[e].xy for e in patch]);pc=(vertices.min(0)+vertices.max(0))/2;scale=max(float(np.max(vertices.max(0)-vertices.min(0)))/2,mesh.h/2)
        Gs=[];Fs=[]
        for eid in patch:
            e=mesh.elements[eid];xy=e.xy[0]+np.column_stack((r,s))@e.J.T;sw=np.sqrt(w*e.detJ)
            _,qg=fem.eval_patch_polynomials(indices,xy,pc,scale);_,du,_=solver.data(xy.astype(np.longdouble),case,center)
            for axis in (0,1):Gs.append((qg[1:,axis]*sw).T);Fs.append(du[:,axis]*sw)
        G=np.vstack(Gs);F=np.concatenate(Fs);Q,_=la.qr(G,mode='economic',check_finite=False)
        orth=max(orth,float(np.max(abs(Q.T@Q-np.eye(Q.shape[1])))));q=Q.astype(np.longdouble)
        residual=F-q@(q.T@F);total+=residual@residual
    assert orth<1e-12
    return dict(norm=float(np.sqrt(case['gamma1']*total)),qr_orthogonality=orth)

def controls_study(run, degrees=range(1,9)):
    mesh,bd=fitted_mesh();dump(run/'output/fitted-geometry.json',dict(vertices=mesh.vertices.tolist(),triangles=[e.vids for e in mesh.elements],h=mesh.h,
        boundary='inscribed straight-sided regular 20-gon; not the exact disk',radius=.62,
        max_diameter_over_h=max(np.linalg.norm(e.xy[a]-e.xy[b]) for e in mesh.elements for a,b in fem.EDGE_PAIRS)/mesh.h))
    fit=[]
    for p in degrees:
        start=time.perf_counter();case=wc.normalize_case(dict(p=p,N=6,center=[0,0],patch_layers=2));_,center,space,basis,dm=fitted_setup(case,mesh);rules=[]
        for order in (p+3,p+7):
            M=fitted_assemble(mesh,bd,space,basis,order);spec,_=spectrum_record(M['H'],M['K'],np.zeros_like(M['K']),dict(case,h=mesh.h),dm)
            stats=threshold_condition(M,p,mesh.h);rules.append(dict(order=order,spectrum=spec['unstabilized'],**stats))
            linear_checks(mesh,dm,M['H'],M['K'],sum(e.detJ/2 for e in mesh.elements),None,center,None)
            np.savez_compressed(run/'output'/f'fitted-p{p}-q{order}.npz',**M)
        for key in ['kappa_J','nominal_coercivity']:
            assert abs(rules[0][key]/rules[1][key]-1)<1e-4,(p,key,rules)
        assert all(r['spectrum']['resolved'] for r in rules),(p,rules)
        assert abs(rules[0]['spectrum']['value']/rules[1]['spectrum']['value']-1)<1e-5
        assert abs(rules[0]['beta_upper']-rules[1]['beta_upper'])<=2e-4
        fit.append(dict(p=p,ndof=len(dm),rules=rules,seconds=time.perf_counter()-start));dump(run/'output/fitted-results.json',fit);print('fitted',p,fit[-1]['seconds'],flush=True)
    extension=[]
    for disk in ['fixed_A','fixed_B']:
        for p in degrees:
            case=wc.normalize_case(dict(p=p,N=6,path=disk,patch_layers=2));mesh,center,space,basis,dm=solver.setup(case)
            bd=fem.boundary_quadrature_groups(mesh,center,.62,p,space['active'],8);patches=fem.build_patches(mesh,space['active'],sorted(set(g[0] for g in bd)),2)
            values=[extension_norm(mesh,space,center,case,patches,boost) for boost in (8,12)]
            change=abs(values[0]['norm']/values[1]['norm']-1);assert change<1e-5,(disk,p,values)
            extension.append(dict(disk=disk,p=p,N=6,L=2,gamma1=.1,rules=values,relative_change=change))
            dump(run/'output/extension-results.json',extension);print('extension',disk,p,values[-1]['norm'],flush=True)

if __name__=='__main__':
    run=Path(sys.argv[1]).resolve();mode=sys.argv[2]
    assert platform.node().split('.')[0]=='compmath1' and getpass.getuser()=='karl'
    start=time.time();code=1
    dump(run/'environment.json',dict(host=platform.node(),user=getpass.getuser(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,dtype='float64; longdouble residual checks',device='CPU',seed=None,threads=os.environ.get('OPENBLAS_NUM_THREADS')))
    try:
        {'cut':cut_study,'controls':controls_study}[mode](run);code=0
    finally:dump(run/'completion.json',dict(exit_code=code,mode=mode,started=start,finished=time.time(),seconds=time.time()-start,job_id=os.environ.get('SLURM_JOB_ID')))
