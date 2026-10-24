"""Eight fixed high-degree radial cases; squared-radial Duffy rule at the centre.

u=r^(5/2), grad u~r^(3/2), f~r^(1/2) become integer powers along
x=c+t^2(y-c). This avoids ordinary Gauss integration of a half power at 0.
Failures remain records and cause a nonzero final exit, without omitting cases.
"""
from pathlib import Path
import os,sys,time,json,traceback,platform,getpass
import numpy as np
import scipy
from scipy import linalg as la
import run_2d_disk_study as fem
import paper_solver_2d as solver
import workstation_case as wc

dump=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def central_rule(e,center,p,boost):
    order=max(p+3+boost,2*p+5);t,wt=fem.gauss_legendre_interval(order,0,1)
    z,wz=fem.gauss_legendre_interval(p+3+boost,0,1);points=[];weights=[]
    for a,b in fem.EDGE_PAIRS:
        aa=e.xy[a]-center;bb=e.xy[b]-center;cross=abs(float(np.linalg.det(np.stack((aa,bb)))))
        if cross<1e-15:continue
        edge=aa[None,:]+z[:,None]*(bb-aa)
        points.append((center+t[:,None,None]**2*edge[None,:,:]).reshape(-1,2))
        weights.append((2*cross*(wt*t**3)[:,None]*wz[None,:]).ravel())
    pts=np.vstack(points);w=np.concatenate(weights)
    assert abs(sum(w)/(e.detJ/2)-1)<1e-12
    # Independent radial integral on each subtriangle: squared-map and ordinary
    # Duffy with Gauss-Jacobi (t^4) integrate |grad u|^2 with different nodes.
    from scipy.special import roots_jacobi
    tj,wj=roots_jacobi(order,0,4);tj=(tj+1)/2;wj=wj/32
    exact=0.
    for a,b in fem.EDGE_PAIRS:
        aa=e.xy[a]-center;bb=e.xy[b]-center;cross=abs(float(np.linalg.det(np.stack((aa,bb)))))
        edge=aa[None,:]+z[:,None]*(bb-aa)
        exact+=6.25*cross*sum(wj)*sum(wz*np.linalg.norm(edge,axis=1)**3)
    r=np.linalg.norm(pts-center,axis=1)
    observed=float(np.sum(w*6.25*r**3))
    assert abs(observed/exact-1)<2e-12,(observed,exact)
    return pts,w

def volume(mesh,center,case,space,boost):
    old=fem.ray_edge_radii
    try:
        fem.ray_edge_radii=lambda m,c,R,d:solver.radial.radii(m,c,R,d,15)
        groups,area=fem.volume_quadrature_groups(mesh,center,case['radius'],case['p'],space['active'],boost)
    finally:fem.ray_edge_radii=old
    eid=mesh.locate(center);assert eid in space['interior'],'Central rule requires a complete physical triangle'
    xy,w=central_rule(mesh.elements[eid],center,case['p'],boost)
    groups=[g for g in groups if g[0]!=eid]+[(eid,xy,w)];area[eid]=sum(w)
    return groups,area

def solve_refined(A,b):
    d=1/np.sqrt(np.diag(A));scaled=(A*d[:,None])*d
    lu,piv=la.lu_factor(scaled,check_finite=False);ld=np.longdouble
    x=(d*la.lu_solve((lu,piv),d*b,check_finite=False)).astype(ld)
    aa=A.astype(ld);bb=b.astype(ld);history=[]
    for _ in range(6):
        residual=bb-aa@x
        dx=d*la.lu_solve((lu,piv),np.asarray(d*residual,dtype=float),check_finite=False)
        x+=dx.astype(ld);history.append(float(np.max(abs(dx))))
    residual=bb-aa@x
    backward=float(np.max(abs(residual))/(np.max(np.sum(abs(aa),axis=1))*np.max(abs(x))+np.max(abs(bb))))
    assert backward<1e-15 and np.all(np.isfinite(x))
    return x,dict(backward_error=backward,update_history=history,factorization_reused=True,correction_dtype='float64',iterate_dtype='longdouble')

def sample(mesh,space,basis,center,case,coeffs,eid,xy,subtract):
    v,g=basis.physical(mesh.elements[eid],xy);cc=np.asarray(coeffs)[:,space['dofs'][eid]]
    val=cc@v.astype(np.longdouble);grad=np.einsum('ki,iaq->kaq',cc,g.astype(np.longdouble))
    if subtract:
        u,du,_=solver.data(xy.astype(np.longdouble),case,center);val-=u;grad-=du.T
    return val,grad

def norms(mesh,space,basis,center,case,patches,coeffs,boost,subtract=True):
    ld=np.longdouble;totals=np.zeros((len(coeffs),4),dtype=ld)
    vol,_=volume(mesh,center,case,space,boost);bd=fem.boundary_quadrature_groups(mesh,center,.62,case['p'],space['active'],boost)
    for phase,groups in [('volume',vol),('boundary',bd)]:
        batches={eid:[] for eid in space['active']}
        for g in groups:batches[g[0]].append((g[1],g[-1]))
        for eid,groups_ in batches.items():
            if not groups_:continue
            xy=np.vstack([g[0] for g in groups_]);w=np.concatenate([g[1] for g in groups_]).astype(ld)
            for offset in range(0,len(w),2048):
                val,grad=sample(mesh,space,basis,center,case,coeffs,eid,xy[offset:offset+2048],subtract);ww=w[offset:offset+2048]
                if phase=='volume':totals[:,0]+=(val*val)@ww;totals[:,1]+=np.sum(grad*grad,axis=1)@ww
                else:totals[:,2]+=(val*val)@ww
    cache={};monomials=solver.indices(case['p']);r,s,w=fem.triangle_quadrature(case['p']+3+boost)
    central=mesh.locate(center);maxorth=0.
    for patch in patches:
        vertices=np.vstack([mesh.elements[eid].xy for eid in patch]);pc=(vertices.min(0)+vertices.max(0))/2;scale=max(float(np.max(vertices.max(0)-vertices.min(0)))/2,mesh.h/2)
        Gs=[];Fs=[]
        for eid in patch:
            if eid not in cache:
                e=mesh.elements[eid]
                if eid==central:xy,ww=central_rule(e,center,case['p'],boost)
                else:xy=e.xy[0]+np.column_stack((r,s))@e.J.T;ww=w*e.detJ
                _,g=sample(mesh,space,basis,center,case,coeffs,eid,xy,subtract);cache[eid]=(xy,np.sqrt(ww),g)
            xy,sw,g=cache[eid];_,qg=fem.eval_patch_polynomials(monomials,xy,pc,scale)
            for axis in (0,1):Gs.append((qg[1:,axis]*sw).T);Fs.append(g[:,axis]*sw)
        G=np.vstack(Gs);F=np.concatenate(Fs,axis=1);Q,_=la.qr(G,mode='economic',check_finite=False)
        maxorth=max(maxorth,float(np.max(abs(Q.T@Q-np.eye(Q.shape[1])))));qq=Q.astype(ld)
        residual=F-(F@qq)@qq.T;totals[:,3]+=np.sum(residual*residual,axis=1)
    assert maxorth<1e-12
    answer=[]
    for total in totals:
        l2,gr,bd,st=total;st*=case['gamma1'];bd*=case['p']**2/mesh.h
        answer.append(dict(L2=float(np.sqrt(l2)),gradient=float(np.sqrt(gr)),boundary=float(np.sqrt(bd)),stabilization=float(np.sqrt(st)),
                           energy=float(np.sqrt(gr+bd+st)),physical=float(np.sqrt(gr+bd))))
    return answer

def main(run):
    records=[]
    parameters=json.loads((run/'parameters.json').read_text())
    degrees=parameters.get('degrees',[10,12,14,16])
    parent=Path(parameters['resume_parent']) if parameters.get('resume_parent') else None
    for p in degrees:
        for disk in parameters.get('disks',['fixed_A','fixed_B']):
            if parent is not None and (parent/'output'/f'p{p}-{disk}'/'result.json').exists():
                previous=json.loads((parent/'output'/f'p{p}-{disk}'/'result.json').read_text())
                records.append(dict(previous,resumed_from=str(parent)))
                dump(run/'output/radial-results.json',records)
                continue
            start=time.perf_counter();out=run/'output'/f'p{p}-{disk}';out.mkdir()
            record=dict(p=p,disk=disk,N=4,status='execution_failed')
            try:
                case=wc.normalize_case(dict(N=4,p=p,path=disk,patch_layers=2,solution_kind='radial_power'))
                mesh,center,space,basis,dm=solver.setup(case);record['ndof']=len(dm);coeffs=[];operators=[]
                # Geometry and low-degree identity check independent of the solve.
                linear=np.zeros(len(dm))
                for key,i in dm.items():
                    if key[0]=='v':linear[i]=mesh.vertices[key[1],0]
                worst=0.
                for eid in space['active']:
                    e=mesh.elements[eid];xy=e.xy[0]+np.array([[.2,.3],[.45,.11]])@e.J.T
                    v,g=basis.physical(e,xy);c=linear[space['dofs'][eid]]
                    worst=max(worst,float(np.max(abs(c@v-xy[:,0]))),float(np.max(abs(np.einsum('i,iaq->aq',c,g)-np.array([1.,0.])[:,None]))))
                assert worst<1e-12
                for boost in [16,24]:
                    groups,area=volume(mesh,center,case,space,boost);original=fem.volume_quadrature_groups
                    try:
                        fem.volume_quadrature_groups=lambda *args,**kwargs:(groups,area)
                        A,E,b,parts,meta=solver.assemble(mesh,space,center,case,basis,boost)
                    finally:fem.volume_quadrature_groups=original
                    x,info=solve_refined(A,b);coeffs.append(x);operators.append(dict(boost=boost,refinement=info,assembly=meta['timings']))
                    np.savez_compressed(out/f'q{boost}.npz',A=A,rhs=b,coefficients=x)
                # Two independent integration rules, plus physical/stabilized norm
                # of the difference of the two assembled solution fields.
                measured=[norms(mesh,space,basis,center,case,meta['patches'],coeffs,boost) for boost in [24,32]]
                difference=norms(mesh,space,basis,center,case,meta['patches'],[coeffs[0]-coeffs[1]],32,False)[0]
                checks={};resolved={}
                for name in ['L2','gradient','energy']:
                    repeat=max(abs(measured[0][i][name]/measured[1][i][name]-1) for i in [0,1])
                    assembly=difference[name]/min(v[name] for v in measured[1]);checks[name]=dict(quadrature_relative_change=repeat,assembly_discrepancy_fraction=assembly)
                    resolved[name]=bool(repeat<=1e-4 and assembly<=.01)
                record.update(status='resolved' if all(resolved.values()) else 'accuracy_flag',case=case,operators=operators,
                    errors=measured[-1][-1],measurements=measured,difference=difference,checks=checks,resolved=resolved,
                    central_quadrature='squared radial Duffy map, checked against Gauss-Jacobi gradient integral',linear_reproduction=worst)
            except Exception as exc:
                record.update(exception=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
            record['seconds']=time.perf_counter()-start;dump(out/'result.json',record);records.append(record);dump(run/'output/radial-results.json',records)
            print(p,disk,record['status'],record['seconds'],flush=True)
    return 0 if all(r['status']=='resolved' for r in records) else 2

if __name__=='__main__':
    run=Path(sys.argv[1]).resolve();assert platform.node().split('.')[0]=='compmath1' and getpass.getuser()=='karl'
    start=time.time();code=1
    dump(run/'environment.json',dict(host=platform.node(),user=getpass.getuser(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,device='CPU',dtype='float64 assembly, longdouble coefficients and norm accumulation',seed=None,threads=os.environ.get('OPENBLAS_NUM_THREADS')))
    try:code=main(run)
    finally:dump(run/'completion.json',dict(exit_code=code,started=start,finished=time.time(),seconds=time.time()-start,job_id=os.environ.get('SLURM_JOB_ID')))
    sys.exit(code)
