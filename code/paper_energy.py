"""Independent energy-error components with local and global evidence.

The sum of weighted squared components is the squared energy norm. The
Laplacian bilinear form contributes the volume gradient error. Stabilization
is measured on the error's patch-projection residual, including the extension.
"""
from decimal import Decimal as D,localcontext
import hashlib,time
import numpy as np
import run_2d_disk_study as fem
import precise_exponential_decimal as exponential
import decimal_coefficient_identity as identity
import precision_eval_2d as pe
import repaired_precision_2d as geometry
import paper_solver_2d as solver


class Fields:
    def __init__(self,mesh,space,center,case,records,candidates,basis,gate):
        self.mesh,self.space,self.center,self.case=mesh,space,center,case
        self.records=records;self.labels=[r['label'] for r in records]
        self.online=list(candidates);self.candidates=candidates;self.basis=basis
        self.reference_index=self.labels.index('reference_error') if 'reference_error' in self.labels else None
        self.labels += ['online_'+key+'_error' for key in self.online]
        if self.reference_index is not None:self.labels += ['online_'+key+'_reference_difference' for key in self.online]
        self.polys={};self.subtract=np.array([r['subtract_exact'] for r in records],dtype=bool)
        for eid in space['active']:
            e=mesh.elements[eid];local=[np.asarray(r['coefficients'],dtype=object)[space['dofs'][eid]] for r in records]
            if case['solution_kind']=='exponential':
                expanded,(z0,E)=exponential.expansion(e,space['basis'][eid],local,space['p'])
            else:expanded=[None]*len(records);z0=E=D(0)
            with localcontext() as ctx:
                ctx.prec=80
                for k,row in enumerate(records):
                    if not row['subtract_exact'] or case['solution_kind']!='exponential':
                        poly={(0,0):D(0)}
                        for c,b in zip(local[k],space['basis'][eid]):
                            for ij,v in b.items():poly[ij]=poly.get(ij,D(0))+identity.exact_value(c)*identity.decimal_value(v)
                        expanded[k]=poly
                self.polys[eid]=([identity.rounded_polynomial(v) for v in expanded],np.longdouble(str(z0)),np.longdouble(str(E)))
            # Direct independent Decimal value/gradient check for every field.
            point=e.xy[0]+e.J@np.array([.2,.3]);observed=self.stable(eid,point[None,:])
            with localcontext() as ctx:
                ctx.prec=100;r,s,inv=identity.decimal_geometry(e,point)
                if case['solution_kind']=='exponential':
                    u=(identity.exact_value(point[0])+identity.exact_value(point[1])).exp();du=[u,u]
                else:
                    delta=[identity.exact_value(point[k])-identity.exact_value(center[k]) for k in (0,1)]
                    rr=sum(v*v for v in delta);alpha=D(str(case['singularity_alpha']))
                    u=rr**(alpha/2);du=[alpha*rr**(alpha/2-1)*v for v in delta] if rr else [D(0),D(0)]
                for k,row in enumerate(records):
                    value=dr=ds=D(0)
                    for c,poly in zip(local[k],space['basis'][eid]):
                        vv,rr,ss=identity.evaluate_decimal({ij:identity.decimal_value(v) for ij,v in poly.items()},r,s)
                        c=identity.exact_value(c);value+=c*vv;dr+=c*rr;ds+=c*ss
                    direct=[value,*[v[0]*dr+v[1]*ds for v in inv]]
                    if row['subtract_exact']:direct=[v-w for v,w in zip(direct,[u,*du])]
                    truth=np.array([str(v) for v in direct],dtype=np.longdouble)
                    got=np.array([observed[0][k,0],*observed[1][k,:,0]])
                    gate(f'e{eid}_{row["label"]}_independent_point',np.max(abs(got-truth))/max(np.max(abs(truth)),1e-300),1e-9)

    def stable(self,eid,points):
        e=self.mesh.elements[eid];polys,z0,E=self.polys[eid]
        r,s,inv=geometry.geometry(e,points);v,gref=pe.horner(polys,r,s,np.longdouble)
        g=np.einsum('ab,ibq->iaq',inv,gref)
        if self.case['solution_kind']=='exponential':
            t=points[:,0].astype(np.longdouble)+points[:,1].astype(np.longdouble)-z0
            v[self.subtract]-=E*exponential.tail(t,self.space['p']+1)
            g[self.subtract]-=E*exponential.tail(t,self.space['p'])[None,None,:]
        else:
            u,du,_=solver.data(points.astype(np.longdouble),self.case,self.center)
            v[self.subtract]-=u;g[self.subtract]-=du.T[None,:,:]
        return v,g

    def sample(self,eid,points):
        v,g=self.stable(eid,points);e=self.mesh.elements[eid]
        bv,bg=self.basis.physical(e,points)
        cc=np.array([self.candidates[k] for k in self.online])[:,self.space['dofs'][eid]]
        ov=(cc@bv).astype(np.longdouble);og=np.einsum('ki,iaq->kaq',cc,bg).astype(np.longdouble)
        u,du,_=solver.data(points.astype(np.longdouble),self.case,self.center)
        ov-=u;og-=du.T[None,:,:]
        values=[v,ov];gradients=[g,og]
        if self.reference_index is not None:
            values.append(ov-v[self.reference_index]);gradients.append(og-g[self.reference_index])
        return np.concatenate(values),np.concatenate(gradients)


def components(gradient,stabilization_raw,boundary_raw,l2,gamma,alpha):
    parts=dict(gradient=float(gradient),stabilization=float(gamma*stabilization_raw),boundary=float(alpha*boundary_raw))
    total=sum(parts.values())
    assert all(np.isfinite(x) and x>=0 for x in [*parts.values(),total,float(l2)])
    return dict(energy_squared=total,energy=float(np.sqrt(total)),squared_components=parts,
        component_norms={k:float(np.sqrt(v)) for k,v in parts.items()},
        squared_shares={k:v/total if total else 0. for k,v in parts.items()},
        raw_integrals=dict(gradient_squared=float(gradient),stabilization_squared=float(stabilization_raw),boundary_l2_squared=float(boundary_raw)),
        weights=dict(gradient=1.,stabilization=float(gamma),boundary=float(alpha)),
        l2_squared=float(l2),l2=float(np.sqrt(l2)))


def measure(mesh,space,center,case,patches,fields,boost):
    tick=time.perf_counter();nf=len(fields.labels);ld=np.longdouble
    totals=np.zeros((nf,4),dtype=ld) # L2, gradient, raw boundary, raw stabilization
    local=[];digests={};counts={};phase_seconds={};quadrature={}
    for phase in ('volume','boundary'):
        start=time.perf_counter()
        if phase=='volume':groups,area=fem.volume_quadrature_groups(mesh,center,case['radius'],space['p'],space['active'],boost)
        else:groups=fem.boundary_quadrature_groups(mesh,center,case['radius'],space['p'],space['active'],boost)
        digest=hashlib.sha256();batches={eid:[] for eid in space['active']};count=0;measure_sum=ld(0)
        for group in groups:
            eid=group[0];digest.update(str(eid).encode()+b':')
            for arr in group[1:]:digest.update(arr.tobytes())
            batches[eid].append((group[1],group[-1]));count+=len(group[-1]);measure_sum+=np.sum(group[-1].astype(ld))
        del groups
        for eid,entries in batches.items():
            if not entries:continue
            points=np.concatenate([a for a,b in entries]);weights=np.concatenate([b for a,b in entries]);sums=np.zeros((nf,2),dtype=ld)
            assert np.all(weights>0)
            for offset in range(0,len(points),4096):
                xy=points[offset:offset+4096];w=weights[offset:offset+4096].astype(ld)
                v,g=fields.sample(eid,xy);sums[:,0]+=(v*v)@w
                if phase=='volume':sums[:,1]+=np.sum(g*g,axis=1)@w
            if phase=='volume':totals[:,:2]+=sums
            else:totals[:,2]+=sums[:,0]
            for k,label in enumerate(fields.labels):
                local.append(dict(region=phase,element=eid,field=label,value_squared=float(sums[k,0]),
                                  gradient_squared=float(sums[k,1]) if phase=='volume' else None))
        digests[phase]=digest.hexdigest();counts[phase]=count;phase_seconds[phase]=time.perf_counter()-start
        quadrature[phase+'_measure']=float(measure_sum)
    # Independent patch rule/order, including exact extension in grad(error).
    start=time.perf_counter();q=max(6,space['p']+3+boost);r,s,w=fem.triangle_quadrature(q)
    indices=solver.indices(space['p']);cache={};patch_rows=[];digest=hashlib.sha256();maxorth=0.;poly_residual=0.
    for index,patch in enumerate(patches):
        vertices=np.vstack([mesh.elements[eid].xy for eid in patch]);pc=(vertices.min(0)+vertices.max(0))/2
        scale=max(float(np.max(vertices.max(0)-vertices.min(0)))/2,mesh.h/2)
        G=np.zeros((2*len(r)*len(patch),len(indices)-1));F=np.zeros((nf,len(G)),dtype=ld);offset=0
        for eid in patch:
            if eid not in cache:
                e=mesh.elements[eid];points=e.xy[0]+np.column_stack((r,s))@e.J.T
                _,grad=fields.sample(eid,points);cache[eid]=(points,grad,np.sqrt(abs(np.linalg.det(e.J))*w))
            points,grad,sqrtw=cache[eid];_,qg=fem.eval_patch_polynomials(indices,points,pc,scale)
            for axis in (0,1):
                end=offset+len(r);F[:,offset:end]=grad[:,axis,:]*sqrtw.astype(ld)
                G[offset:end]=(qg[1:,axis]*sqrtw).T;offset=end
            digest.update(str(eid).encode()+b':');digest.update(points.tobytes());digest.update(sqrtw.tobytes())
        Q,_=np.linalg.qr(G,mode='reduced');maxorth=max(maxorth,float(np.max(abs(Q.T@Q-np.eye(Q.shape[1])))))
        qq=Q.astype(ld);residual=F-(F@qq)@qq.T
        raw=np.sum(residual*residual,axis=1);totals[:,3]+=raw
        # Reproduce a polynomial gradient in the actual sampled patch space.
        probe=G[:,0].astype(ld);pr=probe-qq@(qq.T@probe)
        poly_residual=max(poly_residual,float(np.linalg.norm(pr)/np.linalg.norm(probe)))
        for k,label in enumerate(fields.labels):
            patch_rows.append(dict(patch=index,elements=patch,field=label,raw_stabilization_squared=float(raw[k]),
                                   weighted_stabilization_squared=float(ld(case['gamma1'])*raw[k])))
    phase_seconds['patches']=time.perf_counter()-start;digests['patches']=digest.hexdigest()
    counts['patch_samples_with_multiplicity']=sum(len(patch)*len(r) for patch in patches)
    alpha=ld(space['p'])**2/ld(mesh.h);gamma=ld(case['gamma1'])
    result={label:components(totals[k,1],totals[k,3],totals[k,2],totals[k,0],gamma,alpha) for k,label in enumerate(fields.labels)}
    return dict(boost=boost,fields=result,local_elements=local,local_patches=patch_rows,
        quadrature_digests=digests,point_counts=counts,quadrature=quadrature,patch_gauss_order=q,
        patch_qr_orthogonality=maxorth,patch_polynomial_relative_residual=poly_residual,
        phase_seconds=phase_seconds,seconds=time.perf_counter()-tick,
        definition='gradient + gamma1*patch_projection_residual(error)^2 + (p^2/h)*boundary_error^2; components are squared',
        L2_is_separate=True,beta0_in_energy_weight=False)
