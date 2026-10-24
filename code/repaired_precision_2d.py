"""Consistent affine arithmetic and retained raw-Gram controls on repaired bases."""
from decimal import Decimal as D,localcontext
from types import SimpleNamespace
import numpy as np
import run_2d_disk_study as fem
import precision_eval_2d as pe
import polynomial_current_2d as pc
import local_integrals_2d as li
import stabilization_arithmetic_2d as sa
import decimal_retained_system as ds

def geometry(element,points):
    d=np.longdouble;a,b,c,e=np.asarray(element.J,dtype=d).ravel();det=a*e-b*c
    assert det!=0
    delta=np.asarray(points,dtype=d)-np.asarray(element.xy[0],dtype=d)
    r=(e*delta[:,0]-b*delta[:,1])/det;s=(-c*delta[:,0]+a*delta[:,1])/det
    invT=np.array([[e,-c],[-b,a]],dtype=d)/det
    return r,s,invT

def point_check(element,points):
    with localcontext() as ctx:
        ctx.prec=80;J=[[ds.exact_decimal(x) for x in row] for row in element.J]
        def solve(rhs):
            m=[row[:] for row in J];v=list(rhs)
            if abs(m[1][0])>abs(m[0][0]):m.reverse();v.reverse()
            factor=m[1][0]/m[0][0];y=(v[1]-factor*v[0])/(m[1][1]-factor*m[0][1]);x=(v[0]-m[0][1]*y)/m[0][0]
            return [x,y]
        coords=[solve([ds.exact_decimal(point[k])-ds.exact_decimal(element.xy[0,k]) for k in range(2)]) for point in points]
        inverse_transpose=[solve([D(int(k==j)) for k in range(2)]) for j in range(2)]
        exact=np.asarray([[str(x) for x in row] for row in coords],dtype=np.longdouble).T
        exact_inv=np.asarray([[str(x) for x in row] for row in inverse_transpose],dtype=np.longdouble)
    r,s,inv=geometry(element,points)
    return dict(coordinates=float(np.max(np.abs(np.array([r,s])-exact)/np.maximum(1,np.abs(exact)))),inverse_transpose=float(np.max(np.abs(inv-exact_inv)/np.maximum(1,np.abs(exact_inv)))))

def self_check():
    e=SimpleNamespace(J=np.array([[.37,.11],[-.09,.29]]),xy=np.array([[.17,-.23]]))
    points=np.array([[.31,-.17],[.0,.4],[-.2,.1]])
    check=point_check(e,points);assert max(check.values())<=5e-18
    # An exact null vector whose Gram identity is lost by binary64 storage.
    R=np.array([[2.**27,1.,-2.**27-1.]])
    raw=sa.gram_extended(R);rounded=np.asarray(raw,dtype=float)
    with localcontext() as ctx:
        ctx.prec=80
        action=lambda a:[sum((ds.exact_decimal(x) for x in row),D(0)) for row in a]
        ext=action(raw);binary=action(rounded)
    assert max(map(abs,ext))==0 and max(map(abs,binary))>0
    return dict(Decimal80_geometry_scaled_errors=check,exact_null_vector_retained_in_longdouble=True,binary64_null_action_max=str(max(map(abs,binary))),longdouble_mantissa_bits=np.finfo(np.longdouble).nmant+1)

def assemble_geometry(problem):
    n=1+max(int(np.max(x)) for x in problem['element_dofs'].values())
    blocks={k:np.zeros((n,n) if k in ('K','M','Mb','Cn') else n,dtype=np.longdouble) for k in li.COMPONENTS}
    before=li.groups_digest(problem);checks=dict(coordinates=0.,inverse_transpose=0.);checked=set()
    for phase,key in [('volume','volume_groups'),('boundary','boundary_groups')]:
        for group in problem[key]:
            eid,points=group[:2];e=problem['mesh'].elements[eid];idx=problem['element_dofs'][eid]
            if (phase,eid) not in checked:
                points_check=points[np.unique(np.linspace(0,len(points)-1,min(3,len(points)),dtype=int))]
                for name,value in point_check(e,points_check).items():checks[name]=max(checks[name],value)
                checked.add((phase,eid))
            r,s,inv=geometry(e,points);v,gref=pe.horner(problem['element_basis'][eid],r,s,np.longdouble)
            g=np.einsum('ab,ibq->iaq',inv,gref);u,_,force=pc.data(points)
            local=li.volume_kernel(v,g,group[2],force) if phase=='volume' else li.boundary_kernel(v,g,group[2],group[3],u)
            for name,value in local.items():blocks[name][np.ix_(idx,idx) if value.ndim==2 else idx]+=value
    assert before==li.groups_digest(problem)
    return blocks,dict(quadrature_sha256=before,Decimal80_geometry_scaled_errors=checks,checked_element_phases=len(checked))

def stabilization(mesh,p,patches,element_dofs,element_basis,ndof,quadrature_boost):
    matrices={name:np.zeros((ndof,ndof),dtype=np.longdouble if name=='rawLD' else float) for name in ('clean64','raw64','rawLD')}
    r,s,wref=fem.triangle_quadrature(max(6,p+3+quadrature_boost))
    indices=[(i,total-i) for total in range(p+1) for i in range(total+1)]
    cache={};diagnostics=[];patch_dofs=[]
    for patch in patches:
        dofs=np.unique(np.concatenate([element_dofs[eid] for eid in patch]));patch_dofs.append(dofs)
        local_pos={int(g):i for i,g in enumerate(dofs)}
        vertices=np.vstack([mesh.elements[eid].xy for eid in patch]);center=.5*(vertices.min(axis=0)+vertices.max(axis=0))
        scale=max(.5*float(np.max(vertices.max(axis=0)-vertices.min(axis=0))),.5*mesh.h)
        F=np.zeros((2*len(r)*len(patch),len(dofs)));G=np.zeros((len(F),len(indices)-1));offset=0
        for eid in patch:
            if eid not in cache:
                e=mesh.elements[eid];points=e.xy[0][None,:]+(e.J@np.vstack((r,s))).T
                _,grads=fem.physical_basis(e,element_basis[eid],points);cache[eid]=(points,grads,np.sqrt(e.detJ*wref))
            points,grads,sqrtw=cache[eid];_,qgrads=fem.eval_patch_polynomials(indices,points,center,scale)
            loc=np.asarray([local_pos[int(g)] for g in element_dofs[eid]])
            for axis in (0,1):
                end=offset+len(r);F[offset:end,loc]=(grads[:,axis,:]*sqrtw).T;G[offset:end,:]=(qgrads[1:,axis,:]*sqrtw).T;offset=end
        Q,R=np.linalg.qr(G,mode='reduced');residual=F-Q@(Q.T@F)
        rawLD=sa.gram_extended(residual);raw64=np.asarray(rawLD,dtype=float);diag={};clean=fem.psd_cleanup(raw64,diag)
        ix=np.ix_(dofs,dofs)
        for name,value in [('clean64',clean),('raw64',raw64),('rawLD',rawLD)]:matrices[name][ix]+=value
        diagnostics.append(dict(elements=len(patch),dofs=len(dofs),sample_qr_orthogonality=float(np.linalg.norm(Q.T@Q-np.eye(Q.shape[1]),2)),gradient_R_condition=float(np.linalg.cond(R)),s1_trace=float(np.trace(clean)),**{'s1_'+k:v for k,v in diag.items()}))
    return matrices,patch_dofs,diagnostics
