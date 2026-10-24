"""Isolate radial-breakpoint rounding; preserve the disk quadrature algorithm."""
from contextlib import contextmanager
from decimal import Decimal as D, localcontext
from types import SimpleNamespace
import numpy as np
import run_2d_disk_study as fem
import precision_eval_2d as pe
import local_integrals_2d as li
import polynomial_current_2d as pc

def radii(mesh,center,radius,direction,digits):
    # Exact copy of the frozen ray_edge_radii arithmetic, except digits.
    a=mesh.edges[:,0,:];e=mesh.edges[:,1,:]-a;q=a-center[None,:]
    den=direction[0]*e[:,1]-direction[1]*e[:,0]
    good=np.abs(den)>1.0e-13
    rho=np.full(den.shape,np.nan);t=np.full(den.shape,np.nan)
    rho[good]=(q[good,0]*e[good,1]-q[good,1]*e[good,0])/den[good]
    t[good]=(q[good,0]*direction[1]-q[good,1]*direction[0])/den[good]
    mask=good&(rho>1.0e-12)&(rho<radius-1.0e-12)
    mask&=(t>-1.0e-11)&(t<1.0+1.0e-11)
    vals=np.concatenate(([0.0],rho[mask],[radius]))
    return np.sort(np.unique(np.round(vals,digits)))

def self_check():
    xs=np.array([.123456789012345,.567890123456789])
    mesh=SimpleNamespace(edges=np.array([[[x,-1.],[x,1.]] for x in xs]))
    center=np.array([.071,-.043]);direction=np.array([.8,.6]);radius=.9
    with localcontext() as ctx:
        ctx.prec=80
        exact=[D(0)]+[(D.from_float(float(x))-D.from_float(float(center[0])))/D.from_float(float(direction[0])) for x in xs]+[D.from_float(radius)]
        errors={}
        for digits in (13,15):
            observed=radii(mesh,center,radius,direction,digits)
            assert len(observed)==len(exact)
            errors[str(digits)]=float(max(abs(D.from_float(float(x))-y) for x,y in zip(observed,exact)))
        assert errors['13']<=5.1e-14 and errors['15']<=6e-16
    assert np.array_equal(radii(mesh,center,radius,direction,13),fem.ray_edge_radii(mesh,center,radius,direction))
    return dict(decimal_digits=80,absolute_breakpoint_errors=errors)

@contextmanager
def precision(digits,active,trace):
    """Temporarily replace only the radial breakpoint callback, with trace."""
    previous=fem.ray_edge_radii;active=set(active)
    def callback(mesh,center,radius,direction):
        legacy=previous(mesh,center,radius,direction)
        assert np.array_equal(legacy,radii(mesh,center,radius,direction,13))
        candidate=radii(mesh,center,radius,direction,digits)
        def labels(values):
            return [mesh.locate(center+.5*(a+b)*direction) for a,b in zip(values[:-1],values[1:]) if b-a>=2e-12 and mesh.locate(center+.5*(a+b)*direction) in active]
        old_ids,new_ids=labels(legacy),labels(candidate)
        shift=max(float(np.max(np.min(np.abs(legacy[:,None]-candidate),axis=1))),float(np.max(np.min(np.abs(candidate[:,None]-legacy),axis=1))))
        trace['rays']+=1
        trace['changed_breakpoint_count_rays']+=int(len(legacy)!=len(candidate))
        trace['changed_element_sequence_rays']+=int(old_ids!=new_ids)
        trace['max_breakpoint_shift']=max(trace['max_breakpoint_shift'],shift)
        trace['legacy_used_intervals']+=len(old_ids);trace['candidate_used_intervals']+=len(new_ids)
        for name,values in [('legacy',legacy),('candidate',candidate)]:
            gaps=np.diff(values);small=gaps[gaps<2e-12]
            trace[name+'_skipped_intervals']+=len(small)
            trace[name+'_skipped_radial_length']+=float(np.sum(small))
        if digits!=13:trace['records'].append(dict(direction=direction.tolist(),legacy=legacy.tolist(),candidate=candidate.tolist(),legacy_elements=old_ids,candidate_elements=new_ids))
        return candidate
    try:
        fem.ray_edge_radii=callback
        yield
    finally:fem.ray_edge_radii=previous

def new_trace():
    return dict(rays=0,changed_breakpoint_count_rays=0,changed_element_sequence_rays=0,max_breakpoint_shift=0.,legacy_used_intervals=0,candidate_used_intervals=0,
        legacy_skipped_intervals=0,candidate_skipped_intervals=0,legacy_skipped_radial_length=0.,candidate_skipped_radial_length=0.,records=[])

def assemble(problem):
    """Original mapping, longdouble basis/data and unchanged integral kernels."""
    n=1+max(int(np.max(v)) for v in problem['element_dofs'].values())
    blocks={k:np.zeros((n,n) if k in ('K','M','Mb','Cn') else n,dtype=np.longdouble) for k in li.COMPONENTS}
    before=li.groups_digest(problem)
    for phase,key in [('volume','volume_groups'),('boundary','boundary_groups')]:
        for group in problem[key]:
            eid,points=group[:2];idx=problem['element_dofs'][eid];e=problem['mesh'].elements[eid]
            r,s=fem.reference_coordinates(e,points)
            v,gref=pe.horner(problem['element_basis'][eid],r,s,np.longdouble)
            g=np.einsum('ab,ibq->iaq',e.invJT.astype(np.longdouble),gref)
            u,_,force=pc.data(points)
            local=li.volume_kernel(v,g,group[2],force) if phase=='volume' else li.boundary_kernel(v,g,group[2],group[3],u)
            for key,value in local.items():blocks[key][np.ix_(idx,idx) if value.ndim==2 else idx]+=value
    assert before==li.groups_digest(problem)
    return blocks,before
