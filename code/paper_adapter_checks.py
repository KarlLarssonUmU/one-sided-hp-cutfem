"""Small checks for low-degree, geometry, manufactured-data and norm semantics."""
from types import SimpleNamespace
import numpy as np
import workstation_case as wc
import paper_solver_2d as solver
import paper_energy as energy


def check():
    records=[]
    for p in (1,2,4):
        case=wc.normalize_case(dict(N=2,p=p,path='fixed_A',patch_layers=2))
        mesh,center,space,basis,dofs=solver.setup(case)
        observed=solver.basis_checks(mesh,space,basis,dofs)
        assert observed['linear_reproduction_error']<5e-14 and observed['Decimal_basis_scaled_error']<5e-12
        records.append(dict(p=p,**observed))
    triangle=SimpleNamespace(eid=0,xy=np.array([[0.,0.],[1.,0.],[0.,1.]]));mesh=SimpleNamespace(elements=[triangle])
    assert solver.classify(mesh,np.array([-1.,0.]),1.)==([],[],[])
    assert solver.classify(mesh,np.array([-1.+1e-14,0.]),1.)==([0],[],[0])
    assert solver.classify(mesh,np.array([0.,0.]),2.)==([0],[0],[])
    case=dict(solution_kind='radial_power',singularity_alpha=2.5)
    u,g,f=solver.data(np.array([[.3,.4],[0.,0.]]),case,np.zeros(2))
    assert abs(u[0]-.5**2.5)<1e-16 and np.max(abs(g[0]-2.5*.5**.5*np.array([.3,.4])))<1e-15
    assert abs(f[0]+6.25*.5**.5)<1e-15 and np.all(g[1]==0) and u[1]==f[1]==0
    z=energy.components(4.,9.,16.,25.,.1,3.)
    assert abs(z['energy_squared']-52.9)<1e-13 and z['l2']==5
    assert abs(sum(z['squared_shares'].values())-1)<1e-15
    return dict(low_degree_checks=records,strict_classifier_controls=True,radial_data_values=True,
                energy_component_weighting=True,L2_not_added_to_energy=True)
