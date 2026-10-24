"""Process-local retained-geometry classification without a fixed distance gap."""
from contextlib import contextmanager
from decimal import Decimal as D,localcontext
import run_2d_disk_study as fem
import decimal_retained_system as ds

def squared_distance_margin(vertices,center,radius):
    cross=lambda a,b:a[0]*b[1]-a[1]*b[0]
    edges=[[vertices[(i+1)%3][k]-vertices[i][k] for k in (0,1)] for i in range(3)]
    offsets=[[center[k]-vertices[i][k] for k in (0,1)] for i in range(3)]
    if all(cross(edge,v)>=0 for edge,v in zip(edges,offsets)):return radius*radius
    distances=[]
    for edge,v in zip(edges,offsets):
        t=max(D(0),min(D(1),sum((a*b for a,b in zip(v,edge)),D(0))/sum((a*a for a in edge),D(0))))
        distances.append(sum(((v[k]-t*edge[k])**2 for k in (0,1)),D(0)))
    return radius*radius-min(distances)

def classify(mesh,center,radius):
    with localcontext() as ctx:
        ctx.prec=80;c=list(map(ds.exact_decimal,center));R=ds.exact_decimal(radius);R2=R*R;active=[];interior=[];cut=[]
        for element in mesh.elements:
            vertices=[[ds.exact_decimal(v) for v in row] for row in element.xy]
            if squared_distance_margin(vertices,c,R)>0:
                active.append(element.eid)
                if all(sum(((v[k]-c[k])**2 for k in (0,1)),D(0))<R2 for v in vertices):interior.append(element.eid)
                else:cut.append(element.eid)
    return active,interior,cut

@contextmanager
def classification():
    previous=fem.classify_elements;fem.classify_elements=classify
    try:yield
    finally:fem.classify_elements=previous

def self_check():
    from types import SimpleNamespace
    import numpy as np
    e=SimpleNamespace(eid=0,xy=np.array([[0.,0.],[1.,0.],[0.,1.]]));mesh=SimpleNamespace(elements=[e]);before=fem.classify_elements
    assert classify(mesh,[-1.,0.],1.)==([],[],[])
    assert classify(mesh,[-1.+1e-14,0.],1.)==([0],[],[0])
    assert classify(mesh,[0.,0.],2.)==([0],[0],[])
    assert classify(mesh,[-2.,0.],1.)==([],[],[])
    with classification():assert fem.classify_elements is classify
    assert fem.classify_elements is before
    return dict(tangent_excluded=True,positive_sliver_retained=True,full_triangle=True,outside_excluded=True,context_restored=True)
