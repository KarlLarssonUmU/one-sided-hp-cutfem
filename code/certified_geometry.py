"""Bounded saved-geometry audit: exact rational sector exclusion, no PDE solves.

Circle/mesh-line events have rationally certified quadratic-root brackets.
Each true open boundary arc is covered by a slightly larger rational wedge.
All patch-boundary edges are clipped against that wedge using exact fractions.
The maximum radius of a clipped straight segment occurs at an endpoint.
If every boundary edge is outside the disk or below the retained inner radius,
continuity of an inward radial segment proves containment in the patch.
Mesh-line event points have boundary measure zero and are omitted from Gamma.
"""
from bisect import bisect_right
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import csv, hashlib, json, math, os, platform, resource, sys, time
import mpmath as mp
import numpy as np
import run_2d_disk_study as fem

read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
dump=lambda p,v:p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
cross=lambda a,b:a[0]*b[1]-a[1]*b[0]
dot=lambda a,b:a[0]*b[0]+a[1]*b[1]
sub=lambda a,b:(a[0]-b[0],a[1]-b[1])
add=lambda a,b:(a[0]+b[0],a[1]+b[1])
mul=lambda a,s:(a[0]*s,a[1]*s)


def rotate(v,k):
    x,y=v
    return ((x,y),(-y,x),(-x,-y),(y,-x))[k]


def direction(u,k):
    return rotate(((1-u*u)/(1+u*u),2*u/(1+u*u)),k)


def m(value):
    return mp.mpf(value.numerator)/value.denominator


def roots(A,B,C):
    """Isolate every real quadratic root; all acceptance decisions are exact."""
    if A==0:
        return [] if B==0 else [(-C/B,-C/B)]
    D=B*B-4*A*C
    if D<0:return []
    n=math.isqrt(D.numerator);d=math.isqrt(D.denominator)
    if n*n==D.numerator and d*d==D.denominator:
        s=F(n,d)
        return sorted(set(((v,v) for v in ((-B-s)/(2*A),(-B+s)/(2*A)))))
    out=[]
    for sign in (-1,1):
        x=(-m(B)+sign*mp.sqrt(m(D)))/(2*m(A))
        near=F(mp.nstr(x,75));eps=F(1,10**55)
        lo,hi=near-eps,near+eps
        poly=lambda z:(A*z+B)*z+C
        assert poly(lo)*poly(hi)<0,('root not bracketed',A,B,C)
        assert (2*A*lo+B)*(2*A*hi+B)>0,('root not isolated',A,B,C)
        out.append((lo,hi))
    return sorted(out)


def sqrt_upper(q):
    assert q>=0
    scale=10**35
    z=math.isqrt(q.numerator*scale*scale//q.denominator)
    upper=F(z+1,scale)
    assert upper*upper>=q
    return upper


def owner(point,vertices,N):
    grid=[vertices[i][0] for i in range(N+1)]
    i=bisect_right(grid,point[0])-1;j=bisect_right(grid,point[1])-1
    assert 0<=i<N and 0<=j<N
    bl=vertices[j*(N+1)+i];tr=vertices[(j+1)*(N+1)+i+1]
    tri=0 if cross(sub(tr,bl),sub(point,bl))<=0 else 1
    return 2*(j*N+i)+tri


def partition(mesh,center,R):
    vertices=[tuple(map(F,v)) for v in mesh.vertices]
    # Full supporting lines over-partition the circle; this cannot omit an event.
    lines=set()
    for element in mesh.elements:
        for i,j in ((0,1),(1,2),(2,0)):
            a=vertices[element.vids[i]];b=vertices[element.vids[j]];d=sub(b,a)
            values=(-d[1],d[0],d[1]*a[0]-d[0]*a[1])
            scale=next(v for v in values if v)
            lines.add(tuple(v/scale for v in values))
    arcs=[];root_count=0
    for quadrant in range(4):
        events={(F(0),F(0)),(F(1),F(1))}
        for a,b,c in lines:
            off=a*center[0]+b*center[1]+c
            ex=R*dot((a,b),rotate((F(1),F(0)),quadrant))
            ey=R*dot((a,b),rotate((F(0),F(1)),quadrant))
            for lo,hi in roots(off-ex,2*ey,off+ex):
                if hi<0 or lo>1:continue
                assert lo>=0 and hi<=1,('endpoint root ambiguity',lo,hi)
                events.add((lo,hi));root_count+=1
        events=sorted(events)
        for left,right in zip(events,events[1:]):
            assert left[1]<right[0],('overlapping distinct root intervals',left,right)
            u=(left[1]+right[0])/2
            seed=owner(add(center,mul(direction(u,quadrant),R)),vertices,mesh.N)
            # Every supporting-line sign is constant between isolated events.
            # The bracket-expanded wedge contains the complete true arc.
            arcs.append(dict(quadrant=quadrant,left=left,right=right,seed=seed))
    return vertices,arcs,root_count,len(lines)


def boundary_edges(mesh,patch):
    counts=Counter(tuple(sorted((mesh.elements[e].vids[i],mesh.elements[e].vids[j])))
        for e in patch for i,j in ((0,1),(1,2),(2,0)))
    assert all(v in (1,2) for v in counts.values())
    return [edge for edge,count in counts.items() if count==1]


def clip_edge(a,b,left,right):
    lo,hi=F(0),F(1)
    for f0,f1 in ((cross(left,a),cross(left,b)),(cross(a,right),cross(b,right))):
        slope=f1-f0
        if slope==0:
            if f0<0:return None
        elif slope>0:lo=max(lo,-f0/slope)
        else:hi=min(hi,-f0/slope)
        if lo>hi:return None
    delta=sub(b,a)
    stationary=max(lo,min(hi,-dot(a,delta)/dot(delta,delta)))
    point=lambda s:add(a,mul(delta,s))
    q=lambda s:dot(point(s),point(s))
    return q(stationary),max(q(lo),q(hi))


def arc_depth(vertices,edges,center,R,arc):
    left=direction(arc['left'][0],arc['quadrant'])
    right=direction(arc['right'][1],arc['quadrant'])
    assert cross(left,right)>0
    largest=F(0)
    for i,j in edges:
        clipped=clip_edge(sub(vertices[i],center),sub(vertices[j],center),left,right)
        if clipped is None:continue
        qmin,qmax=clipped
        if qmin>=R*R:continue  # No intersection with the open physical disk.
        if qmax>=R*R:return F(0)
        largest=max(largest,qmax)
    return max(F(0),min(R/2,R-sqrt_upper(largest)))


def check_geometry(entry, read_record=None):
    tick=time.perf_counter();N=entry['N'];R=F(entry['radius']);center=tuple(map(F,entry['center']))
    mesh=fem.build_mesh(N);h=F(mesh.h);vertices,arcs,root_count,line_count=partition(mesh,center,R)
    raw=read_record(entry['runs'][0]) if read_record else read(Path(entry['runs'][0])/'results.json');active=raw['geometry']['active_elements']
    seeds=sorted({a['seed'] for a in arcs});assert set(seeds)<=set(active)
    layers=entry['layers'];patches=fem.build_patches(mesh,active,seeds,layers)
    assert patches==raw['assembly'][-1]['patches'],('saved patch mismatch',entry['id'])
    for run in entry['runs']:
        other=read_record(run) if read_record else read(Path(run)/'results.json')
        assert other['geometry']['active_elements']==active
        assert all(a['patches']==patches for a in other['assembly'])
    records=[]
    for depth in ([2,3,4] if layers==2 else [layers]):
        trial=fem.build_patches(mesh,active,seeds,depth)
        byseed=dict(zip(seeds,trial));edges={s:boundary_edges(mesh,p) for s,p in byseed.items()}
        bounds=[];witness=None;sample_checks=0
        for a in arcs:
            bound=arc_depth(vertices,edges[a['seed']],center,R,a);bounds.append(bound)
            for fraction in (F(1,1000),F(1,2),F(999,1000)):
                u=a['left'][1]+fraction*(a['right'][0]-a['left'][1]);n=direction(u,a['quadrant'])
                assert owner(add(center,mul(n,R)),vertices,N)==a['seed']
                for t in (bound/2,bound*F(999,1000)):
                    if t>0:
                        assert owner(add(center,mul(n,R-t)),vertices,N) in byseed[a['seed']],('certificate/sample conflict',entry['id'])
                        sample_checks+=1
                t=h/F(32)
                x=add(center,mul(n,R-t));eid=owner(x,vertices,N)
                if witness is None and eid not in byseed[a['seed']]:
                    witness=dict(seed=a['seed'],point=[str(v) for v in x],boundary_point=[str(v) for v in add(center,mul(n,R))],
                        t_over_h=str(t/h),outside_element=eid,quadrant=a['quadrant'],u=str(u))
        minimum=min(bounds);passes=minimum>=h/F(16)
        if passes:assert witness is None
        overlap=Counter(e for patch in trial for e in patch)
        row=dict(id=entry['id'],N=N,center=entry['center'],layers=depth,used_for_saved_PDE=(depth==layers),
            cases=entry['case_ids'],patches=len(trial),arcs=len(arcs),supporting_lines=line_count,root_brackets=root_count,
            minimum_depth_over_h_lower_bound=float(minimum/h),exact_depth_over_h_lower_bound=str(minimum/h),
            tube_h_over_16_certified=passes,jacobian_lower_bound=.5,maximum_patch_elements=max(map(len,trial)),
            maximum_overlap=max(overlap.values()),independent_ray_samples=sample_checks,h_over_16_counterexample=witness)
        records.append(row)
    print(json.dumps(dict(geometry=entry['id'],seconds=time.perf_counter()-tick,results=[{k:r[k] for k in ('layers','minimum_depth_over_h_lower_bound','tube_h_over_16_certified')} for r in records])),flush=True)
    return records


def continuation():
    records=[]
    for p in range(1,15):
        values=[]
        for digits in (60,80):
            with mp.workdps(digits):
                x,w=mp.gauss_quadrature(p+1,'legendre')
                V=mp.matrix([[mp.sqrt(mp.mpf(2*j+1)/2)*mp.legendre(j,2+x[k]) for j in range(p+1)] for k in range(p+1)])
                G=V.T*mp.diag(list(w))*V;cp=mp.sqrt(mp.eigsy(G,eigvals_only=True)[p])
                A=mp.acosh(3);D=1-mp.mpf(1)/(4*p*p-1)
                I=1+((mp.cosh((2*p+1)*A)-1)/(2*p+1)-(mp.cosh((2*p-1)*A)-1)/(2*p-1))/4
                trial=mp.sqrt(I/D)
                # An independent direct quadrature checks the closed formula.
                direct=mp.sqrt(mp.quad(lambda z:mp.chebyt(p,z)**2,[1,2,3])/mp.quad(lambda z:mp.chebyt(p,z)**2,[-1,0,1]))
                assert abs(trial/direct-1)<mp.mpf('1e-50')
                assert trial<=cp*(1+mp.mpf('1e-50'))
                assert cp<=mp.sqrt(sum(G[j,j] for j in range(p+1)))*(1+mp.mpf('1e-50'))
                values.append((cp,trial))
        assert abs(values[0][0]/values[1][0]-1)<mp.mpf('1e-50')
        cp,trial=values[-1]
        records.append(dict(p=p,continuation=float(cp),chebyshev_lower=float(trial),monomial_lower=math.sqrt(2**(2*p+1)-1),
            endpoint_constant=p+1,chebyshev_scaled=float(trial*mp.sqrt(p)/(3+mp.sqrt(8))**p),
            computed_scaled=float(cp*mp.sqrt(p)/(3+mp.sqrt(8))**p)))
    return records


def table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

