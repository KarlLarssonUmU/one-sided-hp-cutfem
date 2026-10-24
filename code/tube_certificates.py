"""Exact tube assignment via a finite arrangement of ray-triangle crossing events.

For fixed radii r0<R, the ordered open triangles met by a radial segment can
change only when an endpoint crosses a mesh edge or the segment passes a mesh
vertex in the annulus. Isolating all such angles partitions the circle into
open cells with constant traversed-triangle sets. An exact representative ray
then proves the whole angular cell. This is a continuum combinatorial argument,
not inference from densely sampled rays. Event angles have zero boundary measure.
"""
from pathlib import Path
from fractions import Fraction as F
from bisect import bisect_right
from collections import Counter
import getpass,json,os,platform,resource,sys,time
import certified_geometry as geo
import run_2d_disk_study as fem


def normalize(values):
    scale=next((x for x in values if x),None)
    return None if scale is None else tuple(x/scale for x in values)


def arrangement(vertices,triangles,c,R,r0):
    lines=set()
    for triangle in triangles:
        for i,j in [(0,1),(1,2),(2,0)]:
            a,b=vertices[triangle[i]],vertices[triangle[j]];d=geo.sub(b,a)
            lines.add(normalize((-d[1],d[0],d[1]*a[0]-d[0]*a[1])))
    # Only a vertex on the segment's annulus can change its crossing sequence.
    annulus_vertices=[i for i,v in enumerate(vertices) if r0*r0<=geo.dot(geo.sub(v,c),geo.sub(v,c))<=R*R]
    quadrants=[]
    for k in range(4):
        polynomials=set()
        for a,b,d in lines:
            off=a*c[0]+b*c[1]+d
            for radius in (R,r0):
                ex=radius*geo.dot((a,b),geo.rotate((F(1),F(0)),k))
                ey=radius*geo.dot((a,b),geo.rotate((F(0),F(1)),k))
                polynomials.add(normalize((off-ex,2*ey,off+ex)))
        for i in annulus_vertices:
            vx,vy=geo.sub(vertices[i],c);normal=(-vy,vx)
            ex=geo.dot(normal,geo.rotate((F(1),F(0)),k));ey=geo.dot(normal,geo.rotate((F(0),F(1)),k))
            polynomials.add(normalize((-ex,2*ey,ex)))
        polynomials.discard(None)
        events={(F(0),F(0)),(F(1),F(1))};proof=[]
        for A,B,C in sorted(polynomials):
            brackets=geo.roots(A,B,C)
            proof.append(dict(coefficients=list(map(str,(A,B,C))),roots=[[str(x),str(y)] for x,y in brackets]))
            for lo,hi in brackets:
                if hi<0 or lo>1:continue
                assert lo>=0 and hi<=1
                events.add((lo,hi))
        events=sorted(events)
        assert all(a[1]<b[0] for a,b in zip(events,events[1:])),('root intervals overlap',k)
        quadrants.append(dict(quadrant=k,events=events,polynomials=proof))
    return quadrants,annulus_vertices,len(lines)


def triangle_interval(start,end,triangle):
    """Exact intersection of a segment with a closed CCW triangle."""
    direction=geo.sub(end,start);lo,hi=F(0),F(1)
    for i in range(3):
        a,b=triangle[i],triangle[(i+1)%3];edge=geo.sub(b,a)
        value=geo.cross(edge,geo.sub(start,a));slope=geo.cross(edge,direction)
        if slope==0:
            if value<0:return None
        elif slope>0:lo=max(lo,-value/slope)
        else:hi=min(hi,-value/slope)
        if lo>hi:return None
    return lo,hi


def traversal(mesh,vertices,start,end):
    # With target h/16 each coordinate spans at most two grid cells. The
    # bounding rectangle still works unchanged for larger finite spans.
    grid=[vertices[i][0] for i in range(mesh.N+1)]
    cell=lambda x:bisect_right(grid,x)-1
    imin,imax=sorted([cell(start[0]),cell(end[0])]);jmin,jmax=sorted([cell(start[1]),cell(end[1])])
    assert 0<=imin<=imax<mesh.N and 0<=jmin<=jmax<mesh.N
    parts=[]
    for j in range(jmin,jmax+1):
        for i in range(imin,imax+1):
            for side in [0,1]:
                eid=2*(j*mesh.N+i)+side;triangle=[vertices[x] for x in mesh.elements[eid].vids]
                interval=triangle_interval(start,end,triangle)
                if interval is not None and interval[0]<interval[1]:parts.append((interval[0],interval[1],eid))
    parts.sort();reach=F(0)
    for lo,hi,eid in parts:
        assert lo<=reach,('gap',start,end,parts)
        reach=max(reach,hi)
    assert reach==1
    return parts


def check(entry,prior,read_record=None):
    tick=time.perf_counter();mesh=fem.build_mesh(entry['N']);vertices=[tuple(map(F,v)) for v in mesh.vertices]
    triangles=[list(e.vids) for e in mesh.elements];c=tuple(map(F,entry['center']));R=F(entry['radius']);h=F(mesh.h);r0=R-h/16
    assert 0<r0<R and h/16<R/2
    raw=read_record(entry['runs'][0]) if read_record is not None else geo.read(Path(entry['runs'][0])/'results.json');active=raw['geometry']['active_elements']
    patches={r['seed']:r['elements'] for r in prior['patches']};seeds=sorted(patches)
    assert list(patches.values())==raw['assembly'][-1]['patches']==fem.build_patches(mesh,active,seeds,entry['layers'])
    for path in entry['runs']:
        record=read_record(path) if read_record is not None else geo.read(Path(path)/'results.json');assert all(a['patches']==list(patches.values()) for a in record['assembly'])
    quadrants,annulus_vertices,line_count=arrangement(vertices,triangles,c,R,r0)
    assigned=[];positive=0;reassigned=0;failures=[];natural_fail=0
    patchsets={s:set(p) for s,p in patches.items()}
    for q in quadrants:
        events=q['events']
        for index,(left,right) in enumerate(zip(events,events[1:])):
            u=(left[1]+right[0])/2;n=geo.direction(u,q['quadrant'])
            start=geo.add(c,geo.mul(n,R));end=geo.add(c,geo.mul(n,r0))
            owner=geo.owner(start,vertices,mesh.N);assert owner in patches
            parts=traversal(mesh,vertices,start,end);crossed={e for lo,hi,e in parts}
            assert owner in crossed and crossed<=set(active)
            candidates=[s for s in seeds if owner in patchsets[s]]
            feasible=[s for s in candidates if crossed<=patchsets[s]]
            chosen=owner if owner in feasible else (min(feasible,key=lambda s:(len(patches[s]),s)) if feasible else None)
            if owner not in feasible:natural_fail+=1
            if chosen is not None:positive+=1;reassigned+=chosen!=owner
            record=dict(quadrant=q['quadrant'],event_interval=index,representative_u=str(u),owner_element=owner,
                crossed_elements=sorted(crossed),segment_parts=[dict(element=e,lo=str(lo),hi=str(hi)) for lo,hi,e in parts],
                eligible_patch_seeds=candidates,feasible_patch_seeds=feasible,assigned_patch_seed=chosen,
                reassigned=chosen is not None and chosen!=owner)
            assigned.append(record)
            if chosen is None:
                failures.append(dict(**record,start=list(map(str,start)),end=list(map(str,end)),
                    missing_by_candidate={str(s):sorted(crossed-patchsets[s]) for s in candidates}))
    assert {a['owner_element'] for a in assigned}==set(seeds)
    for q in quadrants:q['events']=[[str(x),str(y)] for x,y in q['events']]
    summary=dict(id=entry['id'],N=entry['N'],layers=entry['layers'],PDE_cases=len(entry['case_ids']),
        angular_cells=len(assigned),annulus_vertices=len(annulus_vertices),natural_failed_cells=natural_fail,
        reassigned_cells=reassigned,assigned_cells=positive,uncovered_cells=len(failures),
        all_tubes_certified=not failures,maximum_triangles_per_tube=max(len(a['crossed_elements']) for a in assigned),
        maximum_patch_elements=max(map(len,patches.values())),maximum_overlap=max(Counter(e for p in patches.values() for e in p).values()),
        seconds=time.perf_counter()-tick)
    full=dict(**entry,h=str(h),radius_exact=str(R),inner_radius=str(r0),center_exact=list(map(str,c)),target_depth_over_h='1/16',
        jacobian_lower_bound=str(r0/R),vertices=[[str(x) for x in v] for v in vertices],triangles=triangles,
        active_elements=active,patches=prior['patches'],annulus_vertex_indices=annulus_vertices,supporting_lines=line_count,
        quadrants=quadrants,assignments=assigned,failures=failures,summary=summary)
    print(json.dumps(summary),flush=True);return full

