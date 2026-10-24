"""Independent exact verifier for finite angular-arrangement tube certificates.

Does not call the search, root finder, exclusion bound or segment-clipping code.
It verifies the exhaustive event polynomials and root brackets, then checks
convex triangle membership at both endpoints of every supplied segment piece.
Coverage of [0,1] proves the complete representative segment; absence of
combinatorial events proves the full open angular cell.
"""
from fractions import Fraction as F
from collections import Counter
import json
from pathlib import Path
import run_2d_disk_study as fem

cross=lambda x,y:x[0]*y[1]-x[1]*y[0]
minus=lambda x,y:(x[0]-y[0],x[1]-y[1])
dot=lambda x,y:x[0]*y[0]+x[1]*y[1]


def norm_poly(values):
    first=next((x for x in values if x),None)
    return None if first is None else tuple(x/first for x in values)


def rotate(x,k):
    for _ in range(k):x=(-x[1],x[0])
    return x


def signs(x,triangle):
    return [cross(minus(triangle[(j+1)%3],triangle[j]),minus(x,triangle[j])) for j in range(3)]


def verify_roots(coefficients,brackets):
    A,B,C=coefficients;poly=lambda u:(A*u+B)*u+C
    if A:
        D=B*B-4*A*C
        expected=0 if D<0 else (1 if D==0 else 2)
    else:expected=1 if B else 0
    assert len(brackets)==expected
    for lo,hi in brackets:
        assert lo<=hi
        if lo==hi:assert poly(lo)==0
        else:
            assert poly(lo)*poly(hi)<0
            assert (2*A*lo+B)*(2*A*hi+B)>0
    assert all(a[1]<b[0] for a,b in zip(brackets,brackets[1:]))


def validate(data,old_witness=None,read_record=None):
    mesh=fem.build_mesh(data['N']);v=[tuple(map(F,row)) for row in data['vertices']]
    triangles=data['triangles'];c=tuple(map(F,data['center_exact']));R=F(data['radius_exact']);r0=F(data['inner_radius']);h=F(data['h'])
    assert v==[tuple(map(F,row)) for row in mesh.vertices]
    assert triangles==[list(e.vids) for e in mesh.elements]
    assert R==F(data['radius']) and c==tuple(map(F,data['center']))
    assert h==F(mesh.h) and R-r0==h/16 and F(data['jacobian_lower_bound'])==r0/R>=F(1,2)
    patches={p['seed']:set(p['elements']) for p in data['patches']}
    for path in data['runs']:
        raw=read_record(path) if read_record is not None else json.loads((Path(path)/'results.json').read_text())
        case=raw['case']
        assert case['N']==data['N'] and case['radius']==data['radius']
        # Older fixed-placement archives store the path name instead of center.
        # These constants are retained in their frozen workstation_case.py.
        center=case.get('center')
        if center is None:center={'fixed_A':[0.071,-0.043],'fixed_B':[-0.113,0.087]}[case['path']]
        assert center==data['center'] and case['patch_layers']==data['layers']
        assert raw['geometry']['active_elements']==data['active_elements']
        assert raw['geometry']['cut_elements']==list(patches)
        assert all(a['patches']==[sorted(p) for p in patches.values()] for a in raw['assembly'])
    lines=set()
    for tri in triangles:
        for j in range(3):
            x,y=v[tri[j]],v[tri[(j+1)%3]]
            a=x[1]-y[1];b=y[0]-x[0];d=x[0]*y[1]-x[1]*y[0]
            lines.add(norm_poly((a,b,d)))
    annulus=[i for i,x in enumerate(v) if r0*r0<=dot(minus(x,c),minus(x,c))<=R*R]
    assert annulus==data['annulus_vertex_indices'] and len(lines)==data['supporting_lines']
    total_polys=0;total_roots=0;quadrants={}
    for q in data['quadrants']:
        k=q['quadrant'];assert k not in quadrants and 0<=k<4
        expected=set()
        e0=rotate((F(1),F(0)),k);e1=rotate((F(0),F(1)),k)
        # Multiply each endpoint's supporting-line equation by 1+u².
        for a,b,d in lines:
            offset=a*c[0]+b*c[1]+d
            for radius in [r0,R]:
                x=radius*(a*e0[0]+b*e0[1]);y=radius*(a*e1[0]+b*e1[1])
                expected.add(norm_poly((offset-x,2*y,offset+x)))
        for i in annulus:
            z=minus(v[i],c);x=cross(z,e0);y=cross(z,e1)
            expected.add(norm_poly((-x,2*y,x)))
        expected.discard(None)
        actual={tuple(map(F,p['coefficients'])) for p in q['polynomials']}
        assert actual==expected and len(actual)==len(q['polynomials'])
        events={(F(0),F(0)),(F(1),F(1))}
        for p in q['polynomials']:
            coeff=tuple(map(F,p['coefficients']));brackets=[tuple(map(F,x)) for x in p['roots']]
            verify_roots(coeff,brackets);total_polys+=1;total_roots+=len(brackets)
            for lo,hi in brackets:
                if hi<0 or lo>1:continue
                assert 0<=lo<=hi<=1;events.add((lo,hi))
        events=sorted(events);assert events==[tuple(map(F,x)) for x in q['events']]
        assert all(a[1]<b[0] for a,b in zip(events,events[1:]))
        quadrants[k]=events
    assert set(quadrants)=={0,1,2,3}
    expected_cells={(k,i) for k,e in quadrants.items() for i in range(len(e)-1)}
    actual_cells=set();reassigned=0;negative=0;parts_checked=0;changed_owners=set();used=Counter()
    for a in data['assignments']:
        key=(a['quadrant'],a['event_interval']);assert key not in actual_cells;actual_cells.add(key)
        events=quadrants[key[0]];left,right=events[key[1]],events[key[1]+1];u=F(a['representative_u'])
        assert left[1]<u<right[0]
        n=rotate(((1-u*u)/(1+u*u),2*u/(1+u*u)),key[0])
        start=tuple(c[j]+R*n[j] for j in range(2));end=tuple(c[j]+r0*n[j] for j in range(2))
        assert dot(minus(start,c),minus(start,c))==R*R
        point=lambda t:tuple((1-t)*start[j]+t*end[j] for j in range(2))
        owner=a['owner_element'];assert all(s>0 for s in signs(start,[v[i] for i in triangles[owner]]))
        intervals=[];crossed=set()
        for part in a['segment_parts']:
            lo,hi=F(part['lo']),F(part['hi']);eid=part['element'];assert 0<=lo<hi<=1
            tri=[v[i] for i in triangles[eid]]
            assert all(s>=0 for s in signs(point(lo),tri)+signs(point(hi),tri))
            assert all(s>0 for s in signs(point((lo+hi)/2),tri))
            crossed.add(eid);intervals.append((lo,hi));parts_checked+=1
        reach=F(0)
        for lo,hi in sorted(intervals):assert lo<=reach;reach=max(reach,hi)
        assert reach==1 and crossed==set(a['crossed_elements']) and owner in crossed
        eligible=sorted(s for s,p in patches.items() if owner in p)
        feasible=[s for s in eligible if crossed<=patches[s]]
        assert eligible==a['eligible_patch_seeds'] and feasible==a['feasible_patch_seeds']
        chosen=a['assigned_patch_seed'];assert chosen in feasible
        assert chosen==(owner if owner in feasible else min(feasible,key=lambda s:(len(patches[s]),s)))
        assert a['reassigned']==(chosen!=owner);reassigned+=chosen!=owner;used[chosen]+=1
        if chosen!=owner:
            changed_owners.add(owner)
            # An explicit open segment piece is absent from the natural patch.
            assert crossed-patches[owner];negative+=1
    assert actual_cells==expected_cells
    summary=data['summary']
    assert summary['angular_cells']==len(actual_cells) and summary['reassigned_cells']==reassigned
    assert summary['natural_failed_cells']==negative and summary['uncovered_cells']==0 and summary['all_tubes_certified']
    witness_fixed=None
    if old_witness is not None:
        w=old_witness;u=F(w['u']);k=w['quadrant'];events=quadrants[k]
        interval=next(i for i,(a,b) in enumerate(zip(events,events[1:])) if a[1]<u<b[0])
        a=next(a for a in data['assignments'] if a['quadrant']==k and a['event_interval']==interval)
        x=tuple(map(F,w['point']));chosen=a['assigned_patch_seed']
        n=rotate(((1-u*u)/(1+u*u),2*u/(1+u*u)),k)
        depth=F(w['t_over_h'])*h
        assert 0<depth<h/16 and a['owner_element']==w['seed']
        assert tuple(map(F,w['boundary_point']))==tuple(c[j]+R*n[j] for j in range(2))
        assert x==tuple(c[j]+(R-depth)*n[j] for j in range(2))
        assert not any(all(s>=0 for s in signs(x,[v[i] for i in triangles[e]])) for e in patches[w['seed']])
        assert any(all(s>=0 for s in signs(x,[v[i] for i in triangles[e]])) for e in patches[chosen])
        witness_fixed=dict(owner_element=w['seed'],assigned_patch_seed=chosen,quadrant=k,event_interval=interval)
    return dict(id=data['id'],verified=True,cells=len(actual_cells),root_polynomials=total_polys,root_brackets=total_roots,
        segment_parts=parts_checked,reassigned_cells=reassigned,natural_failed_cells=negative,
        distinct_owners_reassigned=len(changed_owners),used_patch_count=len(used),original_witness_fixed=witness_fixed)
