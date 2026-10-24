"""Decimal polynomial moments on a triangle intersected with an exact disk.

Green's formula integrates oriented clipped triangle edges and the circle arcs
inside the triangle. No sampled volume quadrature enters these moments.
"""
from decimal import Decimal as D,localcontext
from math import comb,factorial
import decimal_retained_system as ds
import precise_quadrature_2d as pq
PI=pq.pi_decimal()

def dec(x):return x if isinstance(x,D) else ds.exact_decimal(x)
def power(x,k):return x**k if k else D(1)
def atan(x):
    if not x:return D(0)
    sign=D(-1) if x<0 else D(1);x=abs(x);invert=x>1
    if invert:x=1/x
    scale=1
    while x>D('.1'):x=x/(1+(1+x*x).sqrt());scale*=2
    term=x;total=x;k=0
    while True:
        k+=1;term*=-x*x;delta=term/(2*k+1);total+=delta
        if abs(delta)<D('1e-90'):break
        assert k<200
    result=scale*total
    return sign*(PI/2-result if invert else result)
def angle(x,y):
    if x>0:t=atan(y/x)
    elif x<0:t=atan(y/x)+(PI if y>=0 else -PI)
    else:t=PI/2 if y>0 else -PI/2
    return t if t>=0 else t+2*PI

def sincos(t):
    tau=2*PI
    while t>PI:t-=tau
    while t<-PI:t+=tau
    s=t;c=D(1);st=t;ct=D(1)
    for k in range(1,100):
        st*=-t*t/D(2*k*(2*k+1));ct*=-t*t/D((2*k-1)*2*k);s+=st;c+=ct
        if max(abs(st),abs(ct))<D('1e-90'):break
    else:raise AssertionError('trigonometric series failed')
    return s,c

def trig_moments(a,b,degree):
    sa,ca=sincos(a);sb,cb=sincos(b);out={}
    for total in range(degree+1):
        for m in range(total+1):
            n=total-m
            if n%2:
                k=(n-1)//2
                value=-sum((D(comb(k,j)*(-1)**j)*(power(cb,m+2*j+1)-power(ca,m+2*j+1))/D(m+2*j+1) for j in range(k+1)),D(0))
            elif m%2:
                k=(m-1)//2
                value=sum((D(comb(k,j)*(-1)**j)*(power(sb,n+2*j+1)-power(sa,n+2*j+1))/D(n+2*j+1) for j in range(k+1)),D(0))
            elif m>=2:
                endpoint=power(cb,m-1)*power(sb,n+1)-power(ca,m-1)*power(sa,n+1)
                value=(endpoint+(m-1)*out[m-2,n])/D(m+n)
            elif n>=2:
                endpoint=power(sb,n-1)*cb-power(sa,n-1)*ca;value=((n-1)*out[0,n-2]-endpoint)/D(n)
            else:value=b-a
            out[m,n]=value
    return out

def clip_boundary(vertices,center,radius):
    xy=[[dec(x) for x in row] for row in vertices];cx,cy=map(dec,center);R=dec(radius);R2=R*R
    assert len(xy)==3
    cross=lambda a,b:a[0]*b[1]-a[1]*b[0]
    assert cross([xy[1][i]-xy[0][i] for i in (0,1)],[xy[2][i]-xy[0][i] for i in (0,1)])>0
    edges=[];angles=[D(0),2*PI];intersection_count=0
    for i in range(3):
        a=xy[i];b=xy[(i+1)%3];d=[b[j]-a[j] for j in (0,1)];v=[a[0]-cx,a[1]-cy]
        A=sum(x*x for x in d);B=2*sum(x*y for x,y in zip(v,d));C=sum(x*x for x in v)-R2;disc=B*B-4*A*C
        if disc<0:continue
        root=disc.sqrt();lo=(-B-root)/(2*A);hi=(-B+root)/(2*A)
        for t in (lo,hi):
            if 0<=t<=1:
                x=a[0]+t*d[0]-cx;y=a[1]+t*d[1]-cy;angles.append(angle(x,y));intersection_count+=1
        lower=max(D(0),lo);upper=min(D(1),hi)
        if upper>lower:edges.append(([a[j]+lower*d[j] for j in (0,1)],[a[j]+upper*d[j] for j in (0,1)]))
    angles.sort();unique=[];merged=0
    for t in angles:
        if unique and t-unique[-1]<D('1e-65'):merged+=1;continue
        unique.append(t)
    if abs(unique[-1]-2*PI)<D('1e-65'):unique[-1]=2*PI
    def inside(point):
        return all(cross([xy[(i+1)%3][j]-xy[i][j] for j in (0,1)],[point[j]-xy[i][j] for j in (0,1)])>=0 for i in range(3))
    arcs=[]
    for a,b in zip(unique[:-1],unique[1:]):
        sn,cs=sincos((a+b)/2)
        if inside([cx+R*cs,cy+R*sn]):arcs.append((a,b))
    return edges,arcs,dict(edge_segments=len(edges),circle_arcs=len(arcs),intersections=intersection_count,merged_angles=merged,minimum_angle_interval=str(min((b-a for a,b in zip(unique[:-1],unique[1:])),default=D(0))))

def moments(vertices,center,radius,degree,digits=80):
    with localcontext() as ctx:
        ctx.prec=digits;cx,cy=map(dec,center);R=dec(radius);edges,arcs,info=clip_boundary(vertices,center,radius)
        indices=[(i,t-i) for t in range(degree+1) for i in range(t+1)];V={ij:D(0) for ij in indices};M=V.copy()
        for a,b in edges:
            dx=b[0]-a[0];dy=b[1]-a[1]
            xp=[[D(comb(k,j))*power(a[0],k-j)*power(dx,j) for j in range(k+1)] for k in range(degree+2)]
            yp=[[D(comb(k,j))*power(a[1],k-j)*power(dy,j) for j in range(k+1)] for k in range(degree+1)]
            for i,j in indices:V[i,j]+=dy/D(i+1)*sum((x*y/D(k+l+1) for k,x in enumerate(xp[i+1]) for l,y in enumerate(yp[j])),D(0))
        for a,b in arcs:
            T=trig_moments(a,b,degree+2)
            def arc(i,j,extra_cos=0):
                return R*sum((D(comb(i,k)*comb(j,l))*power(cx,i-k)*power(cy,j-l)*power(R,k+l)*T[k+extra_cos,l] for k in range(i+1) for l in range(j+1)),D(0))
            for i,j in indices:M[i,j]+=arc(i,j);V[i,j]+=arc(i+1,j,1)/D(i+1)
        info.update(degree=degree,digits=digits,area=str(V[0,0]),circle_length=str(M[0,0]),arcs=[[str(a),str(b)] for a,b in arcs],edges=[[[str(x) for x in a],[str(x) for x in b]] for a,b in edges])
    return V,M,info

def self_check(degree=16):
    cases=[('triangle',[[0,0],[1,0],[0,1]],D(2)),('quarter',[ [0,0],[3,0],[0,3]],D(1)),('half',[[0,-3],[3,0],[0,3]],D(1)),('disk',[[-3,-3],[3,-3],[0,3]],D(1))];records=[]
    with localcontext() as ctx:
        ctx.prec=80
        def gamma(twice):
            if twice%2==0:return D(factorial(twice//2-1))
            n=(twice-1)//2;return D(factorial(2*n))/D(4**n*factorial(n))*PI.sqrt()
        for label,vertices,R in cases:
            V,M,info=moments(vertices,[D(0),D(0)],R,degree);worstV=worstM=D(0)
            for (i,j),v in V.items():
                if label=='triangle':truth=D(factorial(i)*factorial(j))/D(factorial(i+j+2));surface=D(0)
                else:
                    angular=gamma(i+1)*gamma(j+1)/gamma(i+j+2)/2
                    factor=D(1) if label=='quarter' else D(0) if j%2 or (label=='disk' and i%2) else D(2 if label=='half' else 4)
                    truth=factor*angular/D(i+j+2);surface=factor*angular
                worstV=max(worstV,abs(v-truth));worstM=max(worstM,abs(M[i,j]-surface))
            assert max(worstV,worstM)<D('1e-60'),(label,worstV,worstM)
            records.append(dict(case=label,maximum_volume_error=str(worstV),maximum_boundary_error=str(worstM),geometry=info))
    return records
