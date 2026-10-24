"""Unshifted, diagonally scaled Nitsche threshold with retained brackets."""
import time
import numpy as np


def threshold(parts,E,p,h,gamma1):
    tick=time.perf_counter();d=1/np.sqrt(np.diag(E))
    L=np.linalg.cholesky(E*d[:,None]*d[None,:]);Q=np.linalg.solve(L,np.diag(d))
    A0=parts['K']+gamma1*parts['S1']-parts['Cn']-parts['Cn'].T
    penalty=(p*p/h)*parts['Mb'];H0=Q@A0@Q.T;Hp=Q@penalty@Q.T
    H0=(H0+H0.T)/2;Hp=(Hp+Hp.T)/2;history=[];target=1e-8
    def eigen(beta,vectors=False):
        H=H0+beta*Hp
        if vectors:
            w,V=np.linalg.eigh(H);return float(w[0]),Q.T@V[:,0]
        value=float(np.linalg.eigvalsh(H)[0]);history.append(dict(beta=beta,lambda_min=value));return value
    lo=0.;hi=1.
    if eigen(lo)>target:hi=lo
    else:
        for _ in range(13):
            if eigen(hi)>target:break
            hi*=2
        else:raise ArithmeticError('Threshold not bracketed through beta=4096')
        while hi-lo>1e-4*max(1,hi):
            mid=(lo+hi)/2
            if eigen(mid)>target:hi=mid
            else:lo=mid
    endpoints=[];ld=np.longdouble;ee=E.astype(ld);qq=Q.astype(ld)
    for beta in (lo,hi):
        val,x=eigen(beta,True);xx=x.astype(ld)
        aa=A0.astype(ld)+ld(beta)*penalty.astype(ld)
        residual=qq@(aa@xx-ld(val)*(ee@xx));energy=xx@(ee@xx)
        endpoints.append(dict(beta=beta,lambda_min=val,
            whitened_residual=float(np.sqrt(residual@residual/energy))))
    return dict(lower=lo,upper=hi,width=hi-lo,target=target,metric_shift=0.,
                endpoints=endpoints,history=history,seconds=time.perf_counter()-tick)
