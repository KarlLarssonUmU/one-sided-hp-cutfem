"""Finite arithmetic diagnostics; no PDE recovery or reclassification of flags.

Independently compare the stored mathematical basis with Decimal100 evaluation,
inspect rounded-system spectra, and check whether a failed bubble Gram matrix
has a positive high-precision Cholesky factor. All inputs are explicitly listed.
"""
from pathlib import Path
from decimal import Decimal as D,localcontext
from fractions import Fraction as F
from math import factorial
import os,sys,json,time,platform,getpass,traceback
import numpy as np
import scipy
from scipy import linalg as la
import mpmath as mp
import run_2d_disk_study as fem
import paper_solver_2d as solver
import workstation_case as wc
import decimal_coefficient_identity as identity

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def bubble_gram(p):
    polys=[]
    for t in range(p-2):
        for i in range(t+1):
            j=t-i;polys.append({(i+1,j+1):F(1),(i+2,j+1):F(-1),(i+1,j+2):F(-1)})
    def deriv(poly,axis):
        return {(i-1,j) if axis==0 else (i,j-1):(i if axis==0 else j)*v for (i,j),v in poly.items() if (i if axis==0 else j)}
    gradients=[[deriv(q,a) for q in polys] for a in [0,1]]
    moments={(i,j):F(factorial(i)*factorial(j),factorial(i+j+2)) for i in range(2*p+1) for j in range(2*p+1-i)}
    def inner(a,b):return sum((x*y*moments[i+k,j+l] for (i,j),x in a.items() for (k,l),y in b.items()),F(0))
    gram=[[inner(a,b)+sum(inner(g[i],g[j]) for g in gradients) for j,b in enumerate(polys)] for i,a in enumerate(polys)]
    rr,ss,ww=fem.triangle_quadrature(max(7,p+3));v,g=fem.eval_polys([{ij:float(x) for ij,x in q.items()} for q in polys],rr,ss)
    rounded=(v*ww)@v.T+(g[:,0]*ww)@g[:,0].T+(g[:,1]*ww)@g[:,1].T
    rounded=(rounded+rounded.T)/2
    result=dict(degree=p,dimension=len(polys),binary64_smallest_eigenvalue=float(la.eigvalsh(rounded,subset_by_index=[0,0])[0]))
    try:la.cholesky(rounded);result['binary64_cholesky']=True
    except la.LinAlgError:result['binary64_cholesky']=False
    pivots=[]
    for digits in [80,100]:
        with mp.workdps(digits):
            G=mp.matrix([[mp.mpf(x.numerator)/x.denominator for x in row] for row in gram]);L=mp.cholesky(G)
            err=mp.norm(G-L*L.T,mp.inf)/mp.norm(G,mp.inf)
            assert err<mp.mpf(10)**(-digits+10)
            pivots.append([L[i,i] for i in range(len(polys))])
            result[f'MP{digits}_relative_reconstruction']=str(err)
            result[f'MP{digits}_minimum_cholesky_diagonal']=str(min(pivots[-1]))
    with mp.workdps(100):
        repeat=max(abs(a/b-1) for a,b in zip(*pivots));assert repeat<mp.mpf('1e-55')
        result['MP80_MP100_pivot_relative_change']=str(repeat)
    return result

def basis_audit(p):
    case=wc.normalize_case(dict(N=4,p=p,path='fixed_A',patch_layers=2,solution_kind='radial_power'))
    mesh,c,space,basis,dm=solver.setup(case)
    rs=np.array([[.125,.375],[.499,.499],[.001,.498],[.25,0.],[0.,.25],[.375,.625]])
    maxerror=0.;count=0;seen=set()
    with localcontext() as ctx:
        ctx.prec=100
        for eid,polys in space['basis'].items():
            if id(polys) in seen:continue
            seen.add(id(polys));v,g=basis.reference(eid,rs[:,0],rs[:,1])
            for k,poly in enumerate(polys):
                dec={ij:identity.decimal_value(x) for ij,x in poly.items()}
                for q,(rr,ss) in enumerate(rs):
                    truth=identity.evaluate_decimal(dec,D.from_float(float(rr)),D.from_float(float(ss)))
                    for got,expected in zip((v[k,q],g[k,0,q],g[k,1,q]),truth):
                        maxerror=max(maxerror,float(abs(D.from_float(float(got))-expected)/max(D(1),abs(expected))));count+=1
    # A diagnostic comparison, not a field-error certificate or a new PDE gate.
    assert np.isfinite(maxerror)
    return dict(degree=p,points_per_basis=6,comparisons=count,patterns=len(seen),Decimal100_max_scaled_difference=maxerror,
                interpretation='Sampled basis evaluation comparison only; does not certify assembly or solution accuracy.')

def main(run):
    spec=json.loads((run/'parameters.json').read_text());output=dict(bubble_grams=[],basis=[],systems=[])
    for p in spec.get('degrees',[12,14,16]):
        output['bubble_grams'].append(bubble_gram(p))
        try:output['basis'].append(basis_audit(p))
        except Exception as exc:output['basis'].append(dict(degree=p,status='setup_failed',exception=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc()))
        dump(run/'output/diagnostics.json',output);print('basis',p,flush=True)
    for source in spec['matrix_inputs']:
        path=Path(source);saved=np.load(path);A=saved['A'];assert np.all(np.diag(A)>0)
        d=1/np.sqrt(np.diag(A));J=(A*d[:,None])*d[None,:];assert np.max(abs(J-J.T))<1e-12
        lo=float(la.eigvalsh(J,subset_by_index=[0,0])[0]);hi=float(la.eigvalsh(J,subset_by_index=[len(J)-1,len(J)-1])[0])
        output['systems'].append(dict(source=source,ndof=len(J),lambda_min=lo,lambda_max=hi,
            kappa_J_estimate=hi/lo if lo>0 else None,epsilon_times_kappa=hi/lo*np.finfo(float).eps if lo>0 else None,
            caution='Binary64 eigenvalues near machine precision are diagnostic estimates, not certified enclosures.'))
        dump(run/'output/diagnostics.json',output);print('matrix',path.parent.name,lo,flush=True)
    return 0

if __name__=='__main__':
    run=Path(sys.argv[1]).resolve();assert platform.node().split('.')[0]=='compmath1' and getpass.getuser()=='karl'
    start=time.time();code=1
    dump(run/'environment.json',dict(host=platform.node(),user=getpass.getuser(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,mpmath=mp.__version__,device='CPU',dtype='binary64, exact rational Gram, MP80/100, Decimal100',seed=None,threads=os.environ.get('OPENBLAS_NUM_THREADS')))
    try:code=main(run)
    finally:dump(run/'completion.json',dict(exit_code=code,started=start,finished=time.time(),seconds=time.time()-start,job_id=os.environ.get('SLURM_JOB_ID')))
    sys.exit(code)
