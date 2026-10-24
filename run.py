#!/usr/bin/env python3
"""Explicit, bounded experiment entry points; paths are relative to this file."""
from pathlib import Path
import argparse, ast, gzip, hashlib, importlib, json, os, platform, shutil
import subprocess, sys, time, traceback

ROOT = Path(__file__).resolve().parent

def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as f:
        return json.load(f)

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def catalog(name): return read(ROOT/'data/cases.json')[name]

def source_hashes():
    files = [ROOT/'run.py', ROOT/'platform_support.py', *(ROOT/'code').glob('*.py'), *(ROOT/'data').rglob('*')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files) if p.is_file()}

def stage_inputs(entry, destination):
    destination.mkdir(parents=True, exist_ok=True)
    names = []
    if 'reference' in entry:
        for name, value in read(ROOT/entry['reference']).items():
            dump(destination/name, value); names.append(name)
    if 'ordinary' in entry:
        shutil.copy2(ROOT/entry['ordinary'], destination/'baseline-solutions.npz')
        names.append('baseline-solutions.npz')
    return names

def baseline(entry, run):
    code = ROOT/'code'
    names = stage_inputs(entry, run/'inputs')
    dump(run/'parameters.json', dict(entry['parameters'], inputs={name:str(run/'inputs'/name) for name in names}))
    bootstrap = (
        'import sys; from pathlib import Path; '
        'sys.path.insert(0,sys.argv[1]); import importlib; p=importlib.import_module(sys.argv[3]); '
        'p.RUN=Path(sys.argv[2]); p.main()')
    subprocess.run([sys.executable, '-B', '-c', bootstrap, str(code), str(run), entry['driver']], check=True)
    result = read(run/'results.json')
    return 0 if all(g['passed'] for g in result['gates']) else 2

def schwarz(entry, run, operator):
    assert operator is not None, '--operator must name the output of the matching baseline case'
    operator = operator.resolve()
    saved = read(operator/'results.json')
    assert saved['case_id'] == entry['operator_case'], 'Wrong baseline case'
    code = ROOT/'code'
    (run/'inputs').mkdir(exist_ok=True)
    for source, target in [('results.json','operator-results.json'),('case-identity.json','operator-identity.json'),('q12-paper.npz','operator.npz')]:
        shutil.copy2(operator/source, run/'inputs'/target)
    dump(run/'parameters.json', entry['parameters'])
    bootstrap = (
        'import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); '
        'import importlib; p=importlib.import_module(sys.argv[3]); import driver_sequence as s; '
        'p.RUN=s.RUN=Path(sys.argv[2]); p.main()')
    subprocess.run([sys.executable, '-B', '-c', bootstrap, str(code), str(run), entry['driver']], check=True)
    return 0 if all(g['passed'] for g in read(run/'results.json')['gates']) else 2

def tubes(run, geometry, certificates=None, seed_audit=False):
    import numpy as np
    import paper_solver_2d as solver
    import run_2d_disk_study as fem
    import certified_geometry as geo
    import tube_certificates as generator
    import tube_verifier as verifier
    geo.mp.mp.dps = 80
    entries = [e for e in read(ROOT/'data/geometries.json') if geometry is None or e['id']==geometry]
    assert entries, 'Unknown geometry'
    rows=[]; seed_rows=[]
    for spec in entries:
        entry={k:v for k,v in spec.items() if k!='witness'}
        entry['runs']=entry['case_ids']
        mesh=fem.build_mesh(entry['N'])
        active, interior, cut=solver.classify(mesh,np.array(entry['center']),entry['radius'])
        patches=fem.build_patches(mesh,active,cut,entry['layers'])
        def anchor(case_id):
            case=catalog('baseline')[case_id]['parameters']['case'].copy()
            if 'center' not in case:
                import workstation_case as wc
                case['center']=wc.geometry(wc.normalize_case(case))[1].tolist()
            assert (case['N'],case['radius'],case['center'],case['patch_layers']) == (
                entry['N'],entry['radius'],entry['center'],entry['layers'])
            return dict(case=case,geometry=dict(active_elements=active,cut_elements=cut),assembly=[dict(patches=patches)])
        prior=dict(patches=[dict(seed=seed,elements=els) for seed,els in zip(cut,patches)])
        if certificates is None:
            data=generator.check(entry,prior,read_record=anchor)
            dump(run/'output/certificates'/(entry['id']+'.json'),data)
        else:
            data=read(certificates/(entry['id']+'.json'))
            assert all(data[k]==entry[k] for k in entry), 'Certificate is for another geometry/case selection'
        rows.append(verifier.validate(data,spec['witness'],read_record=anchor))
        if seed_audit:seed_rows.extend(geo.check_geometry(entry,read_record=anchor))
    dump(run/'output/verification.json', dict(verified=True,geometries=len(rows),
        cells=sum(r['cells'] for r in rows),reassigned=sum(r['reassigned_cells'] for r in rows),
        earlier_witnesses=sum(r['original_witness_fixed'] is not None for r in rows),records=rows))
    if seed_audit:dump(run/'output/seed-assignment.json',seed_rows)
    if geometry is None:
        assert (len(rows),sum(r['cells'] for r in rows),sum(r['reassigned_cells'] for r in rows))==(30,5324,97)
        assert sum(r['original_witness_fixed'] is not None for r in rows)==11
    return 0

def conditioning(run):
    """Regenerate the four cached operators, then run the unchanged matrix test."""
    import numpy as np
    import paper_solver_2d as solver
    import workstation_case as wc
    for layers, key in [(2,'PAPER-028'),(4,'PAPER-044')]:
        case = wc.normalize_case(catalog('baseline')[key]['parameters']['case'])
        mesh, center, space, basis, _ = solver.setup(case)
        for boost in [8,12]:
            A,E,b,parts,_ = solver.assemble(mesh,space,center,case,basis,boost)
            np.savez_compressed(run/'inputs'/f'L{layers}-q{boost}.npz',A=A,E=E,S1=parts['S1'])
    dump(run/'parameters.json', dict(previous_L2_kappa_J=4650271.371490046))
    subprocess.run([sys.executable,'-B',str(ROOT/'code/conditioning_controls.py'),str(run)], check=True)
    return 0

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    ls = commands.add_parser('list', help='List published cases without running computations')
    ls.add_argument('family', choices=['baseline','schwarz','accurate','tubes'])
    for name in ['baseline','schwarz','accurate','flux','controls','radial','tubes','continuation','diagnostics','conditioning']:
        q = commands.add_parser(name)
        q.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are refused')
        q.add_argument('--threads', type=int, default=1, help='BLAS threads; allocate this many CPUs')
        if name in ['baseline','schwarz','accurate']: q.add_argument('case')
        if name == 'schwarz': q.add_argument('--operator', type=Path, required=True)
        if name == 'accurate': q.add_argument('--workers', type=int, default=1, help='Assembly worker processes; BLAS is forced to one thread')
        if name in ['flux','controls','radial','diagnostics']:
            q.add_argument('--p', type=int, required=True)
        if name == 'flux':
            q.add_argument('--path', choices=['axis','diagonal','oblique'], required=True)
            q.add_argument('--epsilon', type=float, choices=[.03,.003,.0003,.00003], required=True)
        if name == 'radial': q.add_argument('--disk', choices=['fixed_A','fixed_B'], required=True)
        if name == 'diagnostics': q.add_argument('--matrix', action='append', type=Path, default=[])
        if name == 'tubes':
            q.add_argument('--geometry', help='Optional geometry-01 through geometry-30; default verifies all 30')
            q.add_argument('--certificates',type=Path,help='Verify an existing generated certificate directory; default regenerates it')
            q.add_argument('--seed-audit',action='store_true',help='Also reproduce earlier seed assignment bounds and alternative depths')
    return p

def main():
    args = parser().parse_args()
    if not __debug__: raise RuntimeError('Assertions are part of the numerical checks; do not use python -O')
    if args.command == 'list':
        if args.family == 'tubes':
            print('\n'.join(e['id'] for e in read(ROOT/'data/geometries.json')))
        else:
            for key, entry in sorted(catalog(args.family).items()):
                print(key, json.dumps(entry['parameters'].get('case', {}), sort_keys=True))
        return 0
    assert args.threads > 0
    if args.command in ['flux','controls']: assert 1 <= args.p <= 8
    if args.command == 'radial': assert args.p in [10,12,14,16]
    if args.command == 'diagnostics': assert args.p in [12,14,16]
    threads = 1 if args.command == 'accurate' else args.threads
    if args.command == 'accurate': assert args.workers > 0
    for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name] = str(threads)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PYTHONHASHSEED'] = '0'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT/'code'))
    from platform_support import check_platform
    try: platform_check = check_platform(args.command)
    except RuntimeError as exc: raise SystemExit(str(exc))
    run = args.output.resolve()
    # All fresh run artifacts belong outside the curated source/data directories.
    protected = [ROOT/'code', ROOT/'data', ROOT/'cases', ROOT/'docs', ROOT/'paper', ROOT/'scripts']
    assert not any(run==p or p in run.parents for p in protected), 'Choose an output under runs/ or outside the repository'
    run.mkdir(parents=True, exist_ok=False)
    for name in ['output','inputs','logs']: (run/name).mkdir()
    dump(run/'request.json', {k:str(v) if isinstance(v,Path) else [str(x) for x in v] if isinstance(v,list) else v for k,v in vars(args).items()})
    import numpy as np, scipy, mpmath
    dump(run/'environment.json', dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,mpmath=mpmath.__version__,
        platform=platform.platform(),device='CPU',blas_threads=threads,seed=None,longdouble_mantissa=np.finfo(np.longdouble).nmant,
        source_version=(ROOT/'VERSION').read_text().strip(),slurm_job=os.environ.get('SLURM_JOB_ID'),
        platform_check=platform_check))
    # Capture exactly the code used, not mutable absolute workstation references.
    hashes = source_hashes()
    dump(run/'source-sha256.json', hashes)
    start = time.time(); result = 1
    try:
        if args.command == 'baseline': result = baseline(catalog('baseline')[args.case],run)
        elif args.command == 'schwarz': result = schwarz(catalog('schwarz')[args.case],run,args.operator)
        elif args.command == 'accurate':
            import paper_high_precision_calculation as worker
            entry = catalog('accurate')[args.case]
            stage_inputs(entry,run/'inputs'/args.case)
            worker.RUN = run
            result = 0 if worker.one(dict(entry['parameters'], workers=args.workers)) else 2
        elif args.command == 'flux':
            import flux_spectra as worker
            shutil.copy2(ROOT/'data/cut-geometries.csv',run/'inputs/cut-cases.csv')
            worker.cut_study(run,[args.p],args.path,args.epsilon); result = 0
        elif args.command == 'controls':
            import flux_spectra as worker
            worker.controls_study(run,[args.p]); result = 0
        elif args.command == 'radial':
            import radial_extension as worker
            dump(run/'parameters.json',dict(degrees=[args.p],disks=[args.disk]))
            result = worker.main(run)
        elif args.command == 'tubes': result = tubes(run,args.geometry,args.certificates,args.seed_audit)
        elif args.command == 'continuation':
            import certified_geometry as worker
            worker.mp.mp.dps=80
            rows=worker.continuation();worker.table(run/'output/continuation-sharp.csv',rows);result=0
        elif args.command == 'diagnostics':
            import radial_diagnostics as worker
            dump(run/'parameters.json',dict(degrees=[args.p],matrix_inputs=[str(p.resolve()) for p in args.matrix]))
            result=worker.main(run)
        elif args.command == 'conditioning': result=conditioning(run)
        return result
    except Exception:
        dump(run/'failure.json',dict(traceback=traceback.format_exc()))
        raise
    finally:
        dump(run/'completion.json',dict(exit_code=result,started=start,finished=time.time(),seconds=time.time()-start,
            status='passed' if result==0 else 'accuracy_flag' if result==2 else 'execution_failed'))

if __name__ == '__main__': sys.exit(main())
