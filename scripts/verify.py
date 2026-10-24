#!/usr/bin/env python3
"""Fast source, manifest, input checks; no experiments."""
from pathlib import Path
import argparse,ast,hashlib,re,sys
if not __debug__:raise SystemExit("Run without -O: assertions are part of the verification.")
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));import run
from platform_support import STUDIES, check_platform
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-manifest',action='store_true');p.add_argument('--files-only',action='store_true',help='Check packaged files without importing numerical dependencies');p.add_argument('--strict',action='store_true',help='Require exact manifest coverage of distributable files')
p.add_argument('--study',choices=STUDIES,help='Check platform requirements for this study; default: all studies')
a=p.parse_args()
if a.files_only and a.study is not None:p.error('--study checks numerical requirements and cannot be combined with --files-only')
if not a.files_only:
    study=a.study or 'all'
    try: environment=check_platform(study)
    except RuntimeError as exc: raise SystemExit(str(exc))
    fork='available' if environment['fork_available'] else 'unavailable'
    print(f'Platform verified for {study}: Python {environment["python"]}, '
          f'{environment["longdouble_significand_bits"]}-bit longdouble significand, fork {fork}.')
manifest=ROOT/'SHA256.json'
if a.require_manifest or a.strict:assert manifest.is_file()
if manifest.is_file():
    entries=run.read(manifest)
    for name,digest in entries.items():
        path=Path(name)
        assert not path.is_absolute() and '..' not in path.parts,name
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,name
    if a.strict:
        from package_release import distributable_files
        actual={str(p.relative_to(ROOT)) for p in distributable_files(ROOT)}
        assert actual==set(entries),dict(unlisted=sorted(actual-set(entries)),missing=sorted(set(entries)-actual))
    print(f'{len(entries)} file hashes verified.')
else:print('Manifest not yet sealed.')
files=list((ROOT/'code').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list(ROOT.glob('*.py'))
local={p.stem for p in files}
external=set()
for file in files:
    tree=ast.parse(file.read_text(),filename=str(file))
    assert ('/home/'+'karl/') not in file.read_text(),file
    for node in ast.walk(tree):
        imports=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
        for name in imports:
            top=name.split('.')[0]
            if top not in local and top not in sys.stdlib_module_names:external.add(top)
assert external<={'numpy','scipy','mpmath','matplotlib'},external
catalog=run.read(ROOT/'data/cases.json')
assert {k:len(v) for k,v in catalog.items()}==dict(baseline=118,schwarz=17,accurate=4)
for family,entries in catalog.items():
    for key,e in entries.items():
        if 'driver' in e:assert (ROOT/'code'/(e['driver']+'.py')).is_file()
        if 'reference' in e:
            ref=run.read(ROOT/e['reference']);assert len(ref['reference-identity.json']['sha256'])==64
            assert len(ref['reference-solution.json'])>0
        if family=='schwarz':assert e['operator_case'] in catalog['baseline']
geoms=run.read(ROOT/'data/geometries.json');assert len(geoms)==30
assert sorted(k for g in geoms for k in g['case_ids'])==sorted(catalog['baseline'])
assert sum(g['witness'] is not None for g in geoms)==11
published=run.read(ROOT/'data/published.json.gz');assert len(published['observations'])==4959 and len(published['failed_criteria'])==239
extended=[r for r in published['tables']['radial-fits'] if r['window']=='p4-12']
assert len(extended)==6 and {(r['disk'],r['norm']) for r in extended}=={(d,n) for d in ['fixed_A','fixed_B'] for n in ['L2','H1','energy']}
assert all(r['p_values']=='4;5;6;7;8;10;12' and r['resolved']=='True' for r in extended)
print(f'{len(files)} Python sources parsed; case catalog, compact fields and published comparisons verified.')

version=(ROOT/'VERSION').read_text().strip()
citation=(ROOT/'CITATION.cff').read_text()
assert re.search(r'^version: "'+re.escape(version)+r'"$',citation,re.M), 'Citation version mismatch'
validation=run.read(ROOT/'VALIDATION.json')
assert validation['release']['version']==version, 'Release version mismatch'
assert hashlib.sha256((ROOT/'docs/numerical-verification.pdf').read_bytes()).hexdigest()==validation['guide_build']['sha256'], 'Guide receipt mismatch'
for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',(ROOT/'README.md').read_text()):
    if '://' not in target and not target.startswith('#'):
        assert (ROOT/target.split('#')[0]).exists(), target
print(f'Release {version}: citation, guide receipt and README links verified.')
