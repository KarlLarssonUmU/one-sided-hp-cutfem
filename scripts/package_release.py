#!/usr/bin/env python3
"""Seal and package a local release; never run experiments or publish files."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def distributable_files(root=ROOT):
    """Return the complete source package, excluding documented local artifacts."""
    paths = []
    for directory, folders, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        excluded = {'.git', '.venv', 'venv', '__pycache__'}
        if parent == root:
            excluded.update({'runs', 'reports', 'published'})
        folders[:] = sorted(name for name in folders if name not in excluded)
        for name in folders:
            if (parent / name).is_symlink():
                raise ValueError(f'Symlink not allowed in a release: {parent / name}')
        for name in sorted(files):
            if name in {'.DS_Store', 'Thumbs.db'} or name.endswith(('.pyc', '.pyo')):
                continue
            path = parent / name
            if path.is_symlink():
                raise ValueError(f'Symlink not allowed in a release: {path}')
            if path != root / 'SHA256.json':
                paths.append(path)
    return sorted(paths)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New ZIP file outside the repository')
    parser.add_argument('--refresh-manifest', action='store_true',
                        help='Intentionally reseal SHA256.json after reviewed file changes')
    args = parser.parse_args()
    version = (ROOT / 'VERSION').read_text().strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', version):
        parser.error('VERSION must be a safe, nonempty release identifier')
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('Write the ZIP outside the repository')
    checksum = output.with_name(output.name + '.sha256')
    if output.suffix != '.zip' or output.exists() or checksum.exists():
        parser.error('Choose a new .zip path; existing archives/checksums are preserved')
    files = distributable_files()
    if args.refresh_manifest:
        manifest = {str(p.relative_to(ROOT)): sha256(p) for p in files}
        (ROOT / 'SHA256.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/verify.py'),
                    '--files-only', '--require-manifest', '--strict'], check=True)
    manifest = json.loads((ROOT / 'SHA256.json').read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = f'hp-paper-repository-{version}'
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in [*files, ROOT / 'SHA256.json']:
            payload = path.read_bytes()
            name = str(path.relative_to(ROOT))
            if name != 'SHA256.json' and hashlib.sha256(payload).hexdigest() != manifest[name]:
                raise RuntimeError(f'File changed after verification: {name}')
            entry = zipfile.ZipInfo(f'{prefix}/{name}', date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f'ZIP integrity failure: {bad}')
    checksum.write_text(f'{sha256(output)}  {output.name}\n')
    print(f'Prepared {output} ({output.stat().st_size} bytes) and {checksum.name}')


if __name__ == '__main__':
    main()
