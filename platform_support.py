"""Check the arithmetic and multiprocessing requirements of the selected study."""
import multiprocessing
import platform

STUDIES = ('all', 'baseline', 'schwarz', 'accurate', 'flux', 'controls',
           'radial', 'tubes', 'continuation', 'diagnostics', 'conditioning')
PORTABLE_STUDIES = frozenset(('continuation', 'tubes'))


def check_platform(study='all'):
    if study not in STUDIES:
        raise ValueError(f'Unknown study: {study!r}; choose from {", ".join(STUDIES)}')
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('NumPy is required for the platform check; use an environment containing requirements.txt.') from exc
    bits=int(np.finfo(np.longdouble).nmant)+1
    methods=multiprocessing.get_all_start_methods()
    # These two studies use software multiprecision and exact rational
    # certificates. Preserve the full validated requirements for all others.
    requires_extended = study not in PORTABLE_STUDIES
    requires_fork = requires_extended
    problems=[]
    if requires_extended and bits!=64:
        problems.append(f'numpy.longdouble has a {bits}-bit significand; this study requires the validated 64-bit significand')
    if requires_fork and 'fork' not in methods:
        problems.append('multiprocessing fork is unavailable; accurate assembly requires it')
    if problems:
        raise RuntimeError(f'Unsupported platform for {study}: '+ '; '.join(problems)+
                           '. Use the documented Linux x86-64 environment for this study. '
                           'The continuation and tubes studies do not require extended precision or fork. '
                           'File checks alone can use scripts/verify.py --files-only.')
    return dict(python=platform.python_version(),longdouble_significand_bits=bits,
                fork_available='fork' in methods,study=study,
                requires_extended_precision=requires_extended,requires_fork=requires_fork)
