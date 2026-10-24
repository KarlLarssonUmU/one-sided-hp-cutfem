# Validation Details

The [validation record](../VALIDATION.json) contains numerical checks, tested environments, source hashes and retained accuracy flags. The [Numerical Verification Guide](numerical-verification.pdf) explains the checks and their mathematical scope. For installation and supported platforms, see the [README](../README.md).

## Selected Linux Validation

Version **2026.10.24-code.2** was tested on Linux x86-64 on 24 October 2026 with Python 3.12.14, the pinned dependencies, a 64-bit `numpy.longdouble` significand and multiprocessing `fork`. The run used four CPUs and 16 GiB and took approximately ten minutes.

The selected suite covered **15 numerical commands across ten study families**, including PDE calculations, reporting and resume. Seven platform-policy tests passed. The campaign report contained 61 comparisons, with no differences under its 1% report tolerance and no resolution changes. This report tolerance is distinct from each study's scientific acceptance criteria. All 18 separate norm-repeat checks passed the relative tolerance of `1e-6`; the largest difference was `9.4471384e-7`, for the refined `EXT-N4-p8-fixed_A` L2 error.

Four baseline-accuracy flags in the accurate study and the expected radial $p=16$ `LinAlgError` were retained. The selected outputs also contained 56 unresolved observations. Strict file checks passed before and after execution. Detailed results appear under `selected_validation` in [VALIDATION.json](../VALIDATION.json).

This is a selected validation, not a rerun of the complete 257-command campaign or an audit of omitted raw archives. The selected campaign report is incomplete relative to the full reference bundle. Agreement with stored fields does not establish independence of the assembly implementation.

## Apple Silicon Validation

With Python 3.12.14 and the pinned dependencies on Apple Silicon macOS, the continuation study for degrees 1–14 and all 30 tube geometries passed. Checks included independent exact certificate verification, the 78-row seed-assignment audit, a saved-certificate recheck, reporting and resume. All 479 fresh comparisons matched the reference results, with no resolution changes.

These checks cover continuation and tubes only. The eight PDE and related studies remain unsupported on the tested 53-bit `longdouble` platform. Details appear under `experiment_platform_validation` in [VALIDATION.json](../VALIDATION.json).

## Additional Environment Checks

Selected checks were also reproduced using Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1 and mpmath 1.4.1: the quick-start Schwarz case, continuation and one tube certificate agreed with the reference results. The complete campaign was not tested in that environment. The dependency pins specify the Python 3.12 configuration.

An external review reported a further check with Python 3.13.17, NumPy 2.5.4, SciPy 1.18.1 and mpmath 1.4.1, using two CPUs and 7 GiB. Both portable studies gave 479 comparisons with no differences or resolution changes. The quick-start PDE/Schwarz case gave 43 comparisons with no differences; its L2 and energy errors agreed to approximately `4e-13` relatively. The reviewer also reported seven passing platform tests, a 64-bit `longdouble` significand and available `fork`. These results are recorded as reviewer-reported checks; their raw logs are not included. They do not establish validation of the complete campaign on that dependency stack.

## Running the Checks

All commands below run from the repository root. Integrity and platform-policy checks do not run numerical experiments:

```sh
python scripts/verify.py --files-only --require-manifest --strict
python -B scripts/test_platform_support.py
```

To run the selected numerical suite in the supported Linux environment, choose a new output directory:

```sh
python scripts/verify.py
python scripts/validate.py --output runs/validation
```

This suite exercises ten study families; it does not execute the full campaign. For a full reproduction or individual studies, use the [Reproduction Guide](reproduction.md).

## Provenance and Integrity

[SHA256.json](../SHA256.json) lists the distributed file hashes, and [data/source-origins.json](../data/source-origins.json) records the numerical modules' provenance. The validation record identifies the tested sources, inputs and archive, as well as earlier numerical validation under `historical_validation`. Archive hashes identify the exact tested copies; the current file inventory is defined by `SHA256.json`.

Compact reference data and validation fields are included. Raw run archives and the compact validation receipt archive identified by its hash are not included. Detailed logs, matrices and exact tube certificates are produced by the supplied commands. Validation records retain failed criteria and unresolved results; a successful process exit does not imply that every reported measurement is resolved.
