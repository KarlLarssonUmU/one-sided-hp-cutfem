# One-Sided hp-CutFEM

Numerical companion to **One-Sided p-Explicit Stability for High-Order Cut Finite Element Nitsche Methods**, by Erik Burman, Peter Hansbo, Mats G. Larson and Karl Larsson.

This repository contains the code, experiment parameters, reference data and [Numerical Verification Guide](docs/numerical-verification.pdf) for the paper. The experiments use two-dimensional triangular $P_p$ finite elements and CPU-based dense linear algebra. They investigate stability, convergence, numerical precision, Schwarz comparisons and exact geometric certificates.

**DOI:** [10.5281/zenodo.23205908](https://doi.org/10.5281/zenodo.23205908)

## Installation

Use Python 3.12 and the pinned dependencies in [requirements.txt](requirements.txt). Selected checks on newer dependency stacks are described in [Additional Environment Checks](docs/validation.md#additional-environment-checks). From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify.py --files-only --require-manifest --strict
```

The last command checks file integrity, source syntax, metadata and reference data without running experiments. No TeX installation is needed. After installing dependencies, experiments run without a network connection or GPU.

## Quick Start

The continuation and tube-certificate studies run on both Apple Silicon macOS and Linux x86-64:

```sh
python scripts/verify.py --study continuation --require-manifest --strict
python scripts/verify.py --study tubes --require-manifest --strict
python reproduce.py --case continuation --case tubes --output runs/portable --execute
python report.py --runs runs/portable --output reports/portable
```

These commands compute the continuation constants, generate and independently verify the exact tube certificates, audit the seed assignment, and compare fresh results with the reference data. The report covers only these two studies, so it is incomplete relative to the full reference bundle.

For a small PDE case and its Schwarz comparisons, use the supported Linux environment described below:

```sh
python scripts/verify.py
python reproduce.py --case schwarz/PAPER-004 --output runs/small --execute
python report.py --runs runs/small --output reports/small --plots
```

The matching baseline is included automatically. Use a new output directory for each run and report. Run Python without `-O`, which disables required assertions.

To inspect the reference tables without running numerical experiments:

```sh
python report.py --published --output published
```

See the [Reproduction Guide](docs/reproduction.md) for individual studies, Figure 9, the complete campaign, resource requirements, resuming runs and interpreting reports. The [Numerical Verification Guide](docs/numerical-verification.pdf) describes the methods, triangular basis, accuracy checks and their limitations.

## Platform and Precision Requirements

| Studies | Supported configuration |
|---|---|
| `continuation`, `tubes` | Tested on Apple Silicon macOS and Linux x86-64. Software multiprecision and exact rational arithmetic allow a 53-bit `numpy.longdouble` significand; multiprocessing `fork` is not required. |
| `baseline`, `schwarz`, `accurate`, `flux`, `controls`, `radial`, `diagnostics`, `conditioning`, and the full campaign | Validated on Linux x86-64 with a **64-bit `numpy.longdouble` significand** and multiprocessing **`fork`**. The tested Apple Silicon environment provides only 53 bits and is unsupported for these studies. |

These precision requirements apply to the supplied validation workflow. The standard PDE assembly and initial solve use binary64; subsequent mixed-precision refinement uses `longdouble` residuals and working iterates, with binary64 correction solves and final coefficients. Norm and spectral checks also use extended precision. The mathematical method itself does not require this particular arithmetic format.

`run.py` checks the requirements for the selected study. Use `scripts/verify.py --study NAME` for a study-specific check; without `--study`, it checks the requirements for all studies. File-only verification, case listing, campaign planning and reference-data reporting do not require extended precision.

## Repository Contents

| Location | Contents |
|---|---|
| `run.py`, `reproduce.py`, `report.py` | Individual experiments, campaign plans and reports |
| `code/` | Numerical implementation |
| `data/` | Case parameters, compact reference results and validation fields |
| `docs/` | Reproduction instructions and Numerical Verification Guide |
| `scripts/` | Integrity checks, selected numerical validation and archive tools |
| [VALIDATION.json](VALIDATION.json) | Validation results, environments and tested-source hashes |
| [SHA256.json](SHA256.json) | File integrity manifest |

The compact data include **4,959 reference observations**, **239 failed criteria** and **22 tables**. Known accuracy flags, the original solver-stopping failure and unresolved higher-degree attempts are retained. Raw run archives and manuscript sources are not included; detailed run records, matrices and exact certificates are generated by the experiment commands.

## Validation and Limitations

Selected validation on Linux x86-64 covered **15 numerical commands across ten study families**. All 61 report comparisons agreed within the 1% report tolerance, with no resolution changes. All 18 separate norm-repeat checks passed; the largest relative difference was $9.447\times10^{-7}$ against a tolerance of $10^{-6}$. Known accuracy flags and the expected radial $p=16$ failure were preserved. Apple Silicon checks of continuation and tubes produced 479 matching comparisons with no resolution changes.

These checks cover selected cases; they are neither a rerun of the complete **257-command campaign** nor an audit of omitted raw archives. See [Validation Details](docs/validation.md) for the tested environments, retained flags and validation commands. Reports distinguish completed calculations from resolved measurements. The implementation uses dense algebra and is intended for these research experiments; it does not provide a scalable solver or GPU/3D implementation.

## Citation

Erik Burman, Peter Hansbo, Mats G. Larson and Karl Larsson. *Numerical Companion to One-Sided p-Explicit Stability for High-Order Cut Finite Element Nitsche Methods* (2026). DOI: [10.5281/zenodo.23205908](https://doi.org/10.5281/zenodo.23205908).

Machine-readable citation metadata are provided in [CITATION.cff](CITATION.cff). Repository: [KarlLarssonUmU/one-sided-hp-cutfem](https://github.com/KarlLarssonUmU/one-sided-hp-cutfem).

## Licenses

Code and software documentation use **MIT**; numerical data and the Numerical Verification Guide use **CC BY 4.0**. These licenses apply to different parts of the repository. See [LICENSE](LICENSE) for their scope and attribution requirements. Dependencies retain their own licenses. The separately maintained main manuscript is outside this repository's license grant.

## Usage of Generative AI

OpenAI GPT-6 Astra was used under author direction to assist with planning numerical experiments, developing and checking numerical software, preparing figures, and writing and revising the repository documentation and accompanying Numerical Verification Guide. The authors reviewed and revised the AI-assisted material and take full responsibility for the code, numerical results, and documentation. The validation procedures, their scope, and known numerical limitations are documented in this repository and the guide.
