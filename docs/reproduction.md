# Reproduction Guide

Use the [README](../README.md) for installation, supported platforms and quick-start examples. All commands below run from the repository root with the pinned dependencies installed. The [Numerical Verification Guide](numerical-verification.pdf) explains the methods, triangular basis and accuracy criteria.

## Running Experiments

Use the root entry points documented here. Every calculation uses a new output directory and records parameters, source/input hashes, package versions, numerical checks and completion status. Use normal Python: `-O` disables required assertions and is refused.

Exit **0** means the command completed its required checks; **2** retains a numerical accuracy flag or the attempted radial failure; **1** indicates an execution/check failure. A completed flux command can contain unresolved *unstabilized* spectra, which remain in its results. Process success does not mean that every reported value is resolved.

List baseline cases, inspect the complete plan, or export the reference tables:

```sh
python run.py list baseline
python reproduce.py --output runs/full
python report.py --published --output published
```

The second command prints the plan and resource requests without running calculations. The third writes the reference tables under `published/`. Generated run/report folders, reference exports, Python caches, virtual environments, Git metadata and operating-system metadata are excluded from the distributed file manifest.

## Reproduce Figure 9

In the supported Linux environment, first inspect the plan, then run it inside a suitable CPU allocation:

```sh
python reproduce.py --figure 9 --output runs/figure9
python reproduce.py --figure 9 --output runs/figure9 --execute
python report.py --runs runs/figure9 --output reports/figure9 --plots
```

The plan contains **22 commands**: 14 baseline radial cases (p2–8 on both disks) and eight extensions (p10, p12, p14 and p16). It retains the flagged p14 fields and the p16 failures. The report regenerates the measurements, radial fits and plots for both disks; figure styling is independent of the manuscript's layout. This takes longer than the small quick-start example in the README. The paper's fit window uses exactly p4,5,6,7,8,10,12; unresolved points never enter the fit.

## Complete Campaign

```sh
python reproduce.py --output runs/full --threads 8 --workers 3 --execute
python reproduce.py --output runs/full --threads 8 --workers 3 --execute --resume
python report.py --runs runs/full --output reports/full --require-complete --plots
```

These are **257 sequential commands**: 118 standard PDE cases, 17 Schwarz attempts (16 selected plus one superseded attempt), four accurate cases, 96 flux selections, eight fitted/extension selections, eight radial extensions, three diagnostics and the tube, continuation and conditioning studies. The 17 Schwarz attempts produce 46 comparison rows; 44 enter the selected table. The eight fitted/extension selections produce eight fitted controls and 16 extension norms.

`--study baseline`, `--study accurate`, etc. restrict the plan. `--case baseline/PAPER-059` selects an exact plan ID; repeat `--case` to select several. Dependencies are included. A diagnostic study also regenerates its six radial operator inputs. The plan is finite, visible and editable through the parameters in [data/cases.json](../data/cases.json).

These are dense research implementations. Small checks take seconds to minutes; the largest N24/p8 cases previously requested **8 CPUs, 64 GiB and two hours each**. A complete reproduction takes many hours and generates large matrix files. Allocate sufficient memory and runtime, or run individual cases as separate scheduled jobs. `--threads` limits BLAS threads; accurate assembly uses `--workers` and one BLAS thread per worker. More CPU threads are not guaranteed to improve runtime. No scheduler jobs are submitted by these scripts.

`--resume` skips only runs with completion code 0/2, matching source/input hashes and the same recorded command. It refuses incomplete directories, changed inputs and unrecognized prior runs. Inspect and preserve an interrupted attempt before moving its directory aside and resuming. An execution failure stops the campaign; an accuracy flag remains in the results and the campaign continues.

## Results and Reference Data

[report.py](../report.py) writes `measurements.csv`, `failed-criteria.csv`, `fits.csv`, `comparison.csv` and `summary.json`. Optional plots show regenerated convergence data, including flagged measurements. Logarithmic mesh fits use consecutive resolved levels; radial fits use the specified sampled degrees only. Missing cases and incomplete runs are explicitly reported. `--require-complete` returns 2 if reference observations are missing. A comparison difference is reported, not silently discarded or treated as a new scientific acceptance test.

[data/published.json.gz](../data/published.json.gz) contains 4,959 reference observations, 239 failed criteria, 22 reference tables and their source provenance hashes. The `radial-fits` table includes the six extended `p4-12` rows quoted in Figure 9; their derivation is recorded with the reference data. Export it with `report.py --published`: its `published-tables/` directory includes the cut, smooth/refined, high-degree, radial, Schwarz, cost, flux and geometry tables. Recorded times describe the original machine and allocation; they are not reproducible numerical constants. Failed or superseded attempts are identified in the tables. Unrelated execution logs and the interrupted p10 launch are omitted; the recovered p10 calculation is included.

The six compressed fields in `data/references/` enter **validation only**, not assembly or solution. Their coefficients are stored exactly, with SHA256 checks of mesh/basis identity. Two ordinary p8 arrays support the precision comparison. The accurate solver recomputes its moments, operators and coefficients. Stored references share the established analytic implementation and do not constitute an independent implementation of assembly.

Fresh measurements, tables and optional plots are generated by the commands above. Comparison files identify differences from the reference results; they do not modify the manuscript.

## Studies in Guide Order

### Continuation Barrier

```sh
python run.py continuation --output runs/continuation
```

Recomputes p1–14 with independent 60/80-digit repeats and direct verification of the Chebyshev formula. Output: `output/continuation-sharp.csv`. This is the neighboring-interval obstruction, not the stabilized flux constant.

### Standard PDE Cases and Accuracy Criteria

```sh
python run.py baseline PAPER-004 --output runs/cut
python run.py baseline PAPER-059 --output runs/smooth-reference
```

The 118 cases comprise **48 cut, 56 smooth and 14 radial** configurations. The smooth set includes 16 additional mesh-refinement cases. The drivers use the case-specific assembly/error boosts, six refinement corrections, basis and conformity checks, independent norm checks and selected reference comparisons. The additional refinement cases omit spectral calculations, as specified by their experiment recipes.

In the original 102-case set, all configurations passed the energy criteria and 100 passed every norm. Both smooth p8 L2/gradient flags are retained. The original failed solver-stopping attempt is also included in the Schwarz comparisons.

Field agreement must be within 1% of discretization error in the relevant norm; independent norm integration must agree to 1e-4. Exact per-case limits are saved with each result. Paired agreement cannot exclude shared bias. Energy components are weighted squares: their square roots must not be added. Reference tables: `cut-cases`, `smooth-case-summary`, `standard-case-summary`, `physical-error-components`, `radial-cases`, and their fits.

### Higher-Degree Accuracy and Precision Limits

```sh
python run.py accurate p8-fixed_A --workers 3 --output runs/accurate-p8-A
```

Four selections: p8 on disks A/B, p10/p12 on A. Analytic moments and projection complements use 80 digits, operator contractions 100 digits, Taylor orders 32/40, and 24 corrections with 100-digit residuals. Binary64 LU supplies the correction solves; high/low coefficients remain separate. Online probes and coefficient-field norm identities have distinct scopes. The p12 ordinary component-repeat flags are retained even though the accurate physical fields resolve. Reference tables: `high-p-norms` and `high-p-summary`.

### Finite Regularity in the Full Energy Norm

The 14 baseline radial cases retain the original rules and criteria. They are distinct from the later extensions below. Reference tables: `radial-cases` and `radial-fits`. Fits never include unresolved values.

### A Compact Schwarz Comparison

```sh
python run.py schwarz PAPER-004 --operator runs/cut --output runs/schwarz
```

The supplied operator must match the baseline case and complete mesh/basis identity. `PAPER-075-original` reproduces the earlier two-row attempt; `PAPER-075` uses the retained tighter stopping rule. Exact interior solves and stopping relative to a known direct field make this a bounded comparison, not a scalable production-solver demonstration. Preconditioner depth differs from stabilization depth. Reference tables: `schwarz`, `schwarz-attempts`, `selected-production-costs`.

### Direct Flux Spectra and Matched Thresholds

```sh
python run.py flux --p 2 --path axis --epsilon 0.00003 --output runs/flux
```

p1–8 × axis/diagonal/oblique × offsets 0.03/0.003/0.0003/0.00003 gives 96 selections. Each includes unit, computational and zero stabilization. The p8 quadrature settings are used at every degree; no metric is regularized. All 192 stabilized spectra passed; 70 of 96 unstabilized spectra were unresolved. Positive-integral lower witnesses are retained. Reference tables: `direct-flux`, `flux-threshold-pairs`.

### Fitted Polygon and Extension Seminorms

```sh
python run.py controls --p 2 --output runs/controls
```

One fitted regular 20-gon and two prescribed-extension norms for each p1–8. The mesh is regenerated with the pinned SciPy/Qhull versions and saved with the results; the reference connectivity is in [data/fitted-geometry.json](../data/fitted-geometry.json). The polygon differs from the circle, so this is a geometry-dependent comparison. Reference tables: `fitted-baseline`, `extension-seminorm`.

### Radial Degree Extensions and Arithmetic Limits

```sh
python run.py radial --p 10 --disk fixed_A --output runs/radial-p10-A
python run.py radial --p 16 --disk fixed_A --output runs/radial-p16-A
python run.py diagnostics --p 16 --output runs/diagnostics-p16
```

N4, p10/12/14/16, disks A/B; assembly boosts 16/24 and error boosts 24/32. The squared-radial Duffy rule is checked independently with Gauss–Jacobi integration. In the recorded results, p10/12 resolve, p14 fails field checks, and p16 fails during basis Cholesky before assembly. Degrees 9, 11 and 13 were not sampled. Failure behavior and tiny eigenvalues can vary across platforms.

Diagnostics compare exact rational bubble Grams at MP80/100 and basis evaluation against Decimal100. Add `--matrix runs/radial-p10-A/output/p10-fixed_A/q24.npz` to inspect a generated rounded operator; the campaign supplies all six p10/12/14 matrices. These checks do not produce an accurate p14/16 PDE solution. Reference table: `radial-extension`.

### Stabilization and Coefficient Conditioning

```sh
python run.py conditioning --output runs/conditioning
```

Regenerates paired N6/p6 operators at the axis offset 3e-5, patch depths 2/4 and weights 0.1/1. Matrix conditioning improves in these four controls. Changed-stabilization PDE error norms are not validated by this test. Output: `output/results.json`.

### Exact Tube Certificates

See §A.5, “Exact Tube Certificates,” in the [Numerical Verification Guide](numerical-verification.pdf).

```sh
python run.py tubes --seed-audit --output runs/tubes
python run.py tubes --certificates runs/tubes/output/certificates --output runs/tube-verification
```

[data/geometries.json](../data/geometries.json) gives 30 geometry recipes and the 11 earlier counterexample points. The first command rebuilds the actual patch families, generates all full-arc certificates, independently verifies them, and reproduces the earlier natural-assignment bounds at the tested depths. The second verifies the generated certificates without running the assignment search. `--geometry geometry-02` selects a single geometry.

The `tube_h_over_16_certified` column in `tube-summary` refers to the conservative seed test: it passes two two-layer and six four-layer geometries. Exact seed containment passes six geometries in each group, as recorded by `natural_failed_cells = 0` in `patch-assignment-summary`. For example, `geometry-01` fails the conservative test but passes exact seed containment. A false conservative-test result is not proof that a tube leaves its seed patch.

All 5,324 angular cells are certified at depth h/16 after 97 reassignments within existing patches. The independent checker validates event polynomials, root brackets, full segment coverage and patch membership using exact fractions. Generated certificates live in `output/certificates/`; earlier bounds in `output/seed-assignment.json`. These finite results do not prove uniformity for arbitrary meshes or cuts. Reference tables: `patch-assignment-summary`, `tube-summary`.
