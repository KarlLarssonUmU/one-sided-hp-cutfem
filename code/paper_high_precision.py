"""Opt-in accurate assembly and paired field output for the triangular paper.

The ordinary exponential Poisson problem, conforming space, patch inventory,
Nitsche scaling and projection residual are unchanged. No reference coefficients
enter assembly or solve. Decimal is an assembly/export cost; online fields keep
two separate binary64 arrays. This is a dense research implementation.
"""
from decimal import Decimal as D, localcontext
from pathlib import Path
import multiprocessing as mp
import time
import numpy as np
from scipy.linalg import lu_factor, lu_solve
import paper_solver_2d as ordinary
import paper_energy as energy
import extended_case_common as common
import precise_projection_2d as projection
import analytic_cut_operator_2d as operator
import analytic_exponential_load_v2_2d as loads
import extended_operator_precision_2d as precision
import extended_coefficient_storage as storage
import paired_field_export as exporter
import paired_bernstein_evaluator as evaluator
import decimal_retained_system as ds


def sparse_stabilization(mesh, space, patches, workers):
    """Same local Decimal projector; accumulate sparse rows without zero fill.

    All local polynomial columns are checked. Global conformity and coefficient
    identities are inherited from element_data; no small entries are dropped.
    """
    projection._DATA, P, checks = projection.element_data(mesh, space)
    _, zero, unit = projection.patch_result((-1, [space['active'][0]]))
    with localcontext() as ctx:
        ctx.prec = 80
        scale = max(map(abs, projection._DATA[space['active'][0]]['K'].ravel()))
        unit['relative_zero_matrix'] = str(max(map(abs, zero.ravel())) / scale)
    assert D(unit['relative_zero_matrix']) < D('1e-40')
    rows = [{} for _ in range(len(P))]
    diagnostics = []
    with localcontext() as ctx:
        ctx.prec = 80
        with mp.get_context('fork').Pool(workers) as pool:
            for dofs, S, info in pool.imap(projection.patch_result, enumerate(patches)):
                for i, gi in enumerate(dofs):
                    row = rows[int(gi)]
                    for j in np.flatnonzero(S[i]):
                        gj = int(dofs[j])
                        row[gj] = row.get(gj, D(0)) + S[i, j]
                diagnostics.append(info)
                print('Accurate paper patch', info['index'], flush=True)
    checks.update(single_triangle=unit, patches=diagnostics, workers=workers,
                  kernel_check_scope='All local polynomial columns; global shared identities',
                  omitted_small_entries=False)
    return [[[j, str(v)] for j, v in sorted(row.items()) if v] for row in rows], checks


class Assembly:
    def __init__(self, mesh, space, center, case, workers, gate, directory):
        assert case.get('solution_kind', 'exponential') == 'exponential'
        assert case['gamma0'] == 0 and case['p'] >= 3
        tick = time.perf_counter()
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        moments, geometry = common.moments(mesh, space, center, case['radius'], workers, gate)
        common.write(directory/'analytic-moments.json.gz', moments)
        boundary = sorted(z['element'] for z in moments if D(z['geometry']['circle_length']) > 0)
        import run_2d_disk_study as fem
        self.patches = fem.build_patches(mesh, space['active'], boundary, case['patch_layers'])
        S, patch_checks = sparse_stabilization(mesh, space, self.patches, workers)
        for key in ('coefficient_identity_maximum', 'shared_coefficient_maximum'):
            gate(key, D(patch_checks[key]), 1e-55)
        for z in patch_checks['patches']:
            for key in ('Decimal_inverse_error', 'relative_asymmetry', 'relative_polynomial_kernel'):
                gate(f'patch{z["index"]}_{key}', D(z[key]), 1e-45)
        common.write(directory/'analytic-S1-Decimal.json.gz', S)
        with precision.contraction_precision(100):
            blocks, element_checks = operator.assemble(mesh, space, center, case['radius'], moments, workers)
        for z in element_checks:
            gate(f'e{z["element"]}_raw_symmetry', D(z['relative_asymmetry']), 1e-55)
        self.rows, self.polynomial_rhs = operator.combine(blocks, S, mesh.h, case['p'], case['beta0'], case['gamma1'])
        common.write(directory/'analytic-components-Decimal.json.gz',
                     {k: common.sparse(v) if v.ndim == 2 else list(map(str, v)) for k, v in blocks.items()})
        del blocks, S
        self.rhs, self.bounds, polynomial_repeat, saved, load_checks = loads.assemble(
            mesh, space, center, case['radius'], moments, (32, 40), workers, case['beta0'])
        with localcontext() as ctx:
            ctx.prec = 80
            gate('reordered_polynomial_load', max(abs(x-y) for x, y in zip(polynomial_repeat, self.polynomial_rhs)), 1e-55)
            gate('Taylor_load_difference_bounded', max(max(D(0), abs(x-y)-a-b) for x,y,a,b in
                 zip(self.rhs[32], self.rhs[40], self.bounds[32], self.bounds[40])), 1e-55)
        for z in load_checks:
            gate(f'e{z["element"]}_load_moments_repeat', D(z['lower_moment_repeat']), 0)
        common.write(directory/'exponential-moments.json.gz', saved)
        common.write(directory/'retained-system.json.gz', dict(rows=[[[j,str(v)] for j,v in r] for r in self.rows]))
        for q in (32, 40):
            common.write(directory/f'taylor{q}-load-Decimal.json', list(map(str, self.rhs[q])))
        self.metadata = dict(geometry=geometry, stabilization_checks=patch_checks,
                             element_checks=element_checks, load_checks=load_checks,
                             assembly_seconds=time.perf_counter()-tick,
                             local_contraction_digits=100, moments_and_projection_digits=80,
                             known_solution_stabilization_load=False, diagonal_shift=0)


class RetainedSolver:
    """One scaled binary64 LU, with Decimal100 residuals and coefficient updates."""
    def __init__(self, rows):
        self.rows = rows
        n = len(rows)
        A = np.zeros((n, n))
        for i, row in enumerate(rows):
            for j, value in row:
                A[i, j] = float(value)
        assert np.all(np.isfinite(A)) and np.all(np.diag(A) > 0)
        self.scale = 1 / np.sqrt(np.diag(A))
        self.factor = lu_factor(self.scale[:, None] * A * self.scale[None, :], check_finite=True)
        assert np.all(np.diag(self.factor[0]) != 0)

    def solve(self, rhs, steps=24):
        assert steps == 24
        tick = time.perf_counter()
        def correction(values):
            return self.scale * lu_solve(self.factor, self.scale * np.asarray(values, dtype=float))
        x = [D.from_float(float(v)) for v in correction(rhs)]
        history = []
        with localcontext() as ctx:
            ctx.prec = 100
            scale = max(map(abs, rhs))
            assert scale > 0
            for step in range(steps):
                residual = [v-sum((a*x[j] for j,a in row),D(0)) for row,v in zip(self.rows,rhs)]
                delta = correction(residual)
                assert np.all(np.isfinite(delta))
                previous = x
                x = [a+D.from_float(float(b)) for a,b in zip(x,delta)]
                change = max(abs(a-b) for a,b in zip(x,previous))/max(map(abs,x))
                history.append(dict(step=step+1,relative_residual=str(max(map(abs,residual))/scale),coefficient_change=str(change)))
            residual = [v-sum((a*x[j] for j,a in row),D(0)) for row,v in zip(self.rows,rhs)]
            ratio = max(map(abs,residual))/scale
        metrics = dict(relative_infinity_residual=str(ratio),relative_coefficient_change_final=str(change),
                       history=history,steps=steps,residual_digits=100,coefficient_digits=100,
                       correction_dtype='float64',factorization_reused=True,
                       diagonal_shift=0,seconds=time.perf_counter()-tick)
        return np.asarray(x,dtype=object), metrics


class PairedField:
    """Offline composition, online paired Bernstein value and physical gradient."""
    def __init__(self, mesh, space, retained):
        self.high,self.low,self.reconstructed = storage.split(list(retained))
        self.export,self.polynomials,self.bounds = exporter.export_fields(mesh,space,retained,self.reconstructed)
        self.positions = {int(eid):i for i,eid in enumerate(self.export['elements'])}

    def evaluate(self, element_id, points):
        """Return two arrays (3,n): value, dx, dy. Keep them separate downstream."""
        return evaluator.evaluate_element(self.export,self.positions[int(element_id)],points)

    def save(self, path):
        np.savez_compressed(path, **self.export, coefficients_high=self.high, coefficients_low=self.low)


class RetainedFields(energy.Fields):
    """Independent stable error identities, including reconstructed paired fields.

    These are coefficient-field norms; online evaluation is checked separately.
    This avoids subtracting an order-one exponential after binary64 collapse.
    """
    def sample(self, eid, points):
        return self.stable(eid, points)


def difference(a,b):
    with localcontext() as ctx:
        ctx.prec=100
        def convert(values):
            return np.asarray([ds.exact_decimal(x) if not isinstance(x,D) else x for x in values],dtype=object)
        return convert(a)-convert(b)
