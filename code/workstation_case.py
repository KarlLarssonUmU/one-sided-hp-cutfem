"""One isolated, fully specified experiment; shared by the campaign runner/tests."""
from __future__ import annotations

import math
import time
import numpy as np
import run_2d_disk_study as fem


DEFAULTS = dict(N=6, p=4, epsilon=0.0003, radius=0.62, path="axis",
                patch_layers=4, quadrature_boost=4, error_boost=12,
                projection_basis="h1_orthonormal", solution_kind="exponential",
                singularity_alpha=2.5, gamma0=0.0, gamma1=0.1, beta0=16.0,
                threshold=False, schwarz=False, kind="solve", threshold_iterations=18,
                pcg_maxiter=3000)


class ResourceLimit(RuntimeError):
    pass


def normalize_case(case):
    unknown = set(case) - set(DEFAULTS) - {"center", "label"}
    if unknown:
        raise ValueError(f"Unknown case parameters: {sorted(unknown)}")
    c = {**DEFAULTS, **case}
    for key in ("N", "p", "patch_layers", "threshold_iterations", "pcg_maxiter"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("quadrature_boost", "error_boost"):
        if type(c[key]) is not int or c[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("epsilon", "radius", "gamma0", "gamma1", "beta0", "singularity_alpha"):
        if not math.isfinite(c[key]):
            raise ValueError(f"{key} must be finite")
    if c["radius"] <= 0 or c["epsilon"] < 0 or c["gamma0"] < 0 or c["gamma1"] <= 0 or c["beta0"] <= 0:
        raise ValueError("Invalid geometry/stabilization parameters")
    if c["projection_basis"] not in ("scaled_legendre", "h1_orthonormal"):
        raise ValueError("Unknown projection_basis")
    if c["solution_kind"] not in ("exponential", "radial_power"):
        raise ValueError("Unknown solution_kind")
    if c["solution_kind"] == "radial_power" and c["singularity_alpha"] <= 1:
        raise ValueError("Radial-power forcing must be in L2 (alpha > 1)")
    if c["kind"] not in ("solve", "projector_compare"):
        raise ValueError("Unknown case kind")
    if c["path"] not in ("axis", "diagonal", "oblique", "fixed_A", "fixed_B"):
        raise ValueError("Unknown path")
    if "center" in c and (len(c["center"]) != 2 or not np.isfinite(c["center"]).all()):
        raise ValueError("center must contain two finite coordinates")
    return c


def geometry(c):
    mesh = fem.build_mesh(c["N"])
    if "center" in c:
        center = np.array(c["center"], dtype=float)
    elif c["path"].startswith("fixed_"):
        center = np.array({"fixed_A": [0.071, -0.043], "fixed_B": [-0.113, 0.087]}[c["path"]])
    else:
        directions = {"axis": [1., 0.], "diagonal": [1., 1.], "oblique": [1., 2.]}
        direction = np.array(directions[c["path"]])
        direction /= np.linalg.norm(direction)
        target = np.array([1/3, 0 if c["path"] == "axis" else 1/3])
        candidates = []
        for vertex in mesh.vertices:
            center = vertex - (c["radius"] - c["epsilon"] * mesh.h) * direction
            if np.max(np.abs(center)) + c["radius"] < 0.995:
                candidates.append((float(np.linalg.norm(vertex-target)), center.copy()))
        if not candidates:
            raise ValueError("No admissible tangent vertex")
        center = min(candidates, key=lambda item: item[0])[1]
    if np.max(np.abs(center)) + c["radius"] > 0.999:
        raise ValueError("Disk leaves background square")
    active, _, _ = fem.classify_elements(mesh, center, c["radius"])
    keys = set()
    for eid in active:
        keys.update(fem.local_dof_keys(c["p"], mesh.elements[eid]))
    return mesh, center, len(keys)


def assemble(c, center, basis=None):
    return fem.assemble_problem(
        c["N"], c["p"], c["epsilon"], radius=c["radius"],
        patch_layers=c["patch_layers"], center_override=center,
        quadrature_boost=c["quadrature_boost"],
        projection_basis=basis or c["projection_basis"],
        solution_kind=c["solution_kind"], singularity_center=center,
        singularity_alpha=c["singularity_alpha"])


def relative_difference(a, b):
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-300))


def checked_pcg(A, b, B, maxiter=3000, tolerance=1e-9):
    """Report the true residual and explicitly stop on nonpositive curvature."""
    x = np.zeros_like(b)
    r = b.copy()
    norm0 = float(np.linalg.norm(b))
    if norm0 == 0:
        return dict(iterations=0, true_residual=0., converged=True, breakdown=None)
    z = B @ r
    d = z.copy()
    rz = float(r @ z)
    breakdown = None
    iteration = 0
    for iteration in range(1, maxiter+1):
        Ad = A @ d
        curvature = float(d @ Ad)
        if not np.isfinite([curvature, rz]).all() or min(curvature, rz) <= 0:
            breakdown = "nonpositive_or_nonfinite_curvature"
            break
        alpha = rz / curvature
        x += alpha * d
        r -= alpha * Ad
        if np.linalg.norm(r) <= tolerance * norm0:
            # A recursive residual can drift on poorly conditioned cut systems.
            r = b - A @ x
            if np.linalg.norm(r) <= tolerance * norm0:
                break
            z = B @ r
            d = z.copy()
            rz = float(r @ z)
            continue
        z = B @ r
        rz_new = float(r @ z)
        d = z + (rz_new / rz) * d
        rz = rz_new
    true_residual = float(np.linalg.norm(b-A@x)/norm0)
    return dict(iterations=iteration, true_residual=true_residual,
                converged=bool(true_residual <= tolerance and breakdown is None),
                breakdown=breakdown)


def execute(case, max_dofs=6000, max_estimated_gib=8.0):
    c = normalize_case(case)
    start = time.perf_counter()
    mesh, center, ndof = geometry(c)
    # Planning estimate, not a hard RSS limit: dense temporaries and LAPACK work.
    estimated_gib = 40 * 8 * ndof**2 / 2**30
    if ndof > max_dofs or estimated_gib > max_estimated_gib:
        raise ResourceLimit(f"ndof={ndof}, dense estimate={estimated_gib:.2f} GiB; "
                            f"limits={max_dofs} dofs/{max_estimated_gib} GiB")
    timings = {}
    tick = time.perf_counter()
    problem = assemble(c, center)
    timings["assembly"] = time.perf_counter() - tick
    r = dict(h=mesh.h, ndof=ndof, estimated_dense_gib=estimated_gib,
             center_x=float(center[0]), center_y=float(center[1]),
             eta_min=problem["eta_min"], active_elements=len(problem["active"]),
             cut_elements=len(problem["cut"]), patches=len(problem["patches"]),
             volume_points=sum(len(w) for _, _, w in problem["volume_groups"]),
             boundary_points=sum(len(w) for _, _, _, w in problem["boundary_groups"]))
    warnings = []
    areas = fem.geometry_area_diagnostic(mesh, center, c["radius"], problem["active"])
    area_T = .5 * mesh.h**2
    r["domain_area_relative_error"] = abs(sum(problem["area"].values())-math.pi*c["radius"]**2)/(math.pi*c["radius"]**2)
    r["max_cell_area_error_over_full_cell"] = max(abs(problem["area"][i]-areas[i])/area_T for i in problem["active"])
    fractions = [areas[i]/area_T for i in problem["cut"]]
    r["cut_fractions_at_diagnostic_floor"] = sum(a <= 1e-13 for a in fractions)
    r["min_cut_fraction_unfiltered"] = min(fractions)
    if r["max_cell_area_error_over_full_cell"] > 1e-7:
        warnings.append("cell_area_quadrature_unresolved")
    if r["min_cut_fraction_unfiltered"] <= 1e-12:
        warnings.append("geometry_near_diagnostic_floor")
    diagnostics = problem["patch_diagnostics"]
    for key in ("raw_saddle_condition", "raw_augmented_condition", "projection_condition", "orthogonality_error"):
        r[key + "_max"] = max(d[key] for d in diagnostics)
    r["s1_raw_negative_relative_max"] = max(max(0., -d["s1_raw_min_eigenvalue"])/d["s1_raw_spectral_scale"] for d in diagnostics)
    r["s1_discarded_absolute_trace_sum"] = sum(d["s1_discarded_absolute_trace"] for d in diagnostics)
    r["s1_clipped_modes_sum"] = sum(d["s1_clipped_modes"] for d in diagnostics)
    if r["s1_raw_negative_relative_max"] > 1e-9:
        warnings.append("stabilization_cleanup_material")
    if r["orthogonality_error_max"] > 1e-8:
        warnings.append("projector_orthogonality_lost")
    A, metric, rhs = fem.matrices_for(problem, c["gamma0"], c["beta0"], c["gamma1"])
    tick = time.perf_counter()
    coeff = np.linalg.solve(A, rhs)
    r["solve_relative_residual"] = float(np.linalg.norm(A@coeff-rhs)/max(np.linalg.norm(rhs), 1e-300))
    r["solve_backward_error"] = float(np.linalg.norm(A@coeff-rhs)/(np.linalg.norm(A)*np.linalg.norm(coeff)+np.linalg.norm(rhs)))
    timings["solve"] = time.perf_counter()-tick
    if r["solve_relative_residual"] > 1e-8:
        warnings.append("direct_solve_residual_large")
    tick = time.perf_counter()
    l2, h1 = fem.solution_errors(problem, coeff)
    error_problem = dict(problem)
    error_order = max(c["error_boost"], c["quadrature_boost"]+4)
    error_problem["volume_groups"], _ = fem.volume_quadrature_groups(
        mesh, center, c["radius"], c["p"], problem["active"], error_order)
    r["l2_error"], r["h1_error"] = fem.solution_errors(error_problem, coeff)
    r["error_evaluation_boost"] = error_order
    r["error_quadrature_relative_change"] = max(abs(l2-r["l2_error"])/max(r["l2_error"], 1e-14),
                                                abs(h1-r["h1_error"])/max(r["h1_error"], 1e-14))
    if r["error_quadrature_relative_change"] > 1e-3 and min(r["l2_error"], r["h1_error"]) > 1e-10:
        warnings.append("error_integral_needs_refinement")
    if min(r["l2_error"], r["h1_error"]) < 1e-10:
        warnings.append("error_near_precision_floor")
    del error_problem
    timings["error_evaluation"] = time.perf_counter()-tick
    tick = time.perf_counter()
    r["coercivity_at_beta"] = fem.generalized_min_eigenvalue(A, metric)
    if r["coercivity_at_beta"] <= 1e-8:
        warnings.append("chosen_penalty_not_coercive")
    if c["threshold"]:
        r["beta0_threshold"] = fem.coercivity_threshold(problem, c["gamma0"], c["gamma1"], iterations=c["threshold_iterations"])
    timings["coercivity"] = time.perf_counter()-tick
    if c["schwarz"] and r["coercivity_at_beta"] > 1e-8:
        tick = time.perf_counter()
        jacobi = np.diag(1. / np.maximum(np.diag(A), 1e-30))
        schwarz = fem.build_patch_schwarz_inverse(A, problem["patch_dofs"])
        timings["preconditioner_setup"] = time.perf_counter()-tick
        tick = time.perf_counter()
        r["condition_jacobi"] = fem.preconditioned_condition(A, jacobi)
        r["condition_patch_schwarz"] = fem.preconditioned_condition(A, schwarz)
        for name, B in (("jacobi", jacobi), ("schwarz", schwarz)):
            result = checked_pcg(A, rhs, B, c["pcg_maxiter"])
            r.update({f"pcg_{name}_{k}": v for k, v in result.items()})
            if not result["converged"]:
                warnings.append(f"pcg_{name}_not_converged")
        timings["spectra_and_pcg"] = time.perf_counter()-tick
    if c["kind"] == "projector_compare":
        tick = time.perf_counter()
        other_basis = "scaled_legendre" if c["projection_basis"] == "h1_orthonormal" else "h1_orthonormal"
        other = assemble(c, center, other_basis)
        A_other, _, rhs_other = fem.matrices_for(other, c["gamma0"], c["beta0"], c["gamma1"])
        for key in ("S1", "S0"):
            r[f"relative_{key}_basis_difference"] = relative_difference(problem[key], other[key])
        r["relative_A_basis_difference"] = relative_difference(A, A_other)
        r["relative_coeff_basis_difference"] = relative_difference(coeff, np.linalg.solve(A_other, rhs_other))
        if r["relative_A_basis_difference"] > 1e-8:
            warnings.append("projector_realizations_disagree")
        timings["projector_comparison"] = time.perf_counter()-tick
    r["timings_seconds"] = timings
    r["elapsed_seconds"] = time.perf_counter()-start
    for key, value in list(r.items()):
        if isinstance(value, (float, np.floating)) and not math.isfinite(value):
            r[key] = None
            warnings.append(f"nonfinite_{key}")
    r["review_flags"] = warnings
    r["publication_candidate"] = not warnings
    return r
