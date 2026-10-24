"""Mixed-precision residual refinement of a fixed float64 system.

Only residuals and the working iterate use NumPy longdouble. Correction
systems and returned coefficients use float64. No reference field is needed.
NumPy refactors the scaled matrix for each correction; costs include this.
"""
import time
import numpy as np


def refine(A, b, initial, steps=6):
    assert A.dtype == b.dtype == initial.dtype == np.float64
    assert steps >= 2 and np.finfo(np.longdouble).nmant > np.finfo(float).nmant
    assert A.shape == (len(b), len(b)) and initial.shape == b.shape
    assert np.all(np.diag(A) > 0)
    tick = time.perf_counter()
    ld = np.longdouble
    aa, bb, x = A.astype(ld), b.astype(ld), initial.astype(ld)
    scale = 1 / np.sqrt(np.diag(A))
    scaled = scale[:, None] * A * scale[None, :]
    d = scale.astype(ld)
    anorm = np.max(np.sum(abs(aa), axis=1))
    bnorm = np.max(abs(bb))
    history = []

    def residual():
        r = bb - aa @ x
        return r, dict(relative_rhs_residual=float(np.max(abs(r)) / max(bnorm, ld('1e-300'))),
                       backward_error=float(np.max(abs(r)) / max(anorm*np.max(abs(x))+bnorm, ld('1e-300'))))

    for step in range(steps):
        r, record = residual()
        correction = d * np.linalg.solve(scaled, np.asarray(d*r, dtype=float)).astype(ld)
        previous = x.copy()
        x += correction
        record.update(step=step,update_infinity=float(np.max(abs(correction))))
        history.append(record)
    _, record = residual()
    record['step'] = steps
    history.append(record)
    rounded = x.astype(float)
    rounding = rounded.astype(ld) - x
    final_change = x - previous
    assert np.all(np.isfinite(rounded))
    return rounded, dict(steps=steps,seconds=time.perf_counter()-tick,history=history,
                         matrix_dtype='float64',correction_dtype='float64',
                         residual_dtype='longdouble',working_dtype='longdouble',output_dtype='float64',
                         residual_mantissa_bits=np.finfo(ld).nmant+1,
                         scaling='diagonal',factorization_reused=False), final_change, rounding
