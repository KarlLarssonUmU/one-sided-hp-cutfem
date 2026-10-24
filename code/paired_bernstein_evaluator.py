"""Online paired arithmetic for exported element fields and physical gradients.

Input points are binary64 physical coordinates. Return separate high/low arrays
with shape (3, number_of_points): value, x derivative, y derivative.
All online operations are NumPy binary64; Decimal is confined to export/oracles.
"""
from functools import lru_cache
import numpy as np
import paired_arithmetic as dd


def indices(degree):
    return [(i, t-i) for t in range(degree+1) for i in range(t+1)]


@lru_cache(maxsize=32)
def reduction(degree):
    positions = {ij: k for k, ij in enumerate(indices(degree))}
    lower = indices(degree-1)
    return tuple(np.array([positions[i+di, j+dj] for i, j in lower])
                 for di, dj in ((0, 0), (1, 0), (0, 1)))


def reference_coordinates(origin, inverse_high, inverse_low, points):
    points = np.asarray(points, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 2 and np.all(np.isfinite(points))
    dx = dd.two_sum(points[:, 0], -origin[0])
    dy = dd.two_sum(points[:, 1], -origin[1])
    result = []
    for row in range(2):
        result.append(dd.add(dd.mul((inverse_high[row, 0], inverse_low[row, 0]), dx),
                             dd.mul((inverse_high[row, 1], inverse_low[row, 1]), dy)))
    return result


def casteljau(high, low, degree, r, s):
    a = dd.sub(dd.sub((1., 0.), r), s)
    values = high[:, :, None], low[:, :, None]
    for n in range(degree, 0, -1):
        zero, dr, ds = reduction(n)
        terms = [dd.mul(weight, (values[0][:, ix, :], values[1][:, ix, :]))
                 for weight, ix in ((a, zero), (r, dr), (s, ds))]
        values = dd.add(dd.add(terms[0], terms[1]), terms[2])
    return values[0][:, 0, :], values[1][:, 0, :]


def evaluate_element(export, element, points, chunk_size=256, collapsed=False):
    """Return paired values without combining the two components."""
    points = np.asarray(points, dtype=np.float64)
    out = [np.empty((3, len(points))) for _ in range(2)]
    degree = int(export['degree'])
    for start in range(0, len(points), chunk_size):
        stop = min(start+chunk_size, len(points))
        r, s = reference_coordinates(export['origins'][element],
                                     export['inverse_high'][element],
                                     export['inverse_low'][element], points[start:stop])
        high, low = export['controls_high'][element], export['controls_low'][element]
        if collapsed:
            # Explicit negative control: ordinary binary64 coordinates/controls.
            r, s = r[0]+r[1], s[0]+s[1]
            values = (high+low)[:, :, None]
            a = 1.-r-s
            for n in range(degree, 0, -1):
                zero, dr, ds = reduction(n)
                values = a*values[:, zero, :]+r*values[:, dr, :]+s*values[:, ds, :]
            got = values[:, 0, :], np.zeros((3, stop-start))
        else:
            got = casteljau(high, low, degree, r, s)
        for output, value in zip(out, got):
            output[:, start:stop] = value
    assert all(np.all(np.isfinite(v)) for v in out)
    return tuple(out)
