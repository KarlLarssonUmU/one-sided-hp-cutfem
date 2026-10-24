"""Offline Decimal export and independent oracles for paired field evaluation."""
from decimal import Decimal as D, localcontext
from math import comb
import numpy as np
import paired_arithmetic as dd
from paired_bernstein_evaluator import indices
import precise_polynomial_identity as identity


def composed(basis, coefficients, degree):
    out = {(i, j): D(0) for i, j in indices(degree)}
    for coefficient, poly in zip(coefficients, basis, strict=True):
        for ij, value in poly.items():
            out[ij] += coefficient*identity.decimal_value(value)
    return out


def bernstein(poly, degree):
    return [sum((value*D(comb(i, k)*comb(j, l))/D(comb(degree, k+l)*comb(k+l, k))
                 for (k, l), value in poly.items() if k <= i and l <= j), D(0))
            for i, j in indices(degree)]


def inverse(J):
    a, b, c, d = (identity.exact_value(v) for v in np.asarray(J).flat)
    determinant = a*d-b*c
    assert determinant > 0
    return [[d/determinant, -b/determinant], [-c/determinant, a/determinant]], determinant


def physical_polynomials(poly, inv):
    dr = {(i-1, j): i*v for (i, j), v in poly.items() if i}
    ds = {(i, j-1): j*v for (i, j), v in poly.items() if j}
    result = [poly]
    for k in range(2):
        result.append({ij: inv[0][k]*dr.get(ij, D(0))+inv[1][k]*ds.get(ij, D(0))
                       for ij in set(dr) | set(ds)})
    return result


def export_fields(mesh, space, retained, paired):
    """Global-pair storage, composition and Bernstein rounding remain distinct."""
    import extended_triangle_norms_2d as triangle
    rows = []
    origins, inverses, controls = [], [], []
    polys = []
    with localcontext() as ctx:
        ctx.prec = 100
        storage_squared = [D(0), D(0)]
        export_squared = [D(0), D(0)]
        for eid in space['active']:
            e = mesh.elements[eid]
            dofs = space['dofs'][eid]
            original = composed(space['basis'][eid], [retained[i] for i in dofs], space['p'])
            candidate = composed(space['basis'][eid], [paired[i] for i in dofs], space['p'])
            inv, determinant = inverse(e.J)
            physical = physical_polynomials(candidate, inv)
            ctrl = np.array([bernstein(poly, space['p']) for poly in physical], dtype=object)
            h, l = dd.split(ctrl)
            maxima = [max(abs(dd.exact(a, b)-v) for a, b, v in zip(aa, bb, cc))
                      for aa, bb, cc in zip(h, l, ctrl)]
            export_squared[0] += determinant/2*maxima[0]**2
            export_squared[1] += determinant/2*(maxima[1]**2+maxima[2]**2)
            delta = {ij: candidate[ij]-original[ij] for ij in original}
            norms = triangle.squared_norms(e, delta, 100)
            for k in range(2):
                storage_squared[k] += norms[k]
            origins.append(e.xy[0])
            inverses.append(inv)
            controls.append(ctrl)
            polys.append(original)
            rows.append(dict(element=int(eid), control_rounding_max=list(map(str, maxima))))
        ih, il = dd.split(np.array(inverses, dtype=object))
        ch, cl = dd.split(np.array(controls, dtype=object))
        data = dict(degree=np.array(space['p']), elements=np.array(space['active']),
                    origins=np.asarray(origins, dtype=float), inverse_high=ih, inverse_low=il,
                    controls_high=ch, controls_low=cl)
        bounds = dict(global_storage_full_triangle_bounds=[str(v.sqrt()) for v in storage_squared],
                      control_export_full_triangle_bounds=[str(v.sqrt()) for v in export_squared],
                      elements=rows)
    return data, polys, bounds


def decimal_horner(poly, degree, r, s):
    """Independent power-basis value/reference derivatives, all Decimal."""
    value = dr = ds = D(0)
    for i in range(degree, -1, -1):
        row = drow = D(0)
        for j in range(degree-i, -1, -1):
            drow = drow*s+row
            row = row*s+poly.get((i, j), D(0))
        dr = dr*r+value
        value = value*r+row
        ds = ds*r+drow
    return value, dr, ds


def oracle(poly, degree, origin, J, points, digits=100):
    """The stored physical points are interpreted exactly; no float subtraction."""
    with localcontext() as ctx:
        ctx.prec = digits
        inv, _ = inverse(J)
        x0, y0 = map(identity.exact_value, origin)
        out = []
        for x, y in points:
            dx, dy = identity.exact_value(x)-x0, identity.exact_value(y)-y0
            r, s = (row[0]*dx+row[1]*dy for row in inv)
            value, dr, ds = decimal_horner(poly, degree, r, s)
            out.append([value, inv[0][0]*dr+inv[1][0]*ds,
                        inv[0][1]*dr+inv[1][1]*ds])
        return np.array(out, dtype=object).T


def oracle_controls(poly, element, points, degree):
    """Direct monomial sum and pivoted geometry vs Horner and inverse mapping."""
    with localcontext() as ctx:
        ctx.prec = 100
        observed = oracle(poly, degree, element.xy[0], element.J, points)
        worst = D(0)
        for q, point in enumerate(points):
            r, s, invT = identity.decimal_geometry(element, point)
            v, dr, ds = identity.evaluate_decimal(poly, r, s)
            direct = [v, *[row[0]*dr+row[1]*ds for row in invT]]
            worst = max(worst, max(abs(a-b) for a, b in zip(observed[:, q], direct)))
        assert worst < D('1e-80'), worst
        return str(worst)


def discrepancies(high, low, reference):
    with localcontext() as ctx:
        ctx.prec = 100
        difference = np.empty(high.shape)
        for ix in np.ndindex(high.shape):
            difference[ix] = float(dd.exact(high[ix], low[ix])-reference[ix])
        return difference
