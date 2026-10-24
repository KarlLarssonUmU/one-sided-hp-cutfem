"""Vector binary64 pairs for the bounded, normal-range field evaluator.

No Decimal operations occur in add/multiply or in the online evaluator.
Error-free transforms follow Hida, Li and Bailey, algorithms 3, 5 and 6:
https://www.davidhbailey.com/dhbpapers/qd.pdf
Overflow, subnormal products, and arbitrary exponent ranges are not supported.
"""
from decimal import Decimal, localcontext
import numpy as np


def two_sum(a, b):
    s = a + b
    v = s - a
    return s, (a - (s - v)) + (b - v)


def two_product(a, b):
    p = a * b
    ca = 134217729.0 * a
    cb = 134217729.0 * b
    ah = ca - (ca - a)
    bh = cb - (cb - b)
    al, bl = a - ah, b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def add(a, b):
    s, e = two_sum(a[0], b[0])
    t, f = two_sum(a[1], b[1])
    s, e = two_sum(s, e + t)
    return two_sum(s, e + f)


def neg(a):
    return -a[0], -a[1]


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    p, e = two_product(a[0], b[0])
    e = e + (a[0] * b[1] + a[1] * b[0])
    return two_sum(p, e + a[1] * b[1])


def split(values):
    """Offline conversion; enough guard digits for this normal-range data."""
    values = np.asarray(values, dtype=object)
    high = np.array([float(v) for v in values.flat]).reshape(values.shape)
    with localcontext() as ctx:
        ctx.prec = 1100
        low = np.array([float(v - Decimal.from_float(float(h)))
                        for v, h in zip(values.flat, high.flat)]).reshape(values.shape)
    assert np.all(np.isfinite(high)) and np.all(np.isfinite(low))
    return high, low


def exact(high, low):
    """Oracle/output inspection only, never called by online evaluation."""
    with localcontext() as ctx:
        ctx.prec = 1100
        return Decimal.from_float(float(high)) + Decimal.from_float(float(low))


def self_check():
    with localcontext() as ctx:
        ctx.prec = 1100
        values = [Decimal(0), Decimal(1), Decimal(-1), Decimal(1) + Decimal(2)**-54,
                  Decimal('1.23456789012345678901234567890123456789'),
                  Decimal('-0.987654321098765432109876543210987654321'),
                  Decimal('1e-12'), Decimal('1e12')]
        worst = Decimal(0)
        count = 0
        for x in values:
            for y in values:
                a, b = split(np.array(x)), split(np.array(y))
                xx, yy = exact(*a), exact(*b)
                for op, target, scale in ((add, xx + yy, max(abs(xx), abs(yy), Decimal('1e-100'))),
                                           (mul, xx * yy, max(abs(xx * yy), Decimal('1e-100')))):
                    observed = exact(*op(a, b))
                    error = abs(observed - target) / scale
                    assert error < Decimal('1e-30'), (x, y, op.__name__, error)
                    worst = max(worst, error)
                    count += 1
                h, l = two_sum(float(x), float(y))
                assert exact(h, l) == Decimal.from_float(float(x)) + Decimal.from_float(float(y))
                h, l = two_product(float(x), float(y))
                assert exact(h, l) == Decimal.from_float(float(x)) * Decimal.from_float(float(y))
        tiny = Decimal(1) + Decimal(2)**-54
        a = split(np.array(tiny))
        assert exact(*sub(a, (np.array(1.), np.array(0.)))) == Decimal(2)**-54
        assert float(a[0] + a[1]) == 1.0
        return dict(paired_checks=count, maximum_scaled_error=str(worst),
                    small_increment_retained=True, collapse_loses_small_increment=True)
