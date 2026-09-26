"""Correctly rounded elementary functions over Rationals.

Each function returns ``x`` rounded to ``digits`` significant decimal digits
(round-half-even), as a :class:`~woz.rational.Rational`. Every digit is
correct: the value is computed as a guaranteed enclosing interval in
binary fixed point, and precision is raised until both ends of the
interval round to the same ``digits``-digit result.

Internally an interval is a pair of ints ``(lo, hi)`` meaning
``lo / 2**w <= true value <= hi / 2**w`` at a working precision of ``w``
bits. All roundings go outward, so the enclosure always holds.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

from woz.bigdec import round_significant, round_to_rational
from woz.rational import Rational, RationalLike

__all__ = [
    "MAX_PRECISION",
    "MIN_PRECISION",
    "PrecisionError",
    "atan",
    "cos",
    "e",
    "exp",
    "ln",
    "pi",
    "sin",
    "sqrt",
]

MIN_PRECISION = 1
MAX_PRECISION = 60

# Input limits. Past these the answer would be enormous or the argument
# reduction would need millions of bits.
_EXP_LIMIT = Rational(100_000)
_TRIG_LIMIT = Rational(10**100)
# Stop raising precision here; only reached if a result can't be settled.
_MAX_WORKING_BITS = 1 << 16

_Interval = tuple[int, int]
_RationalInterval = tuple[Rational, Rational]


class PrecisionError(ArithmeticError):
    """Raised when a result can't be settled within the working-bit limit."""


# --------------------------------------------------------------- utilities


def _check_precision(digits: int) -> None:
    if isinstance(digits, bool) or not isinstance(digits, int):
        raise TypeError("precision must be an int")
    if not MIN_PRECISION <= digits <= MAX_PRECISION:
        raise ValueError(
            f"precision must be between {MIN_PRECISION} and {MAX_PRECISION}, got {digits}"
        )


def _to_rational(x: RationalLike) -> Rational:
    if isinstance(x, bool) or not isinstance(x, (Rational, int)):
        raise TypeError(f"expected a Rational or int, got {type(x).__name__}")
    return Rational(x)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _fixed(x: Rational, w: int) -> _Interval:
    """Enclose the exact Rational ``x`` at ``w`` bits."""
    scaled = x.numerator << w
    return scaled // x.denominator, _ceil_div(scaled, x.denominator)


def _to_rationals(iv: _Interval, w: int) -> _RationalInterval:
    return Rational(iv[0], 1 << w), Rational(iv[1], 1 << w)


def _mul(a: _Interval, b: _Interval, w: int) -> _Interval:
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(products) >> w, -(-max(products) >> w)


def _magnitude(iv: _Interval) -> int:
    return max(abs(iv[0]), abs(iv[1]))


def _small_bits(x: Rational) -> int:
    """Roughly how many leading zero bits ``|x| < 1`` has (0 otherwise)."""
    if x.numerator == 0:
        return 0
    return max(0, x.denominator.bit_length() - abs(x.numerator).bit_length())


def _working_bits(digits: int) -> int:
    # log2(10) < 3.33; the extra bits make a retry unlikely.
    return (digits * 333 + 99) // 100 + 24


def _settle(digits: int, enclose: Callable[[int], _RationalInterval]) -> Rational:
    """Raise precision until the enclosure pins down the rounded result."""
    w = _working_bits(digits)
    while w <= _MAX_WORKING_BITS:
        lo, hi = enclose(w)
        if round_significant(lo, digits) == round_significant(hi, digits):
            return round_to_rational(lo, digits)
        w *= 2
    raise PrecisionError("could not settle the result to the requested precision")


# ------------------------------------------------------------------- series


def _atan_inv(n: int, w: int) -> _Interval:
    """Enclose ``atan(1/n) * 2**w`` for an integer ``n >= 2``."""
    one = 1 << w
    lo = hi = 0
    power = n  # n ** (2j + 1)
    j = 0
    n2 = n * n
    while True:
        # 1 / n**(2j+1), enclosed.
        p_lo, p_hi = one // power, _ceil_div(one, power)
        d = 2 * j + 1
        t_lo, t_hi = p_lo // d, _ceil_div(p_hi, d)
        if j % 2 == 0:
            lo, hi = lo + t_lo, hi + t_hi
        else:
            lo, hi = lo - t_hi, hi - t_lo
        if p_hi <= 1:
            # Alternating series with shrinking terms: the tail is smaller
            # than the next term, which is below one unit here.
            return lo - 1, hi + 1
        power *= n2
        j += 1


@lru_cache(maxsize=32)
def _pi_fixed(w: int) -> _Interval:
    """Enclose ``pi * 2**w`` with Machin's formula."""
    a_lo, a_hi = _atan_inv(5, w + 4)
    b_lo, b_hi = _atan_inv(239, w + 4)
    # pi = 16 atan(1/5) - 4 atan(1/239), computed with 4 extra bits.
    lo = 16 * a_lo - 4 * b_hi
    hi = 16 * a_hi - 4 * b_lo
    return lo >> 4, -(-hi >> 4)


def _exp_series(r: _Interval, w: int) -> _Interval:
    """Enclose ``exp(r)`` for an enclosed ``|r| <= 1``."""
    one = 1 << w
    term: _Interval = (one, one)
    lo = hi = one
    k = 0
    while True:
        k += 1
        t_lo, t_hi = _mul(term, r, w)
        term = (t_lo // k, _ceil_div(t_hi, k))
        lo, hi = lo + term[0], hi + term[1]
        bound = _magnitude(term)
        if bound <= 1:
            # Each later term is at most half the one before it (|r| <= 1).
            return lo - bound, hi + bound


def _atanh_series(z: Rational, w: int) -> _Interval:
    """Enclose ``atanh(z)`` for an exact ``|z| <= 1/2``."""
    power = _fixed(z, w)
    z2 = _fixed(z * z, w)
    lo, hi = power
    j = 0
    while True:
        j += 1
        power = _mul(power, z2, w)
        d = 2 * j + 1
        lo, hi = lo + power[0] // d, hi + _ceil_div(power[1], d)
        bound = _magnitude(power)
        if bound <= 1:
            # Tail < |power| * z**2 / (1 - z**2) <= |power|.
            return lo - bound, hi + bound


def _sin_cos_series(r: _Interval, w: int) -> tuple[_Interval, _Interval]:
    """Enclose ``(sin(r), cos(r))`` for an enclosed ``|r| <= 1``."""
    one = 1 << w
    r2 = _mul(r, r, w)

    def series(first: _Interval, start: int) -> _Interval:
        term = first
        lo, hi = first
        k = start
        while True:
            t_lo, t_hi = _mul(term, r2, w)
            d = (k + 1) * (k + 2)
            k += 2
            # Next term is -term * r**2 / ((k+1)(k+2)).
            term = (-_ceil_div(t_hi, d), -(t_lo // d))
            lo, hi = lo + term[0], hi + term[1]
            bound = _magnitude(term)
            if bound <= 1:
                return lo - bound, hi + bound

    return series(r, 1), series((one, one), 0)


# ---------------------------------------------------------------- enclosures


def _enclose_exp(x: Rational, w: int) -> _RationalInterval:
    s = abs(x)
    # Halve until s / 2**m <= 1/256, then square m times.
    m = max(0, s.ceil().bit_length() + 8)
    wb = w + m + 8
    r = _fixed(s / (1 << m), wb)
    lo, hi = _exp_series(r, wb)
    for _ in range(m):
        lo, hi = (lo * lo) >> wb, -(-(hi * hi) >> wb)
    rlo, rhi = _to_rationals((lo, hi), wb)
    if x < 0:
        return 1 / rhi, 1 / rlo
    return rlo, rhi


def _ln2_fixed(w: int) -> _Interval:
    lo, hi = _atanh_series(Rational(1, 3), w)
    return 2 * lo, 2 * hi


def _enclose_ln(x: Rational, w: int) -> _RationalInterval:
    # Write x = 2**k * y with y in [2/3, 4/3], so |(y-1)/(y+1)| <= 1/7.
    k = x.numerator.bit_length() - x.denominator.bit_length()
    y = x / Rational(2) ** k
    while y > Rational(4, 3):
        y /= 2
        k += 1
    while y < Rational(2, 3):
        y *= 2
        k -= 1
    z = (y - 1) / (y + 1)
    # Extra bits when the result is small (x close to 1).
    wb = w + abs(k).bit_length() + _small_bits(z) + 8
    a_lo, a_hi = _atanh_series(z, wb)
    lo, hi = 2 * a_lo, 2 * a_hi
    if k:
        l_lo, l_hi = _ln2_fixed(wb)
        if k > 0:
            lo, hi = lo + k * l_lo, hi + k * l_hi
        else:
            lo, hi = lo + k * l_hi, hi + k * l_lo
    return _to_rationals((lo, hi), wb)


def _sqrt_is_exact(x: Rational) -> Rational | None:
    rn = math.isqrt(x.numerator)
    rd = math.isqrt(x.denominator)
    if rn * rn == x.numerator and rd * rd == x.denominator:
        return Rational(rn, rd)
    return None


def _enclose_sqrt(x: Rational, w: int) -> _RationalInterval:
    # Add bits for small x so the relative precision stays near w bits.
    wb = w + max(0, (x.denominator.bit_length() - x.numerator.bit_length()) // 2) + 4
    n = (x.numerator << (2 * wb)) // x.denominator
    root = math.isqrt(n)
    return _to_rationals((root, root + 1), wb)


def _enclose_sin_cos(x: Rational, w: int, want_sin: bool) -> _RationalInterval:
    # Pick k with x - k*pi/2 small, using a rough pi; the enclosure below
    # is exact for whatever k is picked.
    rough_bits = abs(x).ceil().bit_length() + 64
    rough_lo, _ = _pi_fixed(rough_bits)
    k = (x * 2 * (1 << rough_bits) / rough_lo + Rational(1, 2)).floor()
    wb = w + abs(k).bit_length() + 8
    p_lo, p_hi = _pi_fixed(wb + 1)
    # k * pi / 2 enclosed at wb + 2 bits.
    if k >= 0:
        kp_lo, kp_hi = k * p_lo, k * p_hi
    else:
        kp_lo, kp_hi = k * p_hi, k * p_lo
    scale = 1 << (wb + 2)
    r_lo = x - Rational(kp_hi, scale)
    r_hi = x - Rational(kp_lo, scale)
    r = (_fixed(r_lo, wb)[0], _fixed(r_hi, wb)[1])
    s, c = _sin_cos_series(r, wb)
    quadrant = k % 4
    if not want_sin:
        quadrant = (quadrant + 1) % 4
    # sin(r + q*pi/2) for q = 0..3 is sin r, cos r, -sin r, -cos r.
    chosen = {0: s, 1: c, 2: (-s[1], -s[0]), 3: (-c[1], -c[0])}[quadrant]
    return _to_rationals(chosen, wb)


def _atan_fixed_unit(y: Rational, w: int) -> _Interval:
    """Enclose ``atan(y)`` for an exact ``0 < y <= 1`` with Euler's series.

    atan(y) = sum_n (2n)!! / (2n+1)!! * y**(2n+1) / (1+y**2)**(n+1),
    with every term positive and the ratio of terms at most 1/2.
    """
    q = y * y / (1 + y * y)
    qn, qd = q.numerator, q.denominator
    term = _fixed(y / (1 + y * y), w)
    lo, hi = term
    n = 0
    while True:
        n += 1
        num = 2 * n * qn
        den = (2 * n + 1) * qd
        term = (term[0] * num // den, _ceil_div(term[1] * num, den))
        lo, hi = lo + term[0], hi + term[1]
        if term[1] <= 1:
            return lo, hi + term[1]


def _enclose_atan(x: Rational, w: int) -> _RationalInterval:
    y = abs(x)
    wb = w + _small_bits(y) + 8
    if y <= 1:
        lo, hi = _atan_fixed_unit(y, wb)
    else:
        # atan(y) = pi/2 - atan(1/y) for y > 1.
        a_lo, a_hi = _atan_fixed_unit(1 / y, wb)
        p_lo, p_hi = _pi_fixed(wb + 1)
        # _pi_fixed(wb + 1) >> 2 is pi/2 at wb bits.
        lo, hi = (p_lo >> 2) - a_hi, -(-p_hi >> 2) - a_lo
    if x < 0:
        lo, hi = -hi, -lo
    return _to_rationals((lo, hi), wb)


# --------------------------------------------------------------- public API


def sqrt(x: RationalLike, digits: int) -> Rational:
    """Square root of ``x >= 0``; exact square roots are rounded exactly."""
    _check_precision(digits)
    x = _to_rational(x)
    if x < 0:
        raise ValueError("sqrt of a negative number")
    exact = _sqrt_is_exact(x)
    if exact is not None:
        return round_to_rational(exact, digits)
    return _settle(digits, lambda w: _enclose_sqrt(x, w))


def exp(x: RationalLike, digits: int) -> Rational:
    """``e ** x`` for ``|x| <= 100000``."""
    _check_precision(digits)
    x = _to_rational(x)
    if abs(x) > _EXP_LIMIT:
        raise OverflowError(f"exp argument out of range (|x| > {_EXP_LIMIT})")
    if x == 0:
        return Rational(1)
    return _settle(digits, lambda w: _enclose_exp(x, w))


def ln(x: RationalLike, digits: int) -> Rational:
    """Natural logarithm of ``x > 0``."""
    _check_precision(digits)
    x = _to_rational(x)
    if x <= 0:
        raise ValueError("ln of a non-positive number")
    if x == 1:
        return Rational(0)
    return _settle(digits, lambda w: _enclose_ln(x, w))


def sin(x: RationalLike, digits: int) -> Rational:
    """Sine of ``x`` radians, for ``|x| <= 10**100``."""
    _check_precision(digits)
    x = _to_rational(x)
    if abs(x) > _TRIG_LIMIT:
        raise ValueError("sin argument out of range (|x| > 1e100)")
    if x == 0:
        return Rational(0)
    return _settle(digits, lambda w: _enclose_sin_cos(x, w, True))


def cos(x: RationalLike, digits: int) -> Rational:
    """Cosine of ``x`` radians, for ``|x| <= 10**100``."""
    _check_precision(digits)
    x = _to_rational(x)
    if abs(x) > _TRIG_LIMIT:
        raise ValueError("cos argument out of range (|x| > 1e100)")
    if x == 0:
        return Rational(1)
    return _settle(digits, lambda w: _enclose_sin_cos(x, w, False))


def atan(x: RationalLike, digits: int) -> Rational:
    """Arctangent of ``x``, in radians."""
    _check_precision(digits)
    x = _to_rational(x)
    if x == 0:
        return Rational(0)
    return _settle(digits, lambda w: _enclose_atan(x, w))


def pi(digits: int) -> Rational:
    """``pi`` rounded to ``digits`` significant digits."""
    _check_precision(digits)
    return _settle(digits, lambda w: _to_rationals(_pi_fixed(w), w))


def e(digits: int) -> Rational:
    """Euler's number rounded to ``digits`` significant digits."""
    return exp(1, digits)
