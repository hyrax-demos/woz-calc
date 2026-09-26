"""Elementary functions evaluated to a requested number of digits.

Every public function returns ``x`` rounded (round-half-even) to exactly
``digits`` significant digits, as a :class:`~woz.rational.Rational`, so
``format_sig(f(x, digits), digits)`` is correct in every printed digit.

Internally each function computes a rigorous enclosing interval using
fixed-point integer arithmetic (values scaled by ``2**w``), series
expansions and exact argument reduction.  If both ends of the interval
round to the same value that value is returned; otherwise the working
precision is doubled (Ziv's strategy).  None of the functions can land
exactly on a rounding midpoint for a non-trivial argument (the results are
irrational), and trivial arguments are answered exactly up front, so the
loop always terminates.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from woz.bigdec import exponent10, round_sig
from woz.rational import Rational, RationalLike, integer_root

MIN_DIGITS: int = 1
MAX_DIGITS: int = 60
#: Refuse arguments whose magnitude would make exact reduction absurdly slow.
MAX_EXP_ARG: Rational = Rational(100_000)
MAX_TRIG_ARG: Rational = Rational(10**30)

_MAX_REFINEMENTS: int = 12

# A fixed-point value: (approximation, error bound), both in units of 2**-w.
Fixed = tuple[int, int]
Interval = tuple[Rational, Rational]


class DomainError(ValueError):
    """Raised when an argument lies outside a function's real domain."""


# -- helpers -------------------------------------------------------------
def _to_rational(x: RationalLike) -> Rational:
    if isinstance(x, bool):
        raise TypeError("booleans are not numbers here")
    if isinstance(x, Rational):
        return x
    if isinstance(x, int):
        return Rational(x)
    raise TypeError(f"expected Rational or int, got {type(x).__name__}")


def _check_digits(digits: int) -> None:
    if isinstance(digits, bool) or not isinstance(digits, int):
        raise TypeError("digits must be an int")
    if not MIN_DIGITS <= digits <= MAX_DIGITS:
        raise ValueError(f"digits must be between {MIN_DIGITS} and {MAX_DIGITS}")


def _log2_size(x: Rational) -> int:
    """A rough ``log2(|x|)`` (only used to pick starting precisions)."""
    if not x:
        return 0
    return x.numerator.bit_length() - x.denominator.bit_length()


def _interval(value: Fixed, w: int) -> Interval:
    a, e = value
    scale = 1 << w
    return Rational(a - e, scale), Rational(a + e, scale)


def _ziv(digits: int, start_bits: int, compute: Callable[[int], Interval]) -> Rational:
    w = max(start_bits, 16) + int(digits * 3.33) + 16
    for _ in range(_MAX_REFINEMENTS):
        lo, hi = compute(w)
        if (lo > 0 or hi < 0) and round_sig(lo, digits) == round_sig(hi, digits):
            return round_sig(lo, digits)
        w *= 2
    raise ArithmeticError("failed to converge to the requested precision")


# -- fixed-point series (all arguments non-negative) ----------------------
def _exp_series(p: int, q: int, w: int) -> Fixed:
    """``exp(p/q)`` for ``0 <= p/q <= 1/2``."""
    term = 1 << w
    total = term
    k = 0
    while term:
        k += 1
        term = term * p // (q * k)
        total += term
    return total, 3 * k + 4


def _atanh_series(p: int, q: int, w: int) -> Fixed:
    """``atanh(p/q)`` for ``0 <= p/q <= 1/3``."""
    power = (p << w) // q
    p2, q2 = p * p, q * q
    total = 0
    k = 0
    while power:
        total += power // (2 * k + 1)
        power = power * p2 // q2
        k += 1
    return total, 3 * k + 6


def _atan_series(p: int, q: int, w: int) -> Fixed:
    """``atan(p/q)`` for ``0 <= p/q <= 1/2``."""
    power = (p << w) // q
    p2, q2 = p * p, q * q
    total = 0
    k = 0
    while power:
        term = power // (2 * k + 1)
        total += -term if k % 2 else term
        power = power * p2 // q2
        k += 1
    return total, 3 * k + 6


def _sin_series(r: int, w: int) -> Fixed:
    """``sin(r / 2**w)`` for ``0 <= r / 2**w <= 1``."""
    denom = 1 << (2 * w)
    r2 = r * r
    term = r
    total = term
    k = 0
    while term:
        k += 1
        term = term * r2 // (denom * (2 * k) * (2 * k + 1))
        total += -term if k % 2 else term
    return total, 2 * k + 4


def _cos_series(r: int, w: int) -> Fixed:
    """``cos(r / 2**w)`` for ``0 <= r / 2**w <= 1``."""
    denom = 1 << (2 * w)
    r2 = r * r
    term = 1 << w
    total = term
    k = 0
    while term:
        k += 1
        term = term * r2 // (denom * (2 * k - 1) * (2 * k))
        total += -term if k % 2 else term
    return total, 2 * k + 4


@lru_cache(maxsize=64)
def _ln2_fixed(w: int) -> Fixed:
    a, e = _atanh_series(1, 3, w)
    return 2 * a, 2 * e


@lru_cache(maxsize=64)
def _pi_fixed(w: int) -> Fixed:
    a1, e1 = _atan_series(1, 5, w)
    a2, e2 = _atan_series(1, 239, w)
    return 16 * a1 - 4 * a2, 16 * e1 + 4 * e2


def _atan_fixed(x: Rational, w: int) -> Fixed:
    """``atan(x)`` for ``x >= 0`` using only exact argument reductions."""
    half = Rational(1, 2)
    if x > 1:
        pa, pe = _pi_fixed(w)
        a, e = _atan_fixed(1 / x, w)
        return pa // 2 - a, pe // 2 + 1 + e
    if x > half:
        # atan(x) = atan(1/2) + atan((x - 1/2) / (1 + x/2)), exact rationals.
        a1, e1 = _atan_series(1, 2, w)
        y = (x - half) / (1 + x / 2)
        a2, e2 = _atan_series(y.numerator, y.denominator, w)
        return a1 + a2, e1 + e2
    return _atan_series(x.numerator, x.denominator, w)


# -- public functions ----------------------------------------------------
def pi(digits: int) -> Rational:
    """Return pi rounded to ``digits`` significant digits."""
    _check_digits(digits)
    return _ziv(digits, 4, lambda w: _interval(_pi_fixed(w), w))


def sqrt(x: RationalLike, digits: int) -> Rational:
    """Return ``sqrt(x)`` rounded to ``digits`` significant digits."""
    _check_digits(digits)
    x = _to_rational(x)
    if x < 0:
        raise DomainError("sqrt of a negative number")
    rn = integer_root(x.numerator, 2)
    rd = integer_root(x.denominator, 2)
    if rn * rn == x.numerator and rd * rd == x.denominator:
        return round_sig(Rational(rn, rd), digits)

    def compute(w: int) -> Interval:
        s = integer_root((x.numerator << (2 * w)) // x.denominator, 2)
        # s <= sqrt(x) * 2**w < s + 1 + 1 (floor of N, then of isqrt).
        return Rational(s, 1 << w), Rational(s + 2, 1 << w)

    return _ziv(digits, max(0, -_log2_size(x)), compute)


def exp(x: RationalLike, digits: int) -> Rational:
    """Return ``e ** x`` rounded to ``digits`` significant digits."""
    _check_digits(digits)
    x = _to_rational(x)
    if not x:
        return round_sig(Rational(1), digits)
    if abs(x) > MAX_EXP_ARG:
        raise OverflowError("exp argument too large")
    ax = abs(x)
    s = max(0, _log2_size(ax) + 2)
    r = ax / (1 << s)  # 0 < r <= 1/2

    def compute(w: int) -> Interval:
        wa = w + s + 8
        a, e = _exp_series(r.numerator, r.denominator, wa)
        for _ in range(s):
            # (a +- e)**2 bounds the true square; floor adds < 1 ulp.
            new_a = a * a >> wa
            e = -(-(2 * a * e + e * e) >> wa) + 1
            a = new_a
        lo, hi = _interval((a, e), wa)
        if x < 0:
            return 1 / hi, 1 / lo
        return lo, hi

    return _ziv(digits, 0, compute)


def ln(x: RationalLike, digits: int) -> Rational:
    """Return the natural logarithm of ``x`` to ``digits`` significant digits."""
    _check_digits(digits)
    x = _to_rational(x)
    if x <= 0:
        raise DomainError("ln of a non-positive number")
    if x == 1:
        return Rational(0)
    k = _log2_size(x)
    m = x / (1 << k) if k >= 0 else x * (1 << -k)
    while m > Rational(4, 3):
        m /= 2
        k += 1
    while m < Rational(2, 3):
        m *= 2
        k -= 1
    z = (m - 1) / (m + 1)  # |z| <= 1/5
    sign = -1 if z < 0 else 1
    az = abs(z)
    start = max(0, int(-exponent10(x - 1) * 3.33)) + 4

    def compute(w: int) -> Interval:
        a1, e1 = _atanh_series(az.numerator, az.denominator, w)
        a2, e2 = _ln2_fixed(w)
        return _interval((k * a2 + 2 * sign * a1, abs(k) * e2 + 2 * e1), w)

    return _ziv(digits, start, compute)


def atan(x: RationalLike, digits: int) -> Rational:
    """Return ``atan(x)`` (radians) to ``digits`` significant digits."""
    _check_digits(digits)
    x = _to_rational(x)
    if not x:
        return Rational(0)
    ax = abs(x)
    start = max(0, -_log2_size(ax)) + 4

    def compute(w: int) -> Interval:
        lo, hi = _interval(_atan_fixed(ax, w), w)
        return (lo, hi) if x > 0 else (-hi, -lo)

    return _ziv(digits, start, compute)


def _trig(x: Rational, digits: int, want_sin: bool) -> Rational:
    if abs(x) > MAX_TRIG_ARG:
        raise OverflowError("trigonometric argument too large")
    extra = max(0, _log2_size(x)) + 8

    def compute(w: int) -> Interval:
        wp = w + extra
        pa, pe = _pi_fixed(wp)
        half_pi, half_err = pa >> 1, (pe >> 1) + 1
        xf = (x.numerator << wp) // x.denominator  # error < 1 ulp
        k = (2 * xf + half_pi) // (2 * half_pi)  # nearest multiple of pi/2
        r = xf - k * half_pi
        r_err = 1 + abs(k) * half_err
        quadrant = (k + (0 if want_sin else 1)) % 4
        use_sin = quadrant in (0, 2)
        negate = quadrant >= 2
        if use_sin:
            a, e = _sin_series(abs(r), wp)
            if r < 0:
                a = -a
        else:
            a, e = _cos_series(abs(r), wp)
        if negate:
            a = -a
        return _interval((a, e + r_err), wp)

    return _ziv(digits, 0, compute)


def sin(x: RationalLike, digits: int) -> Rational:
    """Return ``sin(x)`` (radians) to ``digits`` significant digits."""
    _check_digits(digits)
    x = _to_rational(x)
    if not x:
        return Rational(0)
    return _trig(x, digits, want_sin=True)


def cos(x: RationalLike, digits: int) -> Rational:
    """Return ``cos(x)`` (radians) to ``digits`` significant digits."""
    _check_digits(digits)
    x = _to_rational(x)
    if not x:
        return round_sig(Rational(1), digits)
    return _trig(x, digits, want_sin=False)
