"""Correctly rounded elementary functions over :class:`~woz.rational.Rational`.

Every public function takes a ``digits`` argument (1 to 60) and returns the
value correctly rounded (half-even) to that many significant digits, as an
exact :class:`Rational`.  Formatting that Rational with
:func:`woz.bigdec.format_significant` at the same ``digits`` therefore prints
only correct digits.

Internally values are *balls*: a pair ``(m, r)`` of integers at a binary
fixed-point scale ``p`` meaning "the true value lies in
``[(m - r) / 2**p, (m + r) / 2**p]``".  Every operation rounds outwards, so the
enclosure is rigorous.  A Ziv loop raises ``p`` until both ends of the
enclosure round to the same ``digits``-digit number, which is then provably
the correctly rounded result.  The arguments used are rational and non-trivial,
so the results are irrational (Lindemann-Weierstrass) and the loop terminates.
"""

from __future__ import annotations

import math
from typing import Callable

from woz.bigdec import round_to_rational
from woz.rational import Rational, RationalLike, as_rational, integer_root_floor

MIN_DIGITS = 1
MAX_DIGITS = 60

#: Arguments to :func:`exp` beyond this magnitude raise ``OverflowError``.
EXP_LIMIT = 100_000
#: Arguments to :func:`sin`, :func:`cos` beyond this magnitude are rejected.
TRIG_LIMIT = 10**30

_MAX_BITS = 1 << 16

Ball = tuple[int, int]
Enclosure = tuple[Rational, Rational]


# ----------------------------------------------------------------- ball maths
def _ball(x: Rational, p: int) -> Ball:
    """Enclose an exact rational at scale ``p``."""
    return (x.numerator << p) // x.denominator, 1


def _mul(a: Ball, b: Ball, p: int) -> Ball:
    (m1, r1), (m2, r2) = a, b
    m = (m1 * m2) >> p
    r = ((abs(m1) * r2 + abs(m2) * r1 + r1 * r2) >> p) + 2
    return m, r


def _div_int(a: Ball, k: int) -> Ball:
    """Divide a ball by a positive integer."""
    return a[0] // k, a[1] // k + 2


def _scale(a: Ball, n: int) -> Ball:
    """Multiply a ball by an integer exactly."""
    return a[0] * n, a[1] * abs(n)


def _add(a: Ball, b: Ball) -> Ball:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: Ball, b: Ball) -> Ball:
    return a[0] - b[0], a[1] + b[1]


def _enclosure(a: Ball, p: int, factor: Rational = Rational(1)) -> Enclosure:
    """Turn a ball into ``(lo, hi)`` Rationals, times a positive ``factor``."""
    m, r = a
    den = 1 << p
    return Rational(m - r, den) * factor, Rational(m + r, den) * factor


def _check_digits(digits: int) -> None:
    if isinstance(digits, bool) or not isinstance(digits, int):
        raise TypeError("digits must be an int")
    if not MIN_DIGITS <= digits <= MAX_DIGITS:
        raise ValueError(f"digits must be between {MIN_DIGITS} and {MAX_DIGITS}")


def _ziv(compute: Callable[[int], Enclosure], digits: int) -> Rational:
    """Refine ``compute(p)`` until its enclosure rounds to a single value."""
    p = int(digits * 3.33) + 24
    while p <= _MAX_BITS:
        lo, hi = compute(p)
        a = round_to_rational(lo, digits)
        if a and a == round_to_rational(hi, digits):
            return a
        p *= 2
    raise ArithmeticError("precision limit exceeded")  # pragma: no cover


# -------------------------------------------------------------- series kernels
def _atan_series(y: Ball, p: int) -> Ball:
    """atan(y) for ``|y| <= 1/2``; the series alternates so the tail is bounded
    by the first omitted term."""
    y2 = _mul(y, y, p)
    total, term = y, y
    k = 1
    while abs(term[0]) > 1:
        term = _mul(term, y2, p)
        k += 2
        piece = _div_int(term, k)
        total = _sub(total, piece) if k % 4 == 3 else _add(total, piece)
    return total[0], total[1] + abs(term[0]) + term[1]


def _atanh_series(z: Ball, p: int) -> Ball:
    """atanh(z) for ``|z| <= 1/3``; tail is at most ``9/8`` of the last term."""
    z2 = _mul(z, z, p)
    total, term = z, z
    k = 1
    while abs(term[0]) > 1:
        term = _mul(term, z2, p)
        k += 2
        total = _add(total, _div_int(term, k))
    return total[0], total[1] + 2 * (abs(term[0]) + term[1])


def _exp_series(x: Ball, p: int) -> Ball:
    """exp(x) for ``|x| <= 1/64``; tail is at most twice the last term."""
    one = 1 << p
    total, term = (one, 0), (one, 0)
    k = 0
    while abs(term[0]) > 1 or k < 2:
        k += 1
        term = _div_int(_mul(term, x, p), k)
        total = _add(total, term)
    return total[0], total[1] + 2 * (abs(term[0]) + term[1])


def _sin_cos_series(x: Ball, p: int) -> tuple[Ball, Ball]:
    """(sin x, cos x) for ``|x| <= 1``; both series alternate and decrease."""
    sin_total, sin_term = x, x
    cos_total, cos_term = (1 << p, 0), (1 << p, 0)
    k = 0
    while abs(sin_term[0]) > 1 or abs(cos_term[0]) > 1:
        cos_term = _div_int(_mul(sin_term, x, p), 2 * k + 2)
        sin_term = _div_int(_mul(cos_term, x, p), 2 * k + 3)
        k += 1
        if k % 2:
            sin_total, cos_total = _sub(sin_total, sin_term), _sub(cos_total, cos_term)
        else:
            sin_total, cos_total = _add(sin_total, sin_term), _add(cos_total, cos_term)
    sin_tail = abs(sin_term[0]) + sin_term[1] + 1
    cos_tail = abs(cos_term[0]) + cos_term[1] + 1
    return (
        (sin_total[0], sin_total[1] + sin_tail),
        (cos_total[0], cos_total[1] + cos_tail),
    )


def _atan_inv_int(k: int, p: int) -> Ball:
    """atan(1/k) for an integer ``k >= 2`` using pure integer arithmetic."""
    term = (1 << p) // k
    total = term
    k2 = k * k
    n = 1
    steps = 1
    while term:
        term //= k2
        n += 2
        steps += 1
        total += -(term // n) if n % 4 == 3 else term // n
    return total, 2 * steps + 2


def _pi_ball(p: int) -> Ball:
    """Machin's formula: pi = 16 atan(1/5) - 4 atan(1/239)."""
    return _sub(_scale(_atan_inv_int(5, p), 16), _scale(_atan_inv_int(239, p), 4))


def _ln2_ball(p: int) -> Ball:
    """ln 2 = 2 atanh(1/3)."""
    return _scale(_atanh_series(_ball(Rational(1, 3), p), p), 2)


def _ln_ball(x: Rational, p: int) -> Ball:
    """ln(x) for ``x > 0``: split off a power of two, then use atanh."""
    e = x.numerator.bit_length() - x.denominator.bit_length()
    m = x / Rational(2) ** e  # m in (1/2, 2)
    z = (m - 1) / (m + 1)  # |z| < 1/3
    result = _scale(_atanh_series(_ball(z, p), p), 2)
    if e:
        result = _add(result, _scale(_ln2_ball(p), e))
    return result


# ------------------------------------------------------------ public functions
def pi(digits: int = 30) -> Rational:
    """pi correctly rounded to ``digits`` significant digits."""
    _check_digits(digits)
    return _ziv(lambda p: _enclosure(_pi_ball(p), p), digits)


def root(x: RationalLike | str, n: int, digits: int = 30) -> Rational:
    """Real ``n``-th root of ``x`` (``n >= 1``), correctly rounded.

    Odd roots of negative numbers are negative; even roots of negative
    numbers raise ``ValueError``.
    """
    _check_digits(digits)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("root degree must be a positive int")
    v = as_rational(x)
    if v < 0:
        if n % 2 == 0:
            raise ValueError("even root of a negative number")
        return -root(-v, n, digits)
    try:
        return round_to_rational(v ** Rational(1, n), digits)
    except ValueError:
        pass
    # x**(1/n) = (a * b**(n-1)) ** (1/n) / b
    a, b = v.numerator, v.denominator
    radicand = a * b ** (n - 1)

    def compute(p: int) -> Enclosure:
        s = integer_root_floor(radicand << (n * p), n)
        den = b << p
        return Rational(s, den), Rational(s + 1, den)

    return _ziv(compute, digits)


def sqrt(x: RationalLike | str, digits: int = 30) -> Rational:
    """Square root of ``x >= 0``, correctly rounded."""
    v = as_rational(x)
    if v < 0:
        raise ValueError("sqrt() of a negative number")
    return root(v, 2, digits)


def exp(x: RationalLike | str, digits: int = 30) -> Rational:
    """e ** x, correctly rounded.  ``|x|`` must not exceed :data:`EXP_LIMIT`."""
    _check_digits(digits)
    v = as_rational(x)
    if not v:
        return Rational(1)
    if abs(v) > EXP_LIMIT:
        raise OverflowError("exp() argument too large")
    # x = n ln2 + r with |r| <~ 0.35; then exp(r) = exp(r / 2**8) ** (2**8).
    n = math.floor(v / Rational(693147180559945, 10**15) + Rational(1, 2))
    halvings = 8

    def compute(p: int) -> Enclosure:
        q = p + n.bit_length() + halvings + 8
        r = _sub(_ball(v, q), _scale(_ln2_ball(q), n))
        y = _exp_series((r[0] >> halvings, (r[1] >> halvings) + 1), q)
        for _ in range(halvings):
            y = _mul(y, y, q)
        return _enclosure(y, q, Rational(2) ** n)

    return _ziv(compute, digits)


def ln(x: RationalLike | str, digits: int = 30) -> Rational:
    """Natural logarithm of ``x > 0``, correctly rounded."""
    _check_digits(digits)
    v = as_rational(x)
    if v <= 0:
        raise ValueError("ln() of a non-positive number")
    if v == 1:
        return Rational(0)
    return _ziv(lambda p: _enclosure(_ln_ball(v, p), p), digits)


def _reduced_sin_cos(v: Rational, p: int) -> tuple[Ball, Ball, int]:
    """Return (sin r, cos r, n mod 4) where ``v = n * pi/2 + r``."""
    # Pick n with a pi accurate enough that |r| stays below ~pi/4 + tiny.
    bits = max(v.numerator.bit_length() - v.denominator.bit_length(), 0) + 64
    half_pi_approx = Rational(_pi_ball(bits)[0], 1 << (bits + 1))
    n = math.floor(v / half_pi_approx + Rational(1, 2))
    q = p + n.bit_length() + 4
    half_pi = _div_int(_pi_ball(q), 2)
    r = _sub(_ball(v, q), _scale(half_pi, n))
    s, c = _sin_cos_series(r, q)
    shift = q - p
    s = (s[0] >> shift, (s[1] >> shift) + 1)
    c = (c[0] >> shift, (c[1] >> shift) + 1)
    return s, c, n % 4


def _check_trig(v: Rational) -> None:
    if abs(v) > TRIG_LIMIT:
        raise OverflowError("trigonometric argument too large")


def sin(x: RationalLike | str, digits: int = 30) -> Rational:
    """Sine of ``x`` radians, correctly rounded."""
    _check_digits(digits)
    v = as_rational(x)
    if not v:
        return Rational(0)
    _check_trig(v)

    def compute(p: int) -> Enclosure:
        s, c, quadrant = _reduced_sin_cos(v, p)
        ball = (s, c, (-s[0], s[1]), (-c[0], c[1]))[quadrant]
        return _enclosure(ball, p)

    return _ziv(compute, digits)


def cos(x: RationalLike | str, digits: int = 30) -> Rational:
    """Cosine of ``x`` radians, correctly rounded."""
    _check_digits(digits)
    v = as_rational(x)
    if not v:
        return Rational(1)
    _check_trig(v)

    def compute(p: int) -> Enclosure:
        s, c, quadrant = _reduced_sin_cos(v, p)
        ball = (c, (-s[0], s[1]), (-c[0], c[1]), s)[quadrant]
        return _enclosure(ball, p)

    return _ziv(compute, digits)


def atan(x: RationalLike | str, digits: int = 30) -> Rational:
    """Arctangent of ``x`` in radians, correctly rounded."""
    _check_digits(digits)
    v = as_rational(x)
    if not v:
        return Rational(0)
    sign = -1 if v < 0 else 1
    a = abs(v)
    # atan(a) = offset * pi/4 + direction * atan(y), with |y| <= tan(pi/8).
    if a > 1:
        a = 1 / a
        offset, direction = 2, -1
    else:
        offset, direction = 0, 1
    if a > Rational(41, 100):
        y = (a - 1) / (a + 1)
        offset += direction
    else:
        y = a

    def compute(p: int) -> Enclosure:
        ball = _scale(_atan_series(_ball(y, p), p), direction)
        if offset:
            ball = _add(ball, _scale(_div_int(_pi_ball(p), 4), offset))
        ball = _scale(ball, sign)
        return _enclosure(ball, p)

    return _ziv(compute, digits)


def e(digits: int = 30) -> Rational:
    """Euler's number correctly rounded to ``digits`` significant digits."""
    return exp(1, digits)


FUNCTIONS: dict[str, Callable[[RationalLike, int], Rational]] = {
    "sqrt": sqrt,
    "exp": exp,
    "ln": ln,
    "sin": sin,
    "cos": cos,
    "atan": atan,
}

CONSTANTS: dict[str, Callable[[int], Rational]] = {"pi": pi, "e": e}
