"""Tests for woz.rational, including randomized cross-checks vs fractions."""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from woz.rational import Rational, integer_root

_RNG = random.Random(20260926)


def _rand_pair() -> tuple[int, int]:
    num = _RNG.randint(-(10**12), 10**12)
    den = _RNG.randint(1, 10**9)
    return num, den


RANDOM_PAIRS: list[tuple[tuple[int, int], tuple[int, int]]] = [
    (_rand_pair(), _rand_pair()) for _ in range(60)
]


def _same(r: Rational, f: Fraction) -> bool:
    return r.numerator == f.numerator and r.denominator == f.denominator


# -- construction & normalisation -----------------------------------------
@pytest.mark.parametrize(
    "n, d, en, ed",
    [
        (6, 8, 3, 4),
        (-6, 8, -3, 4),
        (6, -8, -3, 4),
        (-6, -8, 3, 4),
        (0, 5, 0, 1),
        (0, -5, 0, 1),
        (7, 1, 7, 1),
        (10**30, 10**28, 100, 1),
        (1, 3, 1, 3),
    ],
)
def test_normalisation(n: int, d: int, en: int, ed: int) -> None:
    r = Rational(n, d)
    assert (r.numerator, r.denominator) == (en, ed)


def test_default_is_zero() -> None:
    assert Rational() == 0


def test_zero_denominator_raises() -> None:
    with pytest.raises(ZeroDivisionError):
        Rational(1, 0)


def test_rational_numerator_and_denominator() -> None:
    assert Rational(Rational(1, 2), Rational(3, 4)) == Rational(2, 3)


def test_from_float_is_exact() -> None:
    assert Rational(0.5) == Rational(1, 2)
    assert Rational(0.1) == Rational(*(0.1).as_integer_ratio())


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_non_finite_float_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        Rational(bad)


@pytest.mark.parametrize("bad", [None, [1], 1j])
def test_bad_type_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        Rational(bad)  # type: ignore[arg-type]


def test_string_with_denominator_rejected() -> None:
    with pytest.raises(TypeError):
        Rational("1", 2)


# -- from_str -------------------------------------------------------------
@pytest.mark.parametrize(
    "text, n, d",
    [
        ("3/4", 3, 4),
        ("-3/4", -3, 4),
        ("+3/4", 3, 4),
        ("6/8", 3, 4),
        (" 1 / 2 ", 1, 2),
        ("-1.25", -5, 4),
        ("1.25", 5, 4),
        ("1e-3", 1, 1000),
        ("1E3", 1000, 1),
        ("2.5e+2", 250, 1),
        ("-2.5e-2", -1, 40),
        (".5", 1, 2),
        ("-.75", -3, 4),
        ("5.", 5, 1),
        ("42", 42, 1),
        ("-0", 0, 1),
        ("0.000", 0, 1),
        ("007", 7, 1),
        ("1.23456789e-20", 123456789, 10**28),
        ("123e0", 123, 1),
    ],
)
def test_from_str(text: str, n: int, d: int) -> None:
    r = Rational.from_str(text)
    assert (r.numerator, r.denominator) == (n, d)


@pytest.mark.parametrize(
    "text", ["", "abc", "1/", "/2", "1.2.3", "1e", "e5", "--1", "1/2/3", "1 2", "."]
)
def test_from_str_invalid(text: str) -> None:
    with pytest.raises(ValueError):
        Rational.from_str(text)


def test_from_str_zero_denominator() -> None:
    with pytest.raises(ZeroDivisionError):
        Rational.from_str("1/0")


def test_from_str_requires_str() -> None:
    with pytest.raises(TypeError):
        Rational.from_str(3)  # type: ignore[arg-type]


def test_string_constructor() -> None:
    assert Rational("-1.25") == Rational(-5, 4)


# -- arithmetic -----------------------------------------------------------
@pytest.mark.parametrize(
    "a, b, op, expected",
    [
        (Rational(1, 2), Rational(1, 3), "+", Rational(5, 6)),
        (Rational(1, 2), Rational(1, 3), "-", Rational(1, 6)),
        (Rational(2, 3), Rational(3, 4), "*", Rational(1, 2)),
        (Rational(2, 3), Rational(4, 9), "/", Rational(3, 2)),
        (Rational(1, 2), 1, "+", Rational(3, 2)),
        (1, Rational(1, 2), "+", Rational(3, 2)),
        (1, Rational(1, 4), "-", Rational(3, 4)),
        (3, Rational(1, 4), "*", Rational(3, 4)),
        (3, Rational(3, 4), "/", Rational(4)),
    ],
)
def test_arithmetic(a: object, b: object, op: str, expected: Rational) -> None:
    ops = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
    }
    assert ops[op](a, b) == expected


def test_unary_ops() -> None:
    r = Rational(-3, 4)
    assert -r == Rational(3, 4)
    assert +r is r
    assert abs(r) == Rational(3, 4)


@pytest.mark.parametrize("left", [True, False])
def test_division_by_zero(left: bool) -> None:
    with pytest.raises(ZeroDivisionError):
        if left:
            Rational(1) / Rational(0)
        else:
            1 / Rational(0)


def test_float_mixing_not_supported() -> None:
    with pytest.raises(TypeError):
        Rational(1) + 0.5  # type: ignore[operator]


# -- powers ---------------------------------------------------------------
@pytest.mark.parametrize(
    "base, exp, expected",
    [
        (Rational(2, 3), 3, Rational(8, 27)),
        (Rational(2, 3), -2, Rational(9, 4)),
        (Rational(-2, 3), -3, Rational(-27, 8)),
        (Rational(5), 0, Rational(1)),
        (Rational(0), 0, Rational(1)),
        (Rational(4, 9), Rational(1, 2), Rational(2, 3)),
        (Rational(8, 27), Rational(2, 3), Rational(4, 9)),
        (Rational(-8), Rational(1, 3), Rational(-2)),
        (Rational(16), Rational(-3, 4), Rational(1, 8)),
        (Rational(10**40), Rational(1, 2), Rational(10**20)),
    ],
)
def test_pow(base: Rational, exp: object, expected: Rational) -> None:
    assert base**exp == expected


def test_rpow() -> None:
    assert 2 ** Rational(3) == 8
    assert 4 ** Rational(1, 2) == 2


def test_pow_irrational_raises() -> None:
    with pytest.raises(ValueError):
        Rational(2) ** Rational(1, 2)


def test_pow_even_root_of_negative_raises() -> None:
    with pytest.raises(ValueError):
        Rational(-4) ** Rational(1, 2)


def test_zero_negative_power_raises() -> None:
    with pytest.raises(ZeroDivisionError):
        Rational(0) ** -1


@pytest.mark.parametrize(
    "n, k", [(0, 3), (1, 5), (26, 3), (27, 3), (10**50, 7), (2**200 - 1, 4)]
)
def test_integer_root(n: int, k: int) -> None:
    r = integer_root(n, k)
    assert r**k <= n < (r + 1) ** k


def test_integer_root_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        integer_root(-1, 2)
    with pytest.raises(ValueError):
        integer_root(4, 0)


# -- comparisons, hashing, conversions ------------------------------------
def test_comparisons() -> None:
    a, b = Rational(1, 3), Rational(1, 2)
    assert a < b and a <= b and b > a and b >= a
    assert a <= Rational(2, 6) and a >= Rational(2, 6)
    assert a != b
    assert Rational(2) == 2 and 2 == Rational(2)
    assert Rational(1, 2) < 1 and 0 < Rational(1, 2)


def test_eq_with_other_types() -> None:
    assert Rational(1) != "1"
    with pytest.raises(TypeError):
        _ = Rational(1) < "1"  # type: ignore[operator]


@pytest.mark.parametrize(
    "n, d", [(1, 2), (-1, 2), (5, 1), (-1, 1), (0, 1), (10**40, 3), (7, 10**30)]
)
def test_hash_matches_fraction(n: int, d: int) -> None:
    assert hash(Rational(n, d)) == hash(Fraction(n, d))


def test_hash_equal_values() -> None:
    assert hash(Rational(2, 4)) == hash(Rational(1, 2))
    assert hash(Rational(3)) == hash(3)
    assert len({Rational(1, 2), Rational(2, 4), Rational(3, 6)}) == 1


def test_conversions() -> None:
    r = Rational(-7, 2)
    assert float(r) == -3.5
    assert int(r) == -3
    assert math.trunc(r) == -3
    assert math.floor(r) == -4
    assert math.ceil(r) == -3
    assert bool(Rational(0)) is False
    assert bool(r) is True
    assert Rational(4).is_integer() and not r.is_integer()


def test_str_and_repr() -> None:
    assert str(Rational(3, 4)) == "3/4"
    assert str(Rational(-5)) == "-5"
    assert repr(Rational(3, 4)) == "Rational(3, 4)"
    assert repr(Rational(2)) == "Rational(2)"


# -- randomized cross-checks against fractions.Fraction -------------------
@pytest.mark.parametrize("p, q", RANDOM_PAIRS)
def test_random_add_sub(p: tuple[int, int], q: tuple[int, int]) -> None:
    a, b = Rational(*p), Rational(*q)
    fa, fb = Fraction(*p), Fraction(*q)
    assert _same(a + b, fa + fb)
    assert _same(a - b, fa - fb)


@pytest.mark.parametrize("p, q", RANDOM_PAIRS)
def test_random_mul_div(p: tuple[int, int], q: tuple[int, int]) -> None:
    a, b = Rational(*p), Rational(*q)
    fa, fb = Fraction(*p), Fraction(*q)
    assert _same(a * b, fa * fb)
    if fb:
        assert _same(a / b, fa / fb)


@pytest.mark.parametrize("p, q", RANDOM_PAIRS)
def test_random_compare_hash(p: tuple[int, int], q: tuple[int, int]) -> None:
    a, b = Rational(*p), Rational(*q)
    fa, fb = Fraction(*p), Fraction(*q)
    assert (a < b) == (fa < fb)
    assert (a == b) == (fa == fb)
    assert (a >= b) == (fa >= fb)
    assert hash(a) == hash(fa)


@pytest.mark.parametrize("seed", range(20))
def test_random_pow_and_rounding(seed: int) -> None:
    rng = random.Random(seed)
    n, d = rng.randint(-999, 999) or 1, rng.randint(1, 999)
    e = rng.randint(-6, 6)
    assert _same(Rational(n, d) ** e, Fraction(n, d) ** e)
    r, f = Rational(n, d), Fraction(n, d)
    assert (math.floor(r), math.ceil(r), math.trunc(r)) == (
        math.floor(f),
        math.ceil(f),
        math.trunc(f),
    )


@pytest.mark.parametrize("seed", range(20))
def test_random_from_str(seed: int) -> None:
    rng = random.Random(1000 + seed)
    text = f"{rng.choice(['', '-'])}{rng.randint(0, 10**6)}.{rng.randint(0, 10**6)}e{rng.randint(-30, 30)}"
    assert _same(Rational.from_str(text), Fraction(text))
