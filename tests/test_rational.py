"""Tests for woz.rational, including randomized cross-checks against Fraction."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from woz.rational import Rational


def as_fraction(r: Rational) -> Fraction:
    return Fraction(r.numerator, r.denominator)


def random_pair(rng: random.Random, big: bool = False) -> tuple[Rational, Fraction]:
    limit = 10**30 if big else 1000
    num = rng.randint(-limit, limit)
    den = rng.randint(1, limit)
    return Rational(num, den), Fraction(num, den)


# ------------------------------------------------------------ normalisation


@pytest.mark.parametrize(
    "num, den, expected",
    [
        (2, 4, (1, 2)),
        (-2, 4, (-1, 2)),
        (2, -4, (-1, 2)),
        (-2, -4, (1, 2)),
        (0, 5, (0, 1)),
        (0, -5, (0, 1)),
        (7, 1, (7, 1)),
        (10**20, 10**18, (100, 1)),
        (6, 9, (2, 3)),
        (-15, 25, (-3, 5)),
    ],
)
def test_normalisation(num: int, den: int, expected: tuple[int, int]) -> None:
    r = Rational(num, den)
    assert (r.numerator, r.denominator) == expected


def test_default_is_zero() -> None:
    assert Rational() == 0
    assert Rational().denominator == 1


def test_rational_components() -> None:
    assert Rational(Rational(1, 2), Rational(3, 4)) == Rational(2, 3)


def test_zero_denominator_raises() -> None:
    with pytest.raises(ZeroDivisionError):
        Rational(1, 0)


@pytest.mark.parametrize("bad", [1.5, "1", None, [1]])
def test_constructor_rejects_non_integers(bad: object) -> None:
    with pytest.raises(TypeError):
        Rational(bad)  # type: ignore[arg-type]


def test_immutable() -> None:
    r = Rational(1, 2)
    with pytest.raises(AttributeError):
        r._num = 3  # type: ignore[misc]
    with pytest.raises(AttributeError):
        del r._den


# ------------------------------------------------------------------ parsing


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3/4", Rational(3, 4)),
        ("-3/4", Rational(-3, 4)),
        ("3/-4", Rational(-3, 4)),
        (" 6 / 8 ", Rational(3, 4)),
        ("-1.25", Rational(-5, 4)),
        ("1.25", Rational(5, 4)),
        ("+1.25", Rational(5, 4)),
        ("1e-3", Rational(1, 1000)),
        ("1E3", Rational(1000)),
        ("2.5e2", Rational(250)),
        ("-2.5e-2", Rational(-1, 40)),
        (".5", Rational(1, 2)),
        ("5.", Rational(5)),
        ("0", Rational(0)),
        ("-0", Rational(0)),
        ("007", Rational(7)),
        ("123456789012345678901234567890", Rational(123456789012345678901234567890)),
        ("0.000", Rational(0)),
        ("1e+2", Rational(100)),
    ],
)
def test_from_str(text: str, expected: Rational) -> None:
    assert Rational.from_str(text) == expected


@pytest.mark.parametrize(
    "text", ["", " ", "abc", "1/", "/2", "1.2.3", "e5", ".", "1e", "1e+", "--1", "1/2/3", "0x10", "1 2"]
)
def test_from_str_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        Rational.from_str(text)


def test_from_str_zero_denominator() -> None:
    with pytest.raises(ZeroDivisionError):
        Rational.from_str("1/0")


def test_from_str_huge_exponent_rejected() -> None:
    with pytest.raises(ValueError):
        Rational.from_str("1e999999999")


def test_from_str_type_error() -> None:
    with pytest.raises(TypeError):
        Rational.from_str(12)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", range(10))
def test_from_str_matches_fraction(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(50):
        whole = rng.randint(0, 10**6)
        frac = str(rng.randint(0, 10**6)).zfill(rng.randint(1, 8))
        exp = rng.randint(-20, 20)
        sign = rng.choice(["", "-", "+"])
        text = f"{sign}{whole}.{frac}e{exp}"
        assert as_fraction(Rational.from_str(text)) == Fraction(text)


# --------------------------------------------------------------- arithmetic


@pytest.mark.parametrize(
    "a, b, op, expected",
    [
        (Rational(1, 2), Rational(1, 3), "+", Rational(5, 6)),
        (Rational(1, 2), Rational(1, 3), "-", Rational(1, 6)),
        (Rational(1, 2), Rational(1, 3), "*", Rational(1, 6)),
        (Rational(1, 2), Rational(1, 3), "/", Rational(3, 2)),
        (Rational(1, 2), 1, "+", Rational(3, 2)),
        (1, Rational(1, 2), "+", Rational(3, 2)),
        (1, Rational(1, 2), "-", Rational(1, 2)),
        (3, Rational(1, 2), "*", Rational(3, 2)),
        (1, Rational(1, 4), "/", Rational(4)),
        (Rational(-1, 2), Rational(1, 2), "+", Rational(0)),
    ],
)
def test_arithmetic(a: Rational | int, b: Rational | int, op: str, expected: Rational) -> None:
    result = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "/": lambda: a / b}[op]()
    assert isinstance(result, Rational)
    assert result == expected


def test_division_by_zero() -> None:
    with pytest.raises(ZeroDivisionError):
        Rational(1, 2) / 0
    with pytest.raises(ZeroDivisionError):
        Rational(1, 2) / Rational(0)
    with pytest.raises(ZeroDivisionError):
        1 / Rational(0)


@pytest.mark.parametrize(
    "base, exponent, expected",
    [
        (Rational(2, 3), 2, Rational(4, 9)),
        (Rational(2, 3), -2, Rational(9, 4)),
        (Rational(-2, 3), 3, Rational(-8, 27)),
        (Rational(-2, 3), -3, Rational(-27, 8)),
        (Rational(5), 0, Rational(1)),
        (Rational(0), 0, Rational(1)),
        (Rational(0), 5, Rational(0)),
        (Rational(7, 2), Rational(2), Rational(49, 4)),
        (Rational(10), 30, Rational(10**30)),
    ],
)
def test_power(base: Rational, exponent: int | Rational, expected: Rational) -> None:
    result = base**exponent
    assert result == expected
    assert result.denominator > 0


def test_power_negative_normalised() -> None:
    r = Rational(-2, 3) ** -1
    assert (r.numerator, r.denominator) == (-3, 2)


def test_rpow() -> None:
    assert 2 ** Rational(3) == Rational(8)
    assert 2 ** Rational(-1) == Rational(1, 2)


def test_power_fractional_exponent_rejected() -> None:
    with pytest.raises(ValueError):
        Rational(4) ** Rational(1, 2)


def test_zero_negative_power() -> None:
    with pytest.raises(ZeroDivisionError):
        Rational(0) ** -1


def test_unsupported_operand() -> None:
    with pytest.raises(TypeError):
        Rational(1) + 1.5  # type: ignore[operator]
    with pytest.raises(TypeError):
        Rational(1) ** 0.5  # type: ignore[operator]
    with pytest.raises(TypeError):
        "a" * Rational(2)  # type: ignore[operator]


def test_unary() -> None:
    r = Rational(-3, 4)
    assert -r == Rational(3, 4)
    assert +r == r
    assert abs(r) == Rational(3, 4)


@pytest.mark.parametrize("seed", range(20))
def test_random_arithmetic_matches_fraction(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(50):
        a, fa = random_pair(rng, big=seed % 2 == 1)
        b, fb = random_pair(rng, big=seed % 3 == 0)
        assert as_fraction(a + b) == fa + fb
        assert as_fraction(a - b) == fa - fb
        assert as_fraction(a * b) == fa * fb
        if fb:
            assert as_fraction(a / b) == fa / fb
        n = rng.randint(-6, 6)
        if fa or n >= 0:
            assert as_fraction(a**n) == fa**n
        k = rng.randint(-10**6, 10**6)
        assert as_fraction(a + k) == fa + k
        assert as_fraction(k - a) == k - fa
        assert as_fraction(k * a) == k * fa


@pytest.mark.parametrize("seed", range(10))
def test_random_comparisons_match_fraction(seed: int) -> None:
    rng = random.Random(1000 + seed)
    for _ in range(100):
        a, fa = random_pair(rng)
        b, fb = random_pair(rng)
        assert (a < b) == (fa < fb)
        assert (a <= b) == (fa <= fb)
        assert (a > b) == (fa > fb)
        assert (a >= b) == (fa >= fb)
        assert (a == b) == (fa == fb)
        assert (a != b) == (fa != fb)


@pytest.mark.parametrize("seed", range(10))
def test_random_hash_matches_fraction(seed: int) -> None:
    rng = random.Random(2000 + seed)
    for _ in range(100):
        a, fa = random_pair(rng, big=seed >= 5)
        assert hash(a) == hash(fa)


@pytest.mark.parametrize("seed", range(5))
def test_random_conversions_match_fraction(seed: int) -> None:
    rng = random.Random(3000 + seed)
    for _ in range(100):
        a, fa = random_pair(rng, big=True)
        assert int(a) == int(fa)
        assert a.floor() == fa.__floor__()
        assert a.ceil() == fa.__ceil__()
        assert float(a) == float(fa)
        assert str(a) == str(fa)


# -------------------------------------------------------------- comparisons


def test_compare_with_int() -> None:
    assert Rational(3, 2) > 1
    assert Rational(3, 2) < 2
    assert 1 < Rational(3, 2)
    assert Rational(4, 2) == 2
    assert 2 == Rational(4, 2)
    assert Rational(1, 2) != 0


def test_compare_with_other_types() -> None:
    assert Rational(1, 2) != "1/2"
    assert (Rational(1, 2) == 0.5) is False
    with pytest.raises(TypeError):
        Rational(1) < "x"  # type: ignore[operator]


def test_sorting() -> None:
    values = [Rational(1, 2), Rational(-1, 3), Rational(2), Rational(0)]
    assert sorted(values) == [Rational(-1, 3), Rational(0), Rational(1, 2), Rational(2)]


# ------------------------------------------------------------------ hashing


@pytest.mark.parametrize("n", [0, 1, -1, 2, -2, 10**30, -(10**30), 2**61 - 1, 2**61])
def test_hash_equals_int_hash(n: int) -> None:
    assert hash(Rational(n)) == hash(n)


def test_usable_as_dict_key() -> None:
    table = {Rational(1, 2): "half", Rational(2): "two"}
    assert table[Rational(2, 4)] == "half"
    assert table[2] == "two"  # type: ignore[index]
    assert len({Rational(1, 3), Rational(2, 6), Rational(3, 9)}) == 1


# -------------------------------------------------------------- conversions


@pytest.mark.parametrize(
    "r, truncated, floor, ceil",
    [
        (Rational(7, 2), 3, 3, 4),
        (Rational(-7, 2), -3, -4, -3),
        (Rational(4), 4, 4, 4),
        (Rational(-4), -4, -4, -4),
        (Rational(1, 3), 0, 0, 1),
        (Rational(-1, 3), 0, -1, 0),
    ],
)
def test_rounding_conversions(r: Rational, truncated: int, floor: int, ceil: int) -> None:
    import math

    assert int(r) == truncated
    assert math.floor(r) == floor == r.floor()
    assert math.ceil(r) == ceil == r.ceil()


def test_bool_float_str_repr() -> None:
    assert not Rational(0)
    assert Rational(1, 5)
    assert float(Rational(1, 4)) == 0.25
    assert str(Rational(3, 4)) == "3/4"
    assert str(Rational(-6, 3)) == "-2"
    assert repr(Rational(3, 4)) == "Rational(3, 4)"
    assert Rational(3).is_integer()
    assert not Rational(3, 2).is_integer()
