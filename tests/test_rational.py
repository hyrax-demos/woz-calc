import math
import pickle
import random
from fractions import Fraction

import pytest

from woz.rational import Rational, as_rational, integer_root_floor


def F(r: Rational) -> Fraction:
    return Fraction(r.numerator, r.denominator)


def R(f: Fraction) -> Rational:
    return Rational(f.numerator, f.denominator)


# ----------------------------------------------------------------- construction
@pytest.mark.parametrize(
    "num, den, expected",
    [
        (6, 8, (3, 4)),
        (-6, 8, (-3, 4)),
        (6, -8, (-3, 4)),
        (-6, -8, (3, 4)),
        (0, 5, (0, 1)),
        (0, -5, (0, 1)),
        (7, 1, (7, 1)),
        (10**30, 10**29, (10, 1)),
    ],
)
def test_normalisation(num, den, expected):
    r = Rational(num, den)
    assert (r.numerator, r.denominator) == expected


def test_default_is_zero():
    assert Rational() == 0


def test_rational_arguments():
    assert Rational(Rational(1, 2), Rational(3, 4)) == Rational(2, 3)
    assert Rational(Rational(5, 2)) == Rational(5, 2)


def test_zero_denominator():
    with pytest.raises(ZeroDivisionError):
        Rational(1, 0)


@pytest.mark.parametrize("bad", [1.5, "1", None, [1]])
def test_constructor_rejects_non_integers(bad):
    with pytest.raises(TypeError):
        Rational(bad)
    with pytest.raises(TypeError):
        Rational(1, bad)


def test_immutable():
    r = Rational(1, 2)
    with pytest.raises(AttributeError):
        r._num = 5  # type: ignore[misc]


# -------------------------------------------------------------------- from_str
@pytest.mark.parametrize(
    "text, expected",
    [
        ("3/4", Rational(3, 4)),
        ("-1.25", Rational(-5, 4)),
        ("1e-3", Rational(1, 1000)),
        ("6/8", Rational(3, 4)),
        ("-3/4", Rational(-3, 4)),
        ("3/-4", Rational(-3, 4)),
        ("+7", Rational(7)),
        ("0", Rational(0)),
        ("-0.0", Rational(0)),
        (".5", Rational(1, 2)),
        ("5.", Rational(5)),
        ("2.5e2", Rational(250)),
        ("1E+3", Rational(1000)),
        ("  42  ", Rational(42)),
        ("1.5/2.5", Rational(3, 5)),
        ("1e-3/1e-6", Rational(1000)),
        ("0.000001", Rational(1, 10**6)),
        ("123456789012345678901234567890", Rational(123456789012345678901234567890)),
        ("-12.5e-1", Rational(-5, 4)),
    ],
)
def test_from_str(text, expected):
    assert Rational.from_str(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "1/", "/2", "1.2.3", "1e", "--1", "1/2/3", "e5", "0x10", "1 2"])
def test_from_str_invalid(text):
    with pytest.raises(ValueError):
        Rational.from_str(text)


def test_from_str_zero_denominator():
    with pytest.raises(ZeroDivisionError):
        Rational.from_str("1/0")


def test_from_str_type_error():
    with pytest.raises(TypeError):
        Rational.from_str(3)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["3/4", "-1.25", "1e-3", "22/7", "-0.001", "12345.678e3"])
def test_from_str_matches_fraction(text):
    assert F(Rational.from_str(text)) == Fraction(text)


# ------------------------------------------------------------------ arithmetic
@pytest.mark.parametrize(
    "a, b, op, expected",
    [
        (Rational(1, 2), Rational(1, 3), "+", Rational(5, 6)),
        (Rational(1, 2), Rational(1, 3), "-", Rational(1, 6)),
        (Rational(2, 3), Rational(9, 4), "*", Rational(3, 2)),
        (Rational(2, 3), Rational(4, 9), "/", Rational(3, 2)),
        (Rational(1, 2), 1, "+", Rational(3, 2)),
        (1, Rational(1, 2), "+", Rational(3, 2)),
        (1, Rational(1, 4), "-", Rational(3, 4)),
        (3, Rational(1, 4), "*", Rational(3, 4)),
        (1, Rational(1, 4), "/", Rational(4)),
        (Rational(-1, 2), Rational(1, 2), "+", Rational(0)),
    ],
)
def test_arithmetic(a, b, op, expected):
    result = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "/": lambda: a / b}[op]()
    assert result == expected
    assert isinstance(result, Rational)


def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        Rational(1) / 0
    with pytest.raises(ZeroDivisionError):
        1 / Rational(0)


def test_float_operands_rejected():
    with pytest.raises(TypeError):
        Rational(1) + 1.5  # type: ignore[operator]
    with pytest.raises(TypeError):
        1.5 * Rational(1)  # type: ignore[operator]


@pytest.mark.parametrize(
    "base, exponent, expected",
    [
        (Rational(2, 3), 3, Rational(8, 27)),
        (Rational(2, 3), -2, Rational(9, 4)),
        (Rational(-2, 3), -3, Rational(-27, 8)),
        (Rational(5), 0, Rational(1)),
        (Rational(0), 0, Rational(1)),
        (Rational(0), 5, Rational(0)),
        (Rational(4, 9), Rational(1, 2), Rational(2, 3)),
        (Rational(8), Rational(2, 3), Rational(4)),
        (Rational(-8, 27), Rational(1, 3), Rational(-2, 3)),
        (Rational(16), Rational(-3, 4), Rational(1, 8)),
        (Rational(1, 4), Rational(-1, 2), Rational(2)),
    ],
)
def test_pow(base, exponent, expected):
    assert base**exponent == expected


def test_rpow():
    assert 2 ** Rational(3) == 8
    assert 4 ** Rational(1, 2) == 2


def test_pow_errors():
    with pytest.raises(ZeroDivisionError):
        Rational(0) ** -1
    with pytest.raises(ValueError):
        Rational(2) ** Rational(1, 2)
    with pytest.raises(ValueError):
        Rational(-4) ** Rational(1, 2)


def test_unary():
    r = Rational(-3, 4)
    assert -r == Rational(3, 4)
    assert +r is r
    assert abs(r) == Rational(3, 4)


# ----------------------------------------------------------------- conversions
@pytest.mark.parametrize("num, den", [(7, 2), (-7, 2), (6, 3), (-6, 3), (1, 3), (-1, 3), (0, 1)])
def test_integer_conversions(num, den):
    r, f = Rational(num, den), Fraction(num, den)
    assert int(r) == int(f)
    assert math.floor(r) == math.floor(f)
    assert math.ceil(r) == math.ceil(f)
    assert math.trunc(r) == math.trunc(f)
    assert float(r) == float(f)


def test_bool_and_is_integer():
    assert not Rational(0)
    assert Rational(1, 7)
    assert Rational(4, 2).is_integer()
    assert not Rational(1, 2).is_integer()


def test_str_repr():
    assert str(Rational(3, 4)) == "3/4"
    assert str(Rational(-5)) == "-5"
    assert repr(Rational(-3, 4)) == "Rational(-3, 4)"


def test_pickle_roundtrip():
    r = Rational(-22, 7)
    assert pickle.loads(pickle.dumps(r)) == r


# ---------------------------------------------------------- comparisons & hash
def test_comparisons():
    a, b = Rational(1, 3), Rational(1, 2)
    assert a < b and a <= b and b > a and b >= a
    assert a != b and a == Rational(2, 6)
    assert Rational(2) == 2 and 2 == Rational(2)
    assert Rational(1, 2) < 1 and 0 < Rational(1, 2)
    assert Rational(1, 2) != "1/2"


def test_compare_with_float_unsupported():
    with pytest.raises(TypeError):
        Rational(1) < 1.5  # type: ignore[operator]


@pytest.mark.parametrize("n", [0, 1, -1, 2, -2, 10**40, -(10**40)])
def test_hash_matches_int(n):
    assert hash(Rational(n)) == hash(n)


@pytest.mark.parametrize("num, den", [(1, 2), (-1, 2), (22, 7), (-3, 10**20), (1, 2**61 - 1), (5, 3 * (2**61 - 1))])
def test_hash_matches_fraction(num, den):
    assert hash(Rational(num, den)) == hash(Fraction(num, den))


def test_hash_consistent_for_equal_values():
    assert hash(Rational(2, 4)) == hash(Rational(1, 2))
    assert len({Rational(1, 2), Rational(2, 4), Rational(3, 6), Rational(1, 3)}) == 2


# ------------------------------------------------------------------- helpers
def test_as_rational():
    assert as_rational(3) == Rational(3)
    assert as_rational("1/2") == Rational(1, 2)
    r = Rational(1, 5)
    assert as_rational(r) is r
    with pytest.raises(TypeError):
        as_rational(1.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("n, k, expected", [(0, 2, 0), (1, 3, 1), (15, 2, 3), (16, 2, 4), (26, 3, 2), (27, 3, 3), (10**40, 4, 10**10), (7, 1, 7)])
def test_integer_root_floor(n, k, expected):
    assert integer_root_floor(n, k) == expected


def test_integer_root_floor_invalid():
    with pytest.raises(ValueError):
        integer_root_floor(-1, 2)
    with pytest.raises(ValueError):
        integer_root_floor(4, 0)


# ---------------------------------------------------- randomized cross-checks
def _random_fraction(rng: random.Random) -> Fraction:
    size = rng.choice([10, 1000, 10**12, 10**30])
    return Fraction(rng.randint(-size, size), rng.randint(1, size))


@pytest.mark.parametrize("seed", range(40))
def test_random_arithmetic_matches_fraction(seed):
    rng = random.Random(seed)
    for _ in range(25):
        fa, fb = _random_fraction(rng), _random_fraction(rng)
        a, b = R(fa), R(fb)
        assert F(a + b) == fa + fb
        assert F(a - b) == fa - fb
        assert F(a * b) == fa * fb
        if fb:
            assert F(a / b) == fa / fb
        assert (a < b) == (fa < fb)
        assert (a <= b) == (fa <= fb)
        assert (a == b) == (fa == fb)
        assert hash(a) == hash(fa)
        assert math.floor(a) == math.floor(fa)
        assert math.ceil(a) == math.ceil(fa)
        assert int(a) == int(fa)


@pytest.mark.parametrize("seed", range(20))
def test_random_powers_match_fraction(seed):
    rng = random.Random(1000 + seed)
    for _ in range(10):
        fa = _random_fraction(rng)
        n = rng.randint(-6, 6)
        if not fa and n < 0:
            continue
        assert F(R(fa) ** n) == fa**n


@pytest.mark.parametrize("seed", range(20))
def test_random_from_str_matches_fraction(seed):
    rng = random.Random(2000 + seed)
    for _ in range(10):
        sign = rng.choice(["", "-", "+"])
        whole = str(rng.randint(0, 10**6))
        frac = str(rng.randint(0, 10**6))
        exp = rng.randint(-20, 20)
        text = f"{sign}{whole}.{frac}e{exp}"
        assert F(Rational.from_str(text)) == Fraction(text)
        num, den = rng.randint(-1000, 1000), rng.randint(1, 1000)
        assert F(Rational.from_str(f"{num}/{den}")) == Fraction(num, den)


@pytest.mark.parametrize("seed", range(10))
def test_random_exact_roots(seed):
    rng = random.Random(3000 + seed)
    for _ in range(10):
        k = rng.randint(2, 5)
        f = Fraction(rng.randint(1, 10**6), rng.randint(1, 10**6))
        assert F(R(f**k) ** Rational(1, k)) == f
