"""Tests for woz.bigdec significant-digit formatting."""

from __future__ import annotations

import pytest

from woz.bigdec import decompose, exponent10, format_sig, round_sig
from woz.rational import Rational

R = Rational.from_str


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", 0),
        ("9.999", 0),
        ("10", 1),
        ("0.1", -1),
        ("0.0999", -2),
        ("-12345", 4),
        ("1e-100", -100),
        ("1e100", 100),
        ("1/3", -1),
        ("99999999999999999999/10", 18),
    ],
)
def test_exponent10(value: str, expected: int) -> None:
    assert exponent10(R(value)) == expected


def test_exponent10_zero_raises() -> None:
    with pytest.raises(ValueError):
        exponent10(Rational(0))


@pytest.mark.parametrize(
    "value, digits, expected",
    [
        # round-half-even at exact ties
        ("2.5", 1, "2"),
        ("3.5", 1, "4"),
        ("-2.5", 1, "-2"),
        ("-3.5", 1, "-4"),
        ("0.125", 2, "0.12"),
        ("0.135", 2, "0.14"),
        ("1.0005", 4, "1.000"),
        ("1.0015", 4, "1.002"),
        ("25", 1, "2e+01"),
        ("35", 1, "4e+01"),
        # just past a tie rounds away
        ("2.5000001", 1, "3"),
        ("0.12500001", 2, "0.13"),
        # carry that changes the exponent
        ("9.99", 2, "10"),
        ("9.96", 2, "10"),
        ("0.0999", 1, "0.1"),
        ("999.5", 3, "1.00e+03"),
        # plain values
        ("1/3", 5, "0.33333"),
        ("2/3", 5, "0.66667"),
        ("-1/7", 6, "-0.142857"),
        ("123.456", 5, "123.46"),
        ("0.00012345", 3, "0.000123"),
        ("0.000012345", 3, "1.23e-05"),
        ("123456", 3, "1.23e+05"),
        ("123456", 6, "123456"),
        ("100", 3, "100"),
        ("1", 5, "1.0000"),
        ("0", 3, "0.00"),
        ("22/7", 10, "3.142857143"),
    ],
)
def test_format_sig(value: str, digits: int, expected: str) -> None:
    assert format_sig(R(value), digits) == expected


@pytest.mark.parametrize(
    "value, digits, expected",
    [
        ("1", 5, "1"),
        ("1.5", 5, "1.5"),
        ("100", 5, "100"),
        ("0.000012", 4, "1.2e-05"),
        ("1/8", 10, "0.125"),
    ],
)
def test_format_sig_strip_zeros(value: str, digits: int, expected: str) -> None:
    assert format_sig(R(value), digits, strip_zeros=True) == expected


@pytest.mark.parametrize(
    "value, digits, scientific, expected",
    [
        ("123.456", 4, True, "1.235e+02"),
        ("0.5", 1, True, "5e-01"),
        ("1e-20", 3, False, "0.0000000000000000000100"),
        ("1e10", 2, False, "10000000000"),
    ],
)
def test_format_sig_forced_notation(
    value: str, digits: int, scientific: bool, expected: str
) -> None:
    assert format_sig(R(value), digits, scientific=scientific) == expected


@pytest.mark.parametrize(
    "value, digits, expected",
    [
        ("2.5", 1, "2"),
        ("1/3", 3, "333/1000"),
        ("-2/3", 2, "-67/100"),
        ("123456", 2, "120000"),
        ("0", 4, "0"),
    ],
)
def test_round_sig(value: str, digits: int, expected: str) -> None:
    assert round_sig(R(value), digits) == R(expected)


def test_decompose() -> None:
    assert decompose(R("-0.0123456"), 3) == (True, 123, -2)
    assert decompose(Rational(0), 3) == (False, 0, 0)


@pytest.mark.parametrize("digits", [1, 10, 30, 60])
def test_format_one_third(digits: int) -> None:
    assert format_sig(Rational(1, 3), digits) == "0." + "3" * digits


@pytest.mark.parametrize("digits", [0, -1])
def test_bad_digits(digits: int) -> None:
    with pytest.raises(ValueError):
        format_sig(Rational(1), digits)


def test_non_int_digits() -> None:
    with pytest.raises(TypeError):
        format_sig(Rational(1), 2.0)  # type: ignore[arg-type]
