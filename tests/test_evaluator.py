"""Tests for woz.evaluator."""

from __future__ import annotations

import pytest

from woz.evaluator import (
    ArityError,
    DivisionByZeroError,
    EvalError,
    Evaluator,
    MathDomainError,
    ReadOnlyNameError,
    UndefinedNameError,
)
from woz.lexer import LexError, WozError
from woz.parser import ParseError
from woz.rational import Rational


def run(src: str, digits: int = 30) -> str:
    ev = Evaluator(digits)
    return ev.format(ev.evaluate(src))


@pytest.mark.parametrize(
    "src, expected",
    [
        ("1 + 2", "3"),
        ("1/3 + 1/6", "1/2"),
        ("0.1 + 0.2", "3/10"),
        ("2 ** 10", "1024"),
        ("2 ** -2", "1/4"),
        ("2 ** 3 ** 2", "512"),
        ("-2 ** 2", "-4"),
        ("(-2) ** 2", "4"),
        ("4 ** (1/2)", "2"),
        ("(8/27) ** (2/3)", "4/9"),
        ("1e-3 * 1000", "1"),
        ("abs(-3/4)", "3/4"),
        ("7 - 10", "-3"),
        ("+5", "5"),
        ("2 ^ 5", "32"),
    ],
)
def test_exact(src: str, expected: str) -> None:
    assert run(src) == expected


@pytest.mark.parametrize(
    "src, digits, expected",
    [
        ("sqrt(2)", 10, "1.414213562"),
        ("pi", 10, "3.141592654"),
        ("e", 5, "2.7183"),
        ("ln(2)", 1, "0.7"),
        ("exp(0)", 5, "1"),
        ("2 * pi", 5, "6.2832"),
        ("sin(1/2)", 10, "0.4794255386"),
        ("cos(0) + 1", 5, "2"),
        ("atan(1) * 4", 30, "3.14159265358979323846264338328"),
        ("sqrt(1/4)", 5, "0.5"),
    ],
)
def test_approximate(src: str, digits: int, expected: str) -> None:
    assert run(src, digits) == expected


def test_variables_and_assignment() -> None:
    ev = Evaluator()
    assert ev.format(ev.evaluate("x = 3/4")) == "3/4"
    assert ev.format(ev.evaluate("y = x * 4")) == "3"
    assert ev.format(ev.evaluate("a = b = 2")) == "2"
    assert ev.variables["b"].rational == Rational(2)
    assert ev.format(ev.evaluate("x = x + 1")) == "7/4"


def test_approx_flag_propagates() -> None:
    ev = Evaluator(5)
    v = ev.evaluate("t = pi")
    assert not v.exact
    assert not ev.evaluate("t * 2").exact
    assert ev.evaluate("1/3").exact


@pytest.mark.parametrize(
    "src, error",
    [
        ("1 / 0", DivisionByZeroError),
        ("0 ** -1", DivisionByZeroError),
        ("x", UndefinedNameError),
        ("foo(1)", UndefinedNameError),
        ("sqrt(1, 2)", ArityError),
        ("sqrt()", ArityError),
        ("abs(1, 2)", ArityError),
        ("ln(0)", MathDomainError),
        ("sqrt(-1)", MathDomainError),
        ("(-4) ** (1/2)", MathDomainError),
        ("2 ** (1/2)", MathDomainError),
        ("pi = 3", ReadOnlyNameError),
        ("sqrt = 3", ReadOnlyNameError),
        ("sqrt", EvalError),
        ("pi(2)", EvalError),
        ("exp(10 ** 7)", EvalError),
        ("10 ** 10 ** 10", EvalError),
        ("1 +", ParseError),
        ("1 $ 2", LexError),
    ],
)
def test_errors(src: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        run(src)


def test_calling_a_variable() -> None:
    ev = Evaluator()
    ev.evaluate("f = 2")
    with pytest.raises(EvalError):
        ev.evaluate("f(1)")


def test_error_hierarchy() -> None:
    for cls in (
        EvalError,
        UndefinedNameError,
        ArityError,
        MathDomainError,
        DivisionByZeroError,
        ReadOnlyNameError,
        ParseError,
        LexError,
    ):
        assert issubclass(cls, WozError)


@pytest.mark.parametrize("digits", [0, 61, True])
def test_bad_digits(digits: int) -> None:
    with pytest.raises(EvalError):
        Evaluator(digits)


@pytest.mark.parametrize("digits", [1, 10, 30, 60])
def test_digits_setting(digits: int) -> None:
    out = run("sqrt(2)", digits)
    assert len(out.replace(".", "")) == digits
