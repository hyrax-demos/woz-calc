"""Tests for woz.parser."""

from __future__ import annotations

import pytest

from woz.parser import (
    Assign,
    Binary,
    Call,
    Node,
    Num,
    ParseError,
    Unary,
    Var,
    parse,
)
from woz.rational import Rational


def n(v: int | str) -> Num:
    return Num(Rational.from_str(str(v)))


def show(node: Node) -> str:
    """Fully parenthesised rendering, to make precedence visible."""
    if isinstance(node, Num):
        return str(node.value)
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Unary):
        return f"({node.op}{show(node.operand)})"
    if isinstance(node, Binary):
        return f"({show(node.left)} {node.op} {show(node.right)})"
    if isinstance(node, Call):
        return f"{node.name}(" + ", ".join(show(a) for a in node.args) + ")"
    return f"({node.name} = {show(node.value)})"


@pytest.mark.parametrize(
    "src, expected",
    [
        ("1 + 2 * 3", "(1 + (2 * 3))"),
        ("(1 + 2) * 3", "((1 + 2) * 3)"),
        ("1 - 2 - 3", "((1 - 2) - 3)"),
        ("8 / 4 / 2", "((8 / 4) / 2)"),
        ("2 ** 3 ** 2", "(2 ** (3 ** 2))"),
        ("2 ^ 3 ^ 2", "(2 ** (3 ** 2))"),
        ("-2 ** 2", "(-(2 ** 2))"),
        ("2 ** -1", "(2 ** (-1))"),
        ("--3", "(-(-3))"),
        ("+-3", "(+(-3))"),
        ("-x * y", "((-x) * y)"),
        ("a * b ** c", "(a * (b ** c))"),
        ("2 * -3", "(2 * (-3))"),
        ("f()", "f()"),
        ("f(1)", "f(1)"),
        ("f(1, 2 + 3)", "f(1, (2 + 3))"),
        ("f(g(x))", "f(g(x))"),
        ("sqrt(2) ** 2", "(sqrt(2) ** 2)"),
        ("-f(x)", "(-f(x))"),
        ("x = 1 + 2", "(x = (1 + 2))"),
        ("x = y = 3", "(x = (y = 3))"),
        ("1.5 + 3/4", "(3/2 + (3 / 4))"),
        ("((1))", "1"),
        ("(f)(1)", "f(1)"),
    ],
)
def test_structure(src: str, expected: str) -> None:
    assert show(parse(src)) == expected


def test_nodes() -> None:
    assert parse("x") == Var("x")
    assert parse("2") == n(2)
    assert parse("1e-3") == Num(Rational(1, 1000))
    assert parse("x = 2") == Assign("x", n(2))
    assert parse("f(1)") == Call("f", (n(1),))


@pytest.mark.parametrize(
    "src",
    [
        "",
        "1 +",
        "(1",
        "1)",
        "1 2",
        "* 3",
        "f(1,",
        "f(1 2)",
        "1 = 2",
        "2(3)",
        "(a + b) = 3",
        ",",
        "x =",
    ],
)
def test_errors(src: str) -> None:
    with pytest.raises(ParseError):
        parse(src)


def test_error_position() -> None:
    with pytest.raises(ParseError) as info:
        parse("1 + )")
    assert info.value.pos == 4
