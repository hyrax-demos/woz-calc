"""Tests for woz.lexer."""

from __future__ import annotations

import pytest

from woz.lexer import LexError, TokenKind, WozError, tokenize

K = TokenKind


def kinds(src: str) -> list[TokenKind]:
    return [t.kind for t in tokenize(src)]


def texts(src: str) -> list[str]:
    return [t.text for t in tokenize(src)[:-1]]


def test_empty() -> None:
    assert kinds("") == [K.EOF]
    assert kinds("   ") == [K.EOF]


@pytest.mark.parametrize(
    "src, expected",
    [
        ("+", K.PLUS),
        ("-", K.MINUS),
        ("*", K.STAR),
        ("/", K.SLASH),
        ("**", K.POW),
        ("^", K.POW),
        ("(", K.LPAREN),
        (")", K.RPAREN),
        (",", K.COMMA),
        ("=", K.ASSIGN),
        ("x", K.NAME),
        ("42", K.NUMBER),
    ],
)
def test_single_tokens(src: str, expected: TokenKind) -> None:
    assert kinds(src) == [expected, K.EOF]


@pytest.mark.parametrize(
    "src", ["0", "42", "3.14", ".5", "5.", "1e10", "1E-3", "2.5e+7", "007"]
)
def test_numbers(src: str) -> None:
    toks = tokenize(src)
    assert toks[0].kind is K.NUMBER and toks[0].text == src


@pytest.mark.parametrize("src", ["x", "_x", "abc_12", "Pi", "sqrt"])
def test_names(src: str) -> None:
    assert texts(src) == [src]


def test_expression_and_positions() -> None:
    toks = tokenize("x = 2**-3 + f(a, 1.5)")
    assert [t.text for t in toks[:-1]] == [
        "x",
        "=",
        "2",
        "**",
        "-",
        "3",
        "+",
        "f",
        "(",
        "a",
        ",",
        "1.5",
        ")",
    ]
    assert [t.pos for t in toks[:4]] == [0, 2, 4, 5]
    assert toks[-1].pos == len("x = 2**-3 + f(a, 1.5)")


def test_star_vs_pow() -> None:
    assert kinds("2*3**4") == [K.NUMBER, K.STAR, K.NUMBER, K.POW, K.NUMBER, K.EOF]
    assert kinds("2***3") == [K.NUMBER, K.POW, K.STAR, K.NUMBER, K.EOF]


@pytest.mark.parametrize(
    "src, pos", [("1 $ 2", 2), ("@", 0), ("a # b", 2), ("1 & 2", 2)]
)
def test_bad_character(src: str, pos: int) -> None:
    with pytest.raises(LexError) as info:
        tokenize(src)
    assert info.value.pos == pos


@pytest.mark.parametrize("src", ["1e", "1e+", "1.2.3", "12abc", "3x", "1..2"])
def test_malformed_numbers(src: str) -> None:
    with pytest.raises(LexError):
        tokenize(src)


def test_lex_error_is_woz_error() -> None:
    assert issubclass(LexError, WozError)
