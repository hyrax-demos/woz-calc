"""Tokenizer for the woz expression language."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TokenKind(Enum):
    NUMBER = "number"
    NAME = "name"
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    POW = "**"
    LPAREN = "("
    RPAREN = ")"
    COMMA = ","
    ASSIGN = "="
    EOF = "eof"


class WozError(Exception):
    """Base class for every error raised by the expression language."""


class LexError(WozError):
    """Raised on an unrecognised character or malformed number."""

    def __init__(self, message: str, pos: int) -> None:
        super().__init__(f"{message} at position {pos}")
        self.pos = pos


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    pos: int


_SINGLE: dict[str, TokenKind] = {
    "+": TokenKind.PLUS,
    "-": TokenKind.MINUS,
    "/": TokenKind.SLASH,
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    ",": TokenKind.COMMA,
    "=": TokenKind.ASSIGN,
}


def _scan_number(src: str, i: int) -> int:
    """Return the end index of the number literal starting at ``i``."""
    n = len(src)
    start = i
    while i < n and src[i].isdigit():
        i += 1
    if i < n and src[i] == ".":
        i += 1
        while i < n and src[i].isdigit():
            i += 1
    if i - start == 1 and src[start] == ".":
        raise LexError("malformed number", start)
    if i < n and src[i] in "eE":
        j = i + 1
        if j < n and src[j] in "+-":
            j += 1
        if j < n and src[j].isdigit():
            while j < n and src[j].isdigit():
                j += 1
            i = j
        else:
            raise LexError("malformed exponent", i)
    if i < n and (src[i].isalpha() or src[i] == "_" or src[i] == "."):
        raise LexError("malformed number", start)
    return i


def tokenize(src: str) -> list[Token]:
    """Split ``src`` into tokens, always ending with an ``EOF`` token."""
    tokens: list[Token] = []
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if ch.isspace():
            i += 1
        elif ch.isdigit() or (ch == "." and i + 1 < n and src[i + 1].isdigit()):
            end = _scan_number(src, i)
            tokens.append(Token(TokenKind.NUMBER, src[i:end], i))
            i = end
        elif ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            tokens.append(Token(TokenKind.NAME, src[i:j], i))
            i = j
        elif ch == "*":
            if i + 1 < n and src[i + 1] == "*":
                tokens.append(Token(TokenKind.POW, "**", i))
                i += 2
            else:
                tokens.append(Token(TokenKind.STAR, "*", i))
                i += 1
        elif ch == "^":
            tokens.append(Token(TokenKind.POW, "^", i))
            i += 1
        elif ch in _SINGLE:
            tokens.append(Token(_SINGLE[ch], ch, i))
            i += 1
        else:
            raise LexError(f"unexpected character {ch!r}", i)
    tokens.append(Token(TokenKind.EOF, "", n))
    return tokens
