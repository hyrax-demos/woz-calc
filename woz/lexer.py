"""Tokenizer for woz expressions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["LexError", "Token", "TokenKind", "WozError", "tokenize"]


class WozError(Exception):
    """Base class for every error the calculator reports to the user."""

    def __init__(self, message: str, position: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.position = position

    def __str__(self) -> str:
        if self.position is None:
            return self.message
        return f"{self.message} (at column {self.position + 1})"


class LexError(WozError):
    """Raised for characters or number literals the lexer can't read."""


class TokenKind(Enum):
    NUMBER = "number"
    NAME = "name"
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    POWER = "**"
    LPAREN = "("
    RPAREN = ")"
    COMMA = ","
    ASSIGN = "="
    EOF = "end of input"


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    position: int


_SINGLE: dict[str, TokenKind] = {
    "+": TokenKind.PLUS,
    "-": TokenKind.MINUS,
    "/": TokenKind.SLASH,
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    ",": TokenKind.COMMA,
    "=": TokenKind.ASSIGN,
    "^": TokenKind.POWER,
}


def _is_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def _is_name_start(ch: str) -> bool:
    return ch == "_" or ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _read_number(source: str, start: int) -> int:
    """Return the index just past the number literal starting at ``start``."""
    i = start
    n = len(source)
    while i < n and _is_digit(source[i]):
        i += 1
    if i < n and source[i] == ".":
        i += 1
        while i < n and _is_digit(source[i]):
            i += 1
    if i < n and source[i] in "eE":
        j = i + 1
        if j < n and source[j] in "+-":
            j += 1
        if j < n and _is_digit(source[j]):
            while j < n and _is_digit(source[j]):
                j += 1
            i = j
        else:
            raise LexError("malformed exponent in number", i)
    if i < n and (_is_name_start(source[i]) or source[i] == "."):
        raise LexError(f"unexpected {source[i]!r} after number", i)
    return i


def tokenize(source: str) -> list[Token]:
    """Split ``source`` into tokens, always ending with an EOF token.

    ``^`` is accepted as a synonym for ``**``.
    """
    tokens: list[Token] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in " \t\r\n":
            i += 1
        elif _is_digit(ch) or (ch == "." and i + 1 < n and _is_digit(source[i + 1])):
            end = _read_number(source, i)
            tokens.append(Token(TokenKind.NUMBER, source[i:end], i))
            i = end
        elif _is_name_start(ch):
            end = i + 1
            while end < n and (_is_name_start(source[end]) or _is_digit(source[end])):
                end += 1
            tokens.append(Token(TokenKind.NAME, source[i:end], i))
            i = end
        elif ch == "*":
            if source.startswith("**", i):
                tokens.append(Token(TokenKind.POWER, "**", i))
                i += 2
            else:
                tokens.append(Token(TokenKind.STAR, "*", i))
                i += 1
        elif ch in _SINGLE:
            tokens.append(Token(_SINGLE[ch], ch, i))
            i += 1
        else:
            raise LexError(f"unexpected character {ch!r}", i)
    tokens.append(Token(TokenKind.EOF, "", n))
    return tokens
