"""Tokenizer for the woz expression language."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class WozError(Exception):
    """Base class of every error raised by the woz expression language."""


class LexError(WozError):
    """Raised when the input contains a character sequence that is not a token."""

    def __init__(self, message: str, position: int) -> None:
        super().__init__(f"{message} at position {position}")
        self.position = position


class TokenKind(Enum):
    NUMBER = "number"
    NAME = "name"
    OP = "op"
    LPAREN = "("
    RPAREN = ")"
    COMMA = ","
    EOF = "end of input"


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    position: int

    def __str__(self) -> str:
        return self.kind.value if self.kind is TokenKind.EOF else repr(self.text)


_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op>\*\*|[-+*/^=])
  | (?P<lparen>\()
  | (?P<rparen>\))
  | (?P<comma>,)
    """,
    re.VERBOSE,
)

_KINDS = {
    "number": TokenKind.NUMBER,
    "name": TokenKind.NAME,
    "op": TokenKind.OP,
    "lparen": TokenKind.LPAREN,
    "rparen": TokenKind.RPAREN,
    "comma": TokenKind.COMMA,
}


def tokenize(source: str) -> list[Token]:
    """Split ``source`` into tokens, always ending with an ``EOF`` token.

    ``^`` is accepted as a synonym for ``**`` and normalised to ``**``.
    """
    tokens: list[Token] = []
    pos = 0
    while pos < len(source):
        match = _TOKEN_RE.match(source, pos)
        if match is None:
            raise LexError(f"unexpected character {source[pos]!r}", pos)
        group = match.lastgroup
        assert group is not None
        if group != "ws":
            text = match.group()
            if text == "^":
                text = "**"
            tokens.append(Token(_KINDS[group], text, pos))
        pos = match.end()
    tokens.append(Token(TokenKind.EOF, "", len(source)))
    return tokens
