"""A Pratt parser for the woz expression language.

Grammar (lowest to highest binding)::

    name = expr            right-associative assignment
    a + b, a - b           left-associative
    a * b, a / b           left-associative
    -a, +a                 prefix unary (so -2**2 == -(2**2))
    a ** b                 right-associative (2**3**2 == 2**(3**2))
    f(a, ...), name, 1.5, (expr)
"""

from __future__ import annotations

from dataclasses import dataclass

from woz.lexer import Token, TokenKind, WozError, tokenize
from woz.rational import Rational


class ParseError(WozError):
    """Raised for syntactically invalid input."""

    def __init__(self, message: str, position: int) -> None:
        super().__init__(f"{message} at position {position}")
        self.position = position


@dataclass(frozen=True)
class Number:
    value: Rational


@dataclass(frozen=True)
class Name:
    id: str


@dataclass(frozen=True)
class Unary:
    op: str
    operand: Node


@dataclass(frozen=True)
class Binary:
    op: str
    left: Node
    right: Node


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple[Node, ...]


@dataclass(frozen=True)
class Assign:
    name: str
    value: Node


Node = Number | Name | Unary | Binary | Call | Assign

ASSIGN_BP = 10
UNARY_BP = 40

#: operator -> (left binding power, right binding power)
_INFIX: dict[str, tuple[int, int]] = {
    "=": (ASSIGN_BP, ASSIGN_BP - 1),
    "+": (20, 21),
    "-": (20, 21),
    "*": (30, 31),
    "/": (30, 31),
    "**": (50, 49),
}


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._index = 0

    @property
    def _current(self) -> Token:
        return self._tokens[self._index]

    def _advance(self) -> Token:
        token = self._tokens[self._index]
        if token.kind is not TokenKind.EOF:
            self._index += 1
        return token

    def _expect(self, kind: TokenKind) -> Token:
        token = self._current
        if token.kind is not kind:
            raise ParseError(f"expected {kind.value!r}, found {token}", token.position)
        return self._advance()

    def parse(self) -> Node:
        if self._current.kind is TokenKind.EOF:
            raise ParseError("empty expression", self._current.position)
        node = self.expression(0)
        token = self._current
        if token.kind is not TokenKind.EOF:
            raise ParseError(f"unexpected {token}", token.position)
        return node

    def expression(self, min_bp: int) -> Node:
        left = self._prefix()
        while True:
            token = self._current
            if token.kind is not TokenKind.OP:
                break
            lbp, rbp = _INFIX[token.text]
            if lbp <= min_bp:
                break
            self._advance()
            if token.text == "=":
                if not isinstance(left, Name):
                    raise ParseError("can only assign to a variable name", token.position)
                left = Assign(left.id, self.expression(rbp))
            else:
                left = Binary(token.text, left, self.expression(rbp))
        return left

    def _prefix(self) -> Node:
        token = self._advance()
        if token.kind is TokenKind.NUMBER:
            return Number(Rational.from_str(token.text))
        if token.kind is TokenKind.NAME:
            if self._current.kind is TokenKind.LPAREN:
                return Call(token.text, self._arguments())
            return Name(token.text)
        if token.kind is TokenKind.LPAREN:
            node = self.expression(0)
            self._expect(TokenKind.RPAREN)
            return node
        if token.kind is TokenKind.OP and token.text in ("-", "+"):
            return Unary(token.text, self.expression(UNARY_BP))
        if token.kind is TokenKind.EOF:
            raise ParseError("unexpected end of input", token.position)
        raise ParseError(f"unexpected {token}", token.position)

    def _arguments(self) -> tuple[Node, ...]:
        self._expect(TokenKind.LPAREN)
        args: list[Node] = []
        if self._current.kind is not TokenKind.RPAREN:
            while True:
                args.append(self.expression(0))
                if self._current.kind is not TokenKind.COMMA:
                    break
                self._advance()
        self._expect(TokenKind.RPAREN)
        return tuple(args)


def parse(source: str) -> Node:
    """Tokenize and parse ``source`` into an AST."""
    return Parser(tokenize(source)).parse()
