"""Pratt parser that turns woz tokens into an AST.

Grammar, lowest precedence first:

    statement  := NAME '=' statement | expression
    expression := expression ('+' | '-') expression
                | expression ('*' | '/') expression
                | '-' expression | '+' expression
                | expression '**' expression        (right-associative)
                | NAME '(' [expression (',' expression)*] ')'
                | NAME | NUMBER | '(' expression ')'

Unary minus binds looser than ``**``, as in Python: ``-2**2 == -4``,
and ``2**-1`` is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

from woz.lexer import Token, TokenKind, WozError, tokenize
from woz.rational import Rational

__all__ = [
    "Assign",
    "BinaryOp",
    "Call",
    "Name",
    "Node",
    "Number",
    "ParseError",
    "UnaryOp",
    "parse",
]


class ParseError(WozError):
    """Raised when the tokens don't form a valid statement."""


@dataclass(frozen=True)
class Number:
    value: Rational
    position: int = 0


@dataclass(frozen=True)
class Name:
    name: str
    position: int = 0


@dataclass(frozen=True)
class UnaryOp:
    op: str
    operand: Node
    position: int = 0


@dataclass(frozen=True)
class BinaryOp:
    op: str
    left: Node
    right: Node
    position: int = 0


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple[Node, ...]
    position: int = 0


@dataclass(frozen=True)
class Assign:
    name: str
    value: Node
    position: int = 0


Node = Number | Name | UnaryOp | BinaryOp | Call | Assign

# Binding powers (higher binds tighter).
_ASSIGN_BP = 1
_INFIX_BP: dict[TokenKind, int] = {
    TokenKind.PLUS: 10,
    TokenKind.MINUS: 10,
    TokenKind.STAR: 20,
    TokenKind.SLASH: 20,
    TokenKind.POWER: 40,
}
_UNARY_BP = 30
_RIGHT_ASSOC = {TokenKind.POWER}
# Limit on nesting so deeply nested input fails cleanly, not with RecursionError.
_MAX_DEPTH = 200


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._index = 0
        self._depth = 0

    def _peek(self) -> Token:
        return self._tokens[self._index]

    def _advance(self) -> Token:
        token = self._tokens[self._index]
        if token.kind is not TokenKind.EOF:
            self._index += 1
        return token

    def _expect(self, kind: TokenKind) -> Token:
        token = self._peek()
        if token.kind is not kind:
            raise ParseError(f"expected {kind.value!r}, found {_describe(token)}", token.position)
        return self._advance()

    def statement(self) -> Node:
        node = self.expression(0)
        end = self._peek()
        if end.kind is not TokenKind.EOF:
            raise ParseError(f"unexpected {_describe(end)}", end.position)
        return node

    def expression(self, min_bp: int) -> Node:
        self._depth += 1
        if self._depth > _MAX_DEPTH:
            raise ParseError("expression is nested too deeply", self._peek().position)
        try:
            left = self._prefix()
            while True:
                token = self._peek()
                if token.kind is TokenKind.ASSIGN:
                    if min_bp >= _ASSIGN_BP:
                        break
                    if not isinstance(left, Name):
                        raise ParseError("can only assign to a variable name", token.position)
                    self._advance()
                    value = self.expression(_ASSIGN_BP - 1)
                    left = Assign(left.name, value, left.position)
                    continue
                bp = _INFIX_BP.get(token.kind)
                if bp is None or bp <= min_bp:
                    break
                self._advance()
                right_bp = bp - 1 if token.kind in _RIGHT_ASSOC else bp
                right = self.expression(right_bp)
                left = BinaryOp(token.text if token.kind is not TokenKind.POWER else "**",
                                left, right, token.position)
            return left
        finally:
            self._depth -= 1

    def _prefix(self) -> Node:
        token = self._advance()
        kind = token.kind
        if kind is TokenKind.NUMBER:
            try:
                value = Rational.from_str(token.text)
            except ValueError as exc:
                raise ParseError(str(exc), token.position) from None
            return Number(value, token.position)
        if kind is TokenKind.NAME:
            if self._peek().kind is TokenKind.LPAREN:
                return self._call(token)
            return Name(token.text, token.position)
        if kind in (TokenKind.MINUS, TokenKind.PLUS):
            operand = self.expression(_UNARY_BP)
            return UnaryOp(token.text, operand, token.position)
        if kind is TokenKind.LPAREN:
            inner = self.expression(0)
            if isinstance(inner, Assign):
                raise ParseError("assignment is not allowed inside parentheses", inner.position)
            self._expect(TokenKind.RPAREN)
            return inner
        raise ParseError(f"unexpected {_describe(token)}", token.position)

    def _call(self, name: Token) -> Call:
        self._expect(TokenKind.LPAREN)
        args: list[Node] = []
        if self._peek().kind is not TokenKind.RPAREN:
            while True:
                arg = self.expression(0)
                if isinstance(arg, Assign):
                    raise ParseError("assignment is not allowed in arguments", arg.position)
                args.append(arg)
                if self._peek().kind is TokenKind.COMMA:
                    self._advance()
                    continue
                break
        self._expect(TokenKind.RPAREN)
        return Call(name.text, tuple(args), name.position)


def _describe(token: Token) -> str:
    if token.kind is TokenKind.EOF:
        return "end of input"
    return repr(token.text)


def parse(source: str) -> Node:
    """Parse one statement (an expression or an assignment)."""
    tokens = tokenize(source)
    if tokens[0].kind is TokenKind.EOF:
        raise ParseError("empty input", 0)
    return _Parser(tokens).statement()
