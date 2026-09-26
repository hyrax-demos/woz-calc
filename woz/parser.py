"""A Pratt parser for the woz expression language.

Grammar (loosest to tightest binding)::

    name = expr            assignment, right-associative
    a + b, a - b           left-associative
    a * b, a / b           left-associative
    -a, +a                 unary prefix
    a ** b  (or a ^ b)     right-associative; binds tighter than unary
                           minus on its left, so -2**2 == -(2**2)
    f(a, b), (expr), 42, x
"""

from __future__ import annotations

from dataclasses import dataclass

from woz.lexer import Token, TokenKind, WozError, tokenize
from woz.rational import Rational


class ParseError(WozError):
    """Raised on a syntactically invalid expression."""

    def __init__(self, message: str, pos: int) -> None:
        super().__init__(f"{message} at position {pos}")
        self.pos = pos


@dataclass(frozen=True)
class Num:
    value: Rational


@dataclass(frozen=True)
class Var:
    name: str


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


Node = Num | Var | Unary | Binary | Call | Assign

_ASSIGN_BP = 1
_ADD_BP = 10
_MUL_BP = 20
_UNARY_BP = 30
_POW_BP = 40
_CALL_BP = 50

_INFIX_BP: dict[TokenKind, int] = {
    TokenKind.ASSIGN: _ASSIGN_BP,
    TokenKind.PLUS: _ADD_BP,
    TokenKind.MINUS: _ADD_BP,
    TokenKind.STAR: _MUL_BP,
    TokenKind.SLASH: _MUL_BP,
    TokenKind.POW: _POW_BP,
    TokenKind.LPAREN: _CALL_BP,
}

_OP_TEXT: dict[TokenKind, str] = {
    TokenKind.PLUS: "+",
    TokenKind.MINUS: "-",
    TokenKind.STAR: "*",
    TokenKind.SLASH: "/",
    TokenKind.POW: "**",
}


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._i = 0

    def _peek(self) -> Token:
        return self._tokens[self._i]

    def _next(self) -> Token:
        tok = self._tokens[self._i]
        if tok.kind is not TokenKind.EOF:
            self._i += 1
        return tok

    def _expect(self, kind: TokenKind) -> Token:
        tok = self._next()
        if tok.kind is not kind:
            found = tok.text or "end of input"
            raise ParseError(f"expected {kind.value!r}, found {found!r}", tok.pos)
        return tok

    def parse(self) -> Node:
        if self._peek().kind is TokenKind.EOF:
            raise ParseError("empty expression", self._peek().pos)
        node = self.expression(0)
        tok = self._peek()
        if tok.kind is not TokenKind.EOF:
            raise ParseError(f"unexpected {tok.text!r}", tok.pos)
        return node

    def expression(self, min_bp: int) -> Node:
        left = self._prefix(self._next())
        while True:
            tok = self._peek()
            bp = _INFIX_BP.get(tok.kind)
            if bp is None or bp <= min_bp:
                return left
            self._next()
            left = self._infix(tok, left, bp)

    def _prefix(self, tok: Token) -> Node:
        kind = tok.kind
        if kind is TokenKind.NUMBER:
            return Num(Rational.from_str(tok.text))
        if kind is TokenKind.NAME:
            return Var(tok.text)
        if kind in (TokenKind.MINUS, TokenKind.PLUS):
            return Unary(tok.text, self.expression(_UNARY_BP))
        if kind is TokenKind.LPAREN:
            inner = self.expression(0)
            self._expect(TokenKind.RPAREN)
            return inner
        found = tok.text or "end of input"
        raise ParseError(f"unexpected {found!r}", tok.pos)

    def _infix(self, tok: Token, left: Node, bp: int) -> Node:
        kind = tok.kind
        if kind is TokenKind.ASSIGN:
            if not isinstance(left, Var):
                raise ParseError("can only assign to a variable name", tok.pos)
            return Assign(left.name, self.expression(bp - 1))
        if kind is TokenKind.LPAREN:
            if not isinstance(left, Var):
                raise ParseError("only named functions can be called", tok.pos)
            return Call(left.name, self._arguments())
        if kind is TokenKind.POW:
            # Right-associative: the right operand may absorb another **.
            return Binary("**", left, self.expression(bp - 1))
        return Binary(_OP_TEXT[kind], left, self.expression(bp))

    def _arguments(self) -> tuple[Node, ...]:
        args: list[Node] = []
        if self._peek().kind is TokenKind.RPAREN:
            self._next()
            return ()
        while True:
            args.append(self.expression(_ASSIGN_BP))
            tok = self._next()
            if tok.kind is TokenKind.RPAREN:
                return tuple(args)
            if tok.kind is not TokenKind.COMMA:
                found = tok.text or "end of input"
                raise ParseError(f"expected ',' or ')', found {found!r}", tok.pos)


def parse(src: str) -> Node:
    """Tokenize and parse ``src`` into an AST."""
    return Parser(tokenize(src)).parse()
