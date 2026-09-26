"""Evaluate woz ASTs to exact or correctly-rounded Rationals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from woz import functions
from woz.bigdec import format_sig
from woz.lexer import WozError
from woz.parser import Assign, Binary, Call, Node, Num, Unary, Var, parse
from woz.rational import Rational

#: Refuse powers whose result would exceed roughly this many bits.
MAX_POWER_BITS: int = 1_000_000


class EvalError(WozError):
    """Base class for errors raised while evaluating an expression."""


class UndefinedNameError(EvalError):
    """A variable or function name is not defined."""


class ArityError(EvalError):
    """A function was called with the wrong number of arguments."""


class MathDomainError(EvalError):
    """An argument lies outside a function's domain (e.g. ``ln(-1)``)."""


class DivisionByZeroError(EvalError):
    """Division by zero, or zero raised to a negative power."""


class ReadOnlyNameError(EvalError):
    """An attempt to assign to a built-in constant or function."""


@dataclass(frozen=True)
class Value:
    """An evaluation result.

    ``exact`` is False once a transcendental function (or constant) has
    contributed a rounded value; such results are shown to ``digits``
    significant digits instead of as fractions.
    """

    rational: Rational
    exact: bool = True


UnaryFn = Callable[[Rational, int], Rational]

_UNARY_FUNCTIONS: dict[str, UnaryFn] = {
    "sqrt": functions.sqrt,
    "exp": functions.exp,
    "ln": functions.ln,
    "sin": functions.sin,
    "cos": functions.cos,
    "atan": functions.atan,
}

_CONSTANTS: frozenset[str] = frozenset({"pi", "e"})
_OTHER_BUILTINS: frozenset[str] = frozenset({"abs"})
BUILTIN_NAMES: frozenset[str] = (
    frozenset(_UNARY_FUNCTIONS) | _CONSTANTS | _OTHER_BUILTINS
)


def _power(base: Rational, exponent: Rational) -> Rational:
    if not base and exponent < 0:
        raise DivisionByZeroError("zero cannot be raised to a negative power")
    magnitude = max(abs(base.numerator), base.denominator).bit_length()
    if magnitude > 1 and abs(exponent.numerator) * magnitude > MAX_POWER_BITS:
        raise EvalError("power result is too large")
    try:
        return base**exponent
    except ValueError as exc:
        raise MathDomainError(f"{base} ** {exponent}: {exc}") from exc


class Evaluator:
    """Holds variables and the working precision for a session."""

    def __init__(self, digits: int = 30) -> None:
        self.variables: dict[str, Value] = {}
        self.digits = digits

    @property
    def digits(self) -> int:
        return self._digits

    @digits.setter
    def digits(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise EvalError("digits must be an integer")
        if not functions.MIN_DIGITS <= value <= functions.MAX_DIGITS:
            raise EvalError(
                f"digits must be between {functions.MIN_DIGITS} "
                f"and {functions.MAX_DIGITS}"
            )
        self._digits = value

    def evaluate(self, src: str) -> Value:
        """Parse and evaluate ``src``; assignments also store the value."""
        return self.eval_node(parse(src))

    def format(self, value: Value) -> str:
        """Render a value: exact fractions as-is, approximations to digits."""
        r = value.rational
        if value.exact:
            return str(r)
        return format_sig(r, self.digits, strip_zeros=True)

    def eval_node(self, node: Node) -> Value:
        if isinstance(node, Num):
            return Value(node.value)
        if isinstance(node, Var):
            return self._lookup(node.name)
        if isinstance(node, Unary):
            operand = self.eval_node(node.operand)
            r = -operand.rational if node.op == "-" else operand.rational
            return Value(r, operand.exact)
        if isinstance(node, Binary):
            return self._binary(node)
        if isinstance(node, Call):
            return self._call(node)
        if isinstance(node, Assign):
            if node.name in BUILTIN_NAMES:
                raise ReadOnlyNameError(f"cannot assign to built-in {node.name!r}")
            value = self.eval_node(node.value)
            self.variables[node.name] = value
            return value
        raise EvalError(f"unknown node type {type(node).__name__}")

    def _lookup(self, name: str) -> Value:
        if name == "pi":
            return Value(functions.pi(self.digits), exact=False)
        if name == "e":
            return Value(functions.exp(Rational(1), self.digits), exact=False)
        if name in self.variables:
            return self.variables[name]
        if name in BUILTIN_NAMES:
            raise EvalError(f"{name!r} is a function; call it as {name}(...)")
        raise UndefinedNameError(f"undefined name {name!r}")

    def _binary(self, node: Binary) -> Value:
        left = self.eval_node(node.left)
        right = self.eval_node(node.right)
        a, b = left.rational, right.rational
        exact = left.exact and right.exact
        op = node.op
        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        elif op == "*":
            result = a * b
        elif op == "/":
            if not b:
                raise DivisionByZeroError("division by zero")
            result = a / b
        elif op == "**":
            result = _power(a, b)
        else:
            raise EvalError(f"unknown operator {op!r}")
        return Value(result, exact)

    def _call(self, node: Call) -> Value:
        name = node.name
        if name in self.variables and name not in BUILTIN_NAMES:
            raise EvalError(f"{name!r} is a variable, not a function")
        if name == "abs":
            (arg,) = self._args(node, 1)
            return Value(abs(arg.rational), arg.exact)
        fn = _UNARY_FUNCTIONS.get(name)
        if fn is None:
            if name in _CONSTANTS:
                raise EvalError(f"{name!r} is a constant, not a function")
            raise UndefinedNameError(f"undefined function {name!r}")
        (arg,) = self._args(node, 1)
        try:
            result = fn(arg.rational, self.digits)
        except functions.DomainError as exc:
            raise MathDomainError(f"{name}: {exc}") from exc
        except OverflowError as exc:
            raise EvalError(f"{name}: {exc}") from exc
        return Value(result, exact=False)

    def _args(self, node: Call, count: int) -> list[Value]:
        if len(node.args) != count:
            plural = "" if count == 1 else "s"
            raise ArityError(
                f"{node.name}() takes {count} argument{plural}, got {len(node.args)}"
            )
        return [self.eval_node(a) for a in node.args]
