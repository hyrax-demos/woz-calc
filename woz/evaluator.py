"""Evaluate woz ASTs to exact :class:`~woz.rational.Rational` values."""

from __future__ import annotations

from collections.abc import Callable

from woz import functions
from woz.lexer import WozError
from woz.parser import Assign, Binary, Call, Name, Node, Number, Unary, parse
from woz.rational import Rational

#: Largest ``|b|`` accepted for an integer power ``a ** b``.
MAX_EXPONENT = 10_000
#: Largest root degree ``q`` accepted for a fractional power ``a ** (p/q)``.
MAX_ROOT_DEGREE = 64


class EvalError(WozError):
    """Base class of runtime evaluation errors."""


class UndefinedNameError(EvalError):
    """Raised when a variable or function name is not defined."""


class ArgumentError(EvalError):
    """Raised when a function is called with the wrong number of arguments."""


class DomainError(EvalError):
    """Raised for mathematically undefined operations (1/0, ln(-1), ...)."""


class ReadOnlyNameError(EvalError):
    """Raised when assigning to a built-in constant or function name."""


class Evaluator:
    """Evaluates expressions, keeping variables between calls.

    Irrational results (functions, ``pi``, ``e``) are correctly rounded to
    ``precision`` significant digits; everything else is exact.
    """

    def __init__(self, precision: int = 30) -> None:
        self.precision = precision
        self.variables: dict[str, Rational] = {}

    @property
    def precision(self) -> int:
        return self._precision

    @precision.setter
    def precision(self, digits: int) -> None:
        if isinstance(digits, bool) or not isinstance(digits, int):
            raise TypeError("precision must be an int")
        if not functions.MIN_DIGITS <= digits <= functions.MAX_DIGITS:
            raise ValueError(
                f"precision must be between {functions.MIN_DIGITS} and {functions.MAX_DIGITS}"
            )
        self._precision = digits

    def evaluate(self, source: str) -> Rational:
        """Parse and evaluate one expression or assignment."""
        return self.eval_node(parse(source))

    def eval_node(self, node: Node) -> Rational:
        try:
            return self._eval(node)
        except ZeroDivisionError as exc:
            raise DomainError(str(exc) or "division by zero") from exc
        except OverflowError as exc:
            raise DomainError(str(exc)) from exc

    def _eval(self, node: Node) -> Rational:
        if isinstance(node, Number):
            return node.value
        if isinstance(node, Name):
            return self._lookup(node.id)
        if isinstance(node, Unary):
            value = self._eval(node.operand)
            return -value if node.op == "-" else value
        if isinstance(node, Binary):
            return self._binary(node.op, self._eval(node.left), self._eval(node.right))
        if isinstance(node, Call):
            return self._call(node)
        if isinstance(node, Assign):
            if node.name in functions.CONSTANTS or node.name in functions.FUNCTIONS:
                raise ReadOnlyNameError(f"cannot assign to built-in name {node.name!r}")
            value = self._eval(node.value)
            self.variables[node.name] = value
            return value
        raise EvalError(f"unknown node {node!r}")  # pragma: no cover

    def _lookup(self, name: str) -> Rational:
        if name in self.variables:
            return self.variables[name]
        if name in functions.CONSTANTS:
            return functions.CONSTANTS[name](self.precision)
        if name in functions.FUNCTIONS:
            raise EvalError(f"function {name!r} must be called with arguments")
        raise UndefinedNameError(f"undefined name {name!r}")

    def _binary(self, op: str, left: Rational, right: Rational) -> Rational:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            if not right:
                raise DomainError("division by zero")
            return left / right
        if op == "**":
            return self._power(left, right)
        raise EvalError(f"unknown operator {op!r}")  # pragma: no cover

    def _power(self, base: Rational, exponent: Rational) -> Rational:
        if abs(exponent.numerator) > MAX_EXPONENT or exponent.denominator > MAX_ROOT_DEGREE:
            raise DomainError("exponent too large")
        if not base and exponent < 0:
            raise DomainError("zero cannot be raised to a negative power")
        try:
            return base**exponent
        except ValueError as exc:
            if base < 0 and exponent.denominator % 2 == 0:
                raise DomainError(str(exc)) from exc
        # Irrational result: a**(p/q) == root(a**p, q) (real root; q odd if a < 0).
        return functions.root(base**exponent.numerator, exponent.denominator, self.precision)

    def _call(self, node: Call) -> Rational:
        fn: Callable[[Rational, int], Rational] | None = functions.FUNCTIONS.get(node.name)
        if fn is None:
            if node.name in self.variables or node.name in functions.CONSTANTS:
                raise EvalError(f"{node.name!r} is not a function")
            raise UndefinedNameError(f"undefined function {node.name!r}")
        if len(node.args) != 1:
            raise ArgumentError(f"{node.name}() takes exactly 1 argument ({len(node.args)} given)")
        arg = self._eval(node.args[0])
        try:
            return fn(arg, self.precision)
        except ValueError as exc:
            raise DomainError(str(exc)) from exc


def evaluate(source: str, precision: int = 30) -> Rational:
    """Evaluate a single expression in a fresh environment."""
    return Evaluator(precision).evaluate(source)
