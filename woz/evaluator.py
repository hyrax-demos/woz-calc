"""Evaluate parsed woz statements to exact Rationals."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from woz import functions
from woz.lexer import WozError
from woz.parser import Assign, BinaryOp, Call, Name, Node, Number, UnaryOp, parse
from woz.rational import Rational

__all__ = [
    "ArgumentError",
    "DivisionByZeroError",
    "DomainError",
    "EvalError",
    "Evaluator",
    "UndefinedNameError",
]


class EvalError(WozError):
    """Base class for errors raised while evaluating a statement."""


class UndefinedNameError(EvalError):
    """A variable or function that doesn't exist."""


class ArgumentError(EvalError):
    """A function was called with the wrong number of arguments."""


class DivisionByZeroError(EvalError):
    """Division by zero, or zero raised to a negative power."""


class DomainError(EvalError):
    """An argument outside a function's domain, or a result too large to hold."""


DEFAULT_PRECISION = 30
# Largest result an integer power may build, in bits of numerator plus
# denominator; stops inputs like 9**9**9 from hanging the process.
_MAX_POWER_BITS = 1_000_000

_UnaryFunction = Callable[[Rational, int], Rational]

_FUNCTIONS: Mapping[str, _UnaryFunction] = {
    "sqrt": functions.sqrt,
    "exp": functions.exp,
    "ln": functions.ln,
    "sin": functions.sin,
    "cos": functions.cos,
    "atan": functions.atan,
}

_CONSTANTS: Mapping[str, Callable[[int], Rational]] = {
    "pi": functions.pi,
    "e": functions.e,
}


class Evaluator:
    """Evaluates statements, keeping variables between calls.

    Every value is an exact Rational. Transcendental functions and the
    constants ``pi`` and ``e`` are rounded to ``precision`` significant
    digits; everything else stays exact.
    """

    def __init__(self, precision: int = DEFAULT_PRECISION) -> None:
        self._precision = DEFAULT_PRECISION
        self.precision = precision
        self.variables: dict[str, Rational] = {}

    @property
    def precision(self) -> int:
        return self._precision

    @precision.setter
    def precision(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("precision must be an int")
        if not functions.MIN_PRECISION <= value <= functions.MAX_PRECISION:
            raise ValueError(
                f"precision must be between {functions.MIN_PRECISION} "
                f"and {functions.MAX_PRECISION}"
            )
        self._precision = value

    def evaluate(self, source: str) -> Rational:
        """Parse and evaluate one statement; assignments return the value."""
        return self.eval_node(parse(source))

    def eval_node(self, node: Node) -> Rational:
        if isinstance(node, Number):
            return node.value
        if isinstance(node, Name):
            return self._lookup(node)
        if isinstance(node, UnaryOp):
            value = self.eval_node(node.operand)
            return -value if node.op == "-" else value
        if isinstance(node, BinaryOp):
            return self._binary(node)
        if isinstance(node, Call):
            return self._call(node)
        if isinstance(node, Assign):
            if node.name in _FUNCTIONS or node.name in _CONSTANTS:
                raise EvalError(f"cannot assign to built-in name {node.name!r}", node.position)
            value = self.eval_node(node.value)
            self.variables[node.name] = value
            return value
        raise EvalError(f"unknown node type {type(node).__name__}")

    def _lookup(self, node: Name) -> Rational:
        if node.name in self.variables:
            return self.variables[node.name]
        if node.name in _CONSTANTS:
            return _CONSTANTS[node.name](self._precision)
        if node.name in _FUNCTIONS:
            raise EvalError(f"{node.name!r} is a function; call it like {node.name}(x)",
                            node.position)
        raise UndefinedNameError(f"undefined variable {node.name!r}", node.position)

    def _binary(self, node: BinaryOp) -> Rational:
        left = self.eval_node(node.left)
        right = self.eval_node(node.right)
        op = node.op
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            if not right:
                raise DivisionByZeroError("division by zero", node.position)
            return left / right
        if op == "**":
            return self._power(left, right, node.position)
        raise EvalError(f"unknown operator {op!r}", node.position)

    def _power(self, base: Rational, exponent: Rational, position: int) -> Rational:
        if not exponent.is_integer():
            raise DomainError(
                "exponent must be an integer (use sqrt, or exp(b*ln(a)))", position
            )
        n = exponent.numerator
        if base == 0 and n < 0:
            raise DivisionByZeroError("zero cannot be raised to a negative power", position)
        size = max(base.numerator.bit_length(), base.denominator.bit_length())
        if abs(base) != 1 and base != 0 and abs(n) * size > _MAX_POWER_BITS:
            raise DomainError("result of ** is too large", position)
        return base**n

    def _call(self, node: Call) -> Rational:
        if node.name in _CONSTANTS and node.name not in self.variables:
            raise EvalError(f"{node.name!r} is a constant, not a function", node.position)
        function = _FUNCTIONS.get(node.name)
        if function is None:
            raise UndefinedNameError(f"unknown function {node.name!r}", node.position)
        if len(node.args) != 1:
            raise ArgumentError(
                f"{node.name}() takes exactly 1 argument ({len(node.args)} given)",
                node.position,
            )
        argument = self.eval_node(node.args[0])
        try:
            return function(argument, self._precision)
        except (ValueError, OverflowError, ArithmeticError) as exc:
            raise DomainError(f"{node.name}: {exc}", node.position) from None
