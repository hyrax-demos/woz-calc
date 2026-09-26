"""An exact rational number type, written without ``fractions`` or ``decimal``."""

from __future__ import annotations

import math
import re
import sys
from typing import Union

RationalLike = Union["Rational", int]

_NUMBER_RE = re.compile(
    r"""
    \s*
    (?P<num>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
    (?:\s*/\s*(?P<den>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?))?
    \s*
    \Z
    """,
    re.VERBOSE,
)

_DECIMAL_RE = re.compile(
    r"(?P<sign>[+-]?)(?P<int>\d*)(?:\.(?P<frac>\d*))?(?:[eE](?P<exp>[+-]?\d+))?"
)

_HASH_MODULUS = sys.hash_info.modulus
_HASH_INF = sys.hash_info.inf


def integer_root_floor(n: int, k: int) -> int:
    """Return ``floor(n ** (1/k))`` for ``n >= 0`` and ``k >= 1``."""
    if n < 0 or k < 1:
        raise ValueError("integer_root_floor() needs n >= 0 and k >= 1")
    if n < 2 or k == 1:
        return n
    # Newton iteration on integers, starting from an over-estimate.
    x = 1 << -(-n.bit_length() // k)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            return x
        x = y


def _integer_root(n: int, k: int) -> int | None:
    """Return the exact non-negative integer ``k``-th root of ``n >= 0``, or None."""
    x = integer_root_floor(n, k)
    return x if x**k == n else None


class Rational:
    """An immutable, always-normalised fraction ``numerator / denominator``.

    The denominator is always positive and ``gcd(numerator, denominator) == 1``.
    Arithmetic with ``int`` operands is supported on either side; floats are
    deliberately rejected so that results stay exact.
    """

    __slots__ = ("_num", "_den")

    _num: int
    _den: int

    def __init__(self, numerator: RationalLike = 0, denominator: RationalLike = 1) -> None:
        if isinstance(numerator, Rational) or isinstance(denominator, Rational):
            n = Rational._coerce(numerator)
            d = Rational._coerce(denominator)
            if n is None or d is None:
                raise TypeError("Rational() arguments must be int or Rational")
            num, den = n._num * d._den, n._den * d._num
        elif isinstance(numerator, int) and isinstance(denominator, int):
            num, den = int(numerator), int(denominator)
        else:
            raise TypeError("Rational() arguments must be int or Rational")
        if den == 0:
            raise ZeroDivisionError("Rational denominator must not be zero")
        if den < 0:
            num, den = -num, -den
        g = math.gcd(num, den)
        if g > 1:
            num //= g
            den //= g
        object.__setattr__(self, "_num", num)
        object.__setattr__(self, "_den", den)

    @classmethod
    def _raw(cls, num: int, den: int) -> Rational:
        """Build from an already-normalised pair without re-checking."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "_num", num)
        object.__setattr__(obj, "_den", den)
        return obj

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Rational is immutable")

    @staticmethod
    def _coerce(value: object) -> Rational | None:
        if isinstance(value, Rational):
            return value
        if isinstance(value, int):
            return Rational._raw(int(value), 1)
        return None

    # ------------------------------------------------------------------ parsing
    @classmethod
    def from_str(cls, text: str) -> Rational:
        """Parse ``'3/4'``, ``'-1.25'``, ``'1e-3'``, ``'2.5/1e2'`` and friends."""
        if not isinstance(text, str):
            raise TypeError("from_str() expects a string")
        match = _NUMBER_RE.match(text)
        if match is None:
            raise ValueError(f"invalid rational literal: {text!r}")
        value = cls._parse_decimal(match.group("num"))
        den_text = match.group("den")
        if den_text is not None:
            den = cls._parse_decimal(den_text)
            if not den:
                raise ZeroDivisionError(f"zero denominator in {text!r}")
            value = value / den
        return value

    @classmethod
    def _parse_decimal(cls, text: str) -> Rational:
        match = _DECIMAL_RE.fullmatch(text)
        if match is None:  # pragma: no cover - guarded by _NUMBER_RE
            raise ValueError(f"invalid decimal literal: {text!r}")
        int_part = match.group("int") or ""
        frac_part = match.group("frac") or ""
        exp = int(match.group("exp") or 0) - len(frac_part)
        digits = int(int_part + frac_part or "0")
        if match.group("sign") == "-":
            digits = -digits
        if exp >= 0:
            return cls(digits * 10**exp)
        return cls(digits, 10**-exp)

    # --------------------------------------------------------------- accessors
    @property
    def numerator(self) -> int:
        return self._num

    @property
    def denominator(self) -> int:
        return self._den

    def is_integer(self) -> bool:
        return self._den == 1

    # -------------------------------------------------------------- arithmetic
    def __add__(self, other: object) -> Rational:
        o = Rational._coerce(other)
        if o is None:
            return NotImplemented
        return Rational(self._num * o._den + o._num * self._den, self._den * o._den)

    def __radd__(self, other: object) -> Rational:
        return self.__add__(other)

    def __sub__(self, other: object) -> Rational:
        o = Rational._coerce(other)
        if o is None:
            return NotImplemented
        return Rational(self._num * o._den - o._num * self._den, self._den * o._den)

    def __rsub__(self, other: object) -> Rational:
        o = Rational._coerce(other)
        if o is None:
            return NotImplemented
        return o - self

    def __mul__(self, other: object) -> Rational:
        o = Rational._coerce(other)
        if o is None:
            return NotImplemented
        return Rational(self._num * o._num, self._den * o._den)

    def __rmul__(self, other: object) -> Rational:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> Rational:
        o = Rational._coerce(other)
        if o is None:
            return NotImplemented
        if o._num == 0:
            raise ZeroDivisionError("division by zero")
        return Rational(self._num * o._den, self._den * o._num)

    def __rtruediv__(self, other: object) -> Rational:
        o = Rational._coerce(other)
        if o is None:
            return NotImplemented
        return o / self

    def __pow__(self, exponent: object) -> Rational:
        """Raise to an integer power, or to a rational power when exact.

        ``Rational(4, 9) ** Rational(1, 2) == Rational(2, 3)``; a rational
        exponent whose result is irrational raises ``ValueError``.
        """
        e = Rational._coerce(exponent)
        if e is None:
            return NotImplemented
        if e._den != 1:
            return self._root(e._den) ** e._num
        n = e._num
        if n >= 0:
            return Rational._raw(self._num**n, self._den**n)
        if self._num == 0:
            raise ZeroDivisionError("zero cannot be raised to a negative power")
        num, den = self._den ** -n, self._num ** -n
        if den < 0:
            num, den = -num, -den
        return Rational._raw(num, den)

    def __rpow__(self, base: object) -> Rational:
        b = Rational._coerce(base)
        if b is None:
            return NotImplemented
        return b**self

    def _root(self, k: int) -> Rational:
        """Exact ``k``-th root (``k >= 2``) or ``ValueError`` if irrational."""
        num = self._num
        negative = num < 0
        if negative and k % 2 == 0:
            raise ValueError("even root of a negative number is not real")
        top = _integer_root(abs(num), k)
        bottom = _integer_root(self._den, k)
        if top is None or bottom is None:
            raise ValueError(f"{self} has no exact rational root of degree {k}")
        return Rational._raw(-top if negative else top, bottom)

    def __neg__(self) -> Rational:
        return Rational._raw(-self._num, self._den)

    def __pos__(self) -> Rational:
        return self

    def __abs__(self) -> Rational:
        return Rational._raw(abs(self._num), self._den)

    # ------------------------------------------------------------- conversions
    def __bool__(self) -> bool:
        return self._num != 0

    def __int__(self) -> int:
        return self.__trunc__()

    def __trunc__(self) -> int:
        if self._num < 0:
            return -(-self._num // self._den)
        return self._num // self._den

    def __floor__(self) -> int:
        return self._num // self._den

    def __ceil__(self) -> int:
        return -(-self._num // self._den)

    def __float__(self) -> float:
        return self._num / self._den

    # ------------------------------------------------------------- comparisons
    def _cmp(self, other: object) -> int | None:
        o = Rational._coerce(other)
        if o is None:
            return None
        lhs = self._num * o._den
        rhs = o._num * self._den
        return (lhs > rhs) - (lhs < rhs)

    def __eq__(self, other: object) -> bool:
        o = Rational._coerce(other)
        if o is None:
            return NotImplemented
        return self._num == o._num and self._den == o._den

    def __lt__(self, other: object) -> bool:
        c = self._cmp(other)
        return NotImplemented if c is None else c < 0

    def __le__(self, other: object) -> bool:
        c = self._cmp(other)
        return NotImplemented if c is None else c <= 0

    def __gt__(self, other: object) -> bool:
        c = self._cmp(other)
        return NotImplemented if c is None else c > 0

    def __ge__(self, other: object) -> bool:
        c = self._cmp(other)
        return NotImplemented if c is None else c >= 0

    def __hash__(self) -> int:
        # Same scheme as the built-in numeric tower, so that
        # hash(Rational(n)) == hash(n) and equal values hash equally.
        try:
            dinv = pow(self._den, -1, _HASH_MODULUS)
        except ValueError:
            h = _HASH_INF
        else:
            h = hash(hash(abs(self._num)) * dinv)
        result = h if self._num >= 0 else -h
        return -2 if result == -1 else result

    # ----------------------------------------------------------------- display
    def __repr__(self) -> str:
        return f"Rational({self._num}, {self._den})"

    def __str__(self) -> str:
        if self._den == 1:
            return str(self._num)
        return f"{self._num}/{self._den}"

    def __reduce__(self) -> tuple[type[Rational], tuple[int, int]]:
        return (Rational, (self._num, self._den))


def as_rational(value: RationalLike | str) -> Rational:
    """Coerce an ``int``, ``Rational`` or numeric string to :class:`Rational`."""
    if isinstance(value, str):
        return Rational.from_str(value)
    coerced = Rational._coerce(value)
    if coerced is None:
        raise TypeError(f"cannot convert {type(value).__name__} to Rational")
    return coerced
