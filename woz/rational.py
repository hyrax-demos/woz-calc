"""Exact rational numbers, written from scratch (no ``fractions``/``decimal``).

A :class:`Rational` is always stored in lowest terms with a positive
denominator, so structural equality and hashing are trivial.  Hashing
follows CPython's numeric-hash rules, which means ``hash(Rational(n)) ==
hash(n)`` and equal values compare and hash equal across ``int``.
"""

from __future__ import annotations

import math
import re
import sys
from typing import Union

RationalLike = Union["Rational", int]

_HASH_MODULUS: int = sys.hash_info.modulus
_HASH_INF: int = sys.hash_info.inf

_FRACTION_RE = re.compile(r"\s*([+-]?)(\d+)\s*/\s*(\d+)\s*\Z")
_DECIMAL_RE = re.compile(
    r"\s*([+-]?)(?:(\d+)(?:\.(\d*))?|\.(\d+))(?:[eE]([+-]?\d+))?\s*\Z"
)


def integer_root(n: int, k: int) -> int:
    """Return ``floor(n ** (1/k))`` for ``n >= 0`` and ``k >= 1``, exactly."""
    if n < 0:
        raise ValueError("integer_root requires n >= 0")
    if k < 1:
        raise ValueError("integer_root requires k >= 1")
    if n < 2 or k == 1:
        return n
    if k == 2:
        return math.isqrt(n)
    # Start from a power of two that is guaranteed to be >= the root and
    # run Newton's method downwards; it converges monotonically.
    x = 1 << -(-n.bit_length() // k)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            return x
        x = y


class Rational:
    """An exact fraction ``numerator / denominator`` in lowest terms."""

    __slots__ = ("_den", "_num")

    _num: int
    _den: int

    def __init__(
        self,
        numerator: RationalLike | float | str = 0,
        denominator: RationalLike = 1,
    ) -> None:
        if isinstance(numerator, str):
            if denominator != 1:
                raise TypeError("a string numerator takes no denominator")
            parsed = Rational.from_str(numerator)
            self._num, self._den = parsed._num, parsed._den
            return
        if isinstance(numerator, float):
            if not math.isfinite(numerator):
                raise ValueError(f"cannot convert {numerator!r} to Rational")
            numerator = Rational(*numerator.as_integer_ratio())
        num_n, num_d = _as_pair(numerator)
        den_n, den_d = _as_pair(denominator)
        n = num_n * den_d
        d = num_d * den_n
        if d == 0:
            raise ZeroDivisionError("Rational with zero denominator")
        if d < 0:
            n, d = -n, -d
        g = math.gcd(n, d)
        self._num = n // g
        self._den = d // g

    @classmethod
    def _raw(cls, n: int, d: int) -> Rational:
        """Build from an already-normalised pair (internal fast path)."""
        obj = object.__new__(cls)
        obj._num = n
        obj._den = d
        return obj

    @classmethod
    def from_str(cls, text: str) -> Rational:
        """Parse ``'3/4'``, ``'-1.25'``, ``'1e-3'``, ``'.5E+2'`` or ``'42'``."""
        if not isinstance(text, str):
            raise TypeError("from_str expects a str")
        m = _FRACTION_RE.match(text)
        if m:
            sign, num, den = m.groups()
            if int(den) == 0:
                raise ZeroDivisionError(f"zero denominator in {text!r}")
            value = cls(int(num), int(den))
            return -value if sign == "-" else value
        m = _DECIMAL_RE.match(text)
        if not m:
            raise ValueError(f"invalid rational literal: {text!r}")
        sign, int_part, frac_part, lone_frac, exp_part = m.groups()
        if lone_frac is not None:
            int_part, frac_part = "0", lone_frac
        frac_part = frac_part or ""
        digits = int((int_part or "0") + frac_part)
        exponent = (int(exp_part) if exp_part else 0) - len(frac_part)
        if exponent >= 0:
            value = cls(digits * 10**exponent)
        else:
            value = cls(digits, 10**-exponent)
        return -value if sign == "-" else value

    # -- accessors -------------------------------------------------------
    @property
    def numerator(self) -> int:
        return self._num

    @property
    def denominator(self) -> int:
        return self._den

    def is_integer(self) -> bool:
        return self._den == 1

    # -- conversions -----------------------------------------------------
    def __repr__(self) -> str:
        if self._den == 1:
            return f"Rational({self._num})"
        return f"Rational({self._num}, {self._den})"

    def __str__(self) -> str:
        if self._den == 1:
            return str(self._num)
        return f"{self._num}/{self._den}"

    def __float__(self) -> float:
        return self._num / self._den

    def __int__(self) -> int:
        return self.__trunc__()

    def __trunc__(self) -> int:
        q = abs(self._num) // self._den
        return q if self._num >= 0 else -q

    def __floor__(self) -> int:
        return self._num // self._den

    def __ceil__(self) -> int:
        return -(-self._num // self._den)

    def __bool__(self) -> bool:
        return self._num != 0

    # -- arithmetic ------------------------------------------------------
    def __neg__(self) -> Rational:
        return Rational._raw(-self._num, self._den)

    def __pos__(self) -> Rational:
        return self

    def __abs__(self) -> Rational:
        return Rational._raw(abs(self._num), self._den)

    def __add__(self, other: object) -> Rational:
        pair = _coerce(other)
        if pair is None:
            return NotImplemented
        n, d = pair
        return Rational(self._num * d + n * self._den, self._den * d)

    def __radd__(self, other: object) -> Rational:
        return self.__add__(other)

    def __sub__(self, other: object) -> Rational:
        pair = _coerce(other)
        if pair is None:
            return NotImplemented
        n, d = pair
        return Rational(self._num * d - n * self._den, self._den * d)

    def __rsub__(self, other: object) -> Rational:
        pair = _coerce(other)
        if pair is None:
            return NotImplemented
        n, d = pair
        return Rational(n * self._den - self._num * d, d * self._den)

    def __mul__(self, other: object) -> Rational:
        pair = _coerce(other)
        if pair is None:
            return NotImplemented
        n, d = pair
        return Rational(self._num * n, self._den * d)

    def __rmul__(self, other: object) -> Rational:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> Rational:
        pair = _coerce(other)
        if pair is None:
            return NotImplemented
        n, d = pair
        if n == 0:
            raise ZeroDivisionError("division by zero")
        return Rational(self._num * d, self._den * n)

    def __rtruediv__(self, other: object) -> Rational:
        pair = _coerce(other)
        if pair is None:
            return NotImplemented
        n, d = pair
        if self._num == 0:
            raise ZeroDivisionError("division by zero")
        return Rational(n * self._den, d * self._num)

    def __pow__(self, other: object) -> Rational:
        pair = _coerce(other)
        if pair is None:
            return NotImplemented
        p, q = pair
        if q == 1:
            return self._int_pow(p)
        return self._root(q)._int_pow(p)

    def __rpow__(self, other: object) -> Rational:
        pair = _coerce(other)
        if pair is None:
            return NotImplemented
        return Rational._raw(*pair) ** self

    def _int_pow(self, e: int) -> Rational:
        if e >= 0:
            return Rational._raw(self._num**e, self._den**e)
        if self._num == 0:
            raise ZeroDivisionError("zero cannot be raised to a negative power")
        n, d = self._den**-e, self._num**-e
        if d < 0:
            n, d = -n, -d
        return Rational._raw(n, d)

    def _root(self, k: int) -> Rational:
        """Exact ``k``-th root, or ``ValueError`` if it is not rational."""
        negative = self._num < 0
        if negative and k % 2 == 0:
            raise ValueError("even root of a negative number is not real")
        n = abs(self._num)
        rn = integer_root(n, k)
        rd = integer_root(self._den, k)
        if rn**k != n or rd**k != self._den:
            raise ValueError("result is irrational")
        return Rational._raw(-rn if negative else rn, rd)

    # -- comparison ------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        pair = _coerce(other)
        if pair is None:
            return NotImplemented
        return self._num == pair[0] and self._den == pair[1]

    def _cmp(self, other: object) -> int | None:
        pair = _coerce(other)
        if pair is None:
            return None
        lhs = self._num * pair[1]
        rhs = pair[0] * self._den
        return (lhs > rhs) - (lhs < rhs)

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
        try:
            dinv = pow(self._den, -1, _HASH_MODULUS)
        except ValueError:
            h = _HASH_INF
        else:
            h = hash(hash(abs(self._num)) * dinv)
        result = h if self._num >= 0 else -h
        return -2 if result == -1 else result


def _as_pair(value: object) -> tuple[int, int]:
    pair = _coerce(value)
    if pair is None:
        raise TypeError(f"cannot convert {type(value).__name__} to Rational")
    return pair


def _coerce(value: object) -> tuple[int, int] | None:
    if isinstance(value, Rational):
        return value._num, value._den
    if isinstance(value, int):
        return int(value), 1
    return None
