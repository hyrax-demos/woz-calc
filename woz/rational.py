"""Exact rational numbers, implemented from scratch on Python integers."""

from __future__ import annotations

import math
import re
import sys
from typing import Union

__all__ = ["Rational", "RationalLike"]

RationalLike = Union["Rational", int]

# Largest decimal exponent accepted by ``Rational.from_str``. Keeps inputs
# such as "1e999999999" from building an enormous integer.
_MAX_EXPONENT = 100_000

_FRACTION_RE = re.compile(r"\s*([+-]?\d+)\s*/\s*([+-]?\d+)\s*\Z")
_DECIMAL_RE = re.compile(
    r"\s*(?P<sign>[+-])?(?P<int>\d*)(?:\.(?P<frac>\d*))?(?:[eE](?P<exp>[+-]?\d+))?\s*\Z"
)

_HASH_MODULUS = sys.hash_info.modulus
_HASH_INF = sys.hash_info.inf


class Rational:
    """An exact fraction ``numerator / denominator`` kept in lowest terms.

    The denominator is always positive, and zero is stored as ``0/1``.
    Instances are immutable and hash like the equal ``int`` (and like the
    equal ``fractions.Fraction``), so they can be dict keys.
    """

    __slots__ = ("_num", "_den")

    _num: int
    _den: int

    def __init__(self, numerator: RationalLike = 0, denominator: RationalLike = 1) -> None:
        num, den = _as_pair(numerator)
        dnum, dden = _as_pair(denominator)
        # (num/den) / (dnum/dden) == (num*dden) / (den*dnum)
        num, den = num * dden, den * dnum
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
        """Build from an already-normalised pair without re-reducing it."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "_num", num)
        object.__setattr__(obj, "_den", den)
        return obj

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Rational is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Rational is immutable")

    # ------------------------------------------------------------------ parsing

    @classmethod
    def from_str(cls, text: str) -> Rational:
        """Parse ``'3/4'``, ``'-1.25'``, ``'1e-3'``, ``'.5'`` or ``'7'``."""
        if not isinstance(text, str):
            raise TypeError(f"from_str expects a str, got {type(text).__name__}")
        match = _FRACTION_RE.match(text)
        if match:
            return cls(int(match.group(1)), int(match.group(2)))
        match = _DECIMAL_RE.match(text)
        if not match:
            raise ValueError(f"invalid rational literal: {text!r}")
        int_part = match.group("int") or ""
        frac_part = match.group("frac") or ""
        if not int_part and not frac_part:
            raise ValueError(f"invalid rational literal: {text!r}")
        exponent = int(match.group("exp") or "0") - len(frac_part)
        if abs(exponent) > _MAX_EXPONENT:
            raise ValueError(f"exponent out of range in {text!r}")
        num = int(int_part + frac_part)
        if match.group("sign") == "-":
            num = -num
        if exponent >= 0:
            return cls(num * 10**exponent)
        return cls(num, 10**-exponent)

    # --------------------------------------------------------------- accessors

    @property
    def numerator(self) -> int:
        return self._num

    @property
    def denominator(self) -> int:
        return self._den

    def is_integer(self) -> bool:
        return self._den == 1

    def floor(self) -> int:
        return self._num // self._den

    def ceil(self) -> int:
        return -(-self._num // self._den)

    # ------------------------------------------------------------- arithmetic

    def __add__(self, other: object) -> Rational:
        pair = _other_pair(other)
        if pair is None:
            return NotImplemented
        onum, oden = pair
        return Rational(self._num * oden + onum * self._den, self._den * oden)

    def __radd__(self, other: object) -> Rational:
        return self.__add__(other)

    def __sub__(self, other: object) -> Rational:
        pair = _other_pair(other)
        if pair is None:
            return NotImplemented
        onum, oden = pair
        return Rational(self._num * oden - onum * self._den, self._den * oden)

    def __rsub__(self, other: object) -> Rational:
        pair = _other_pair(other)
        if pair is None:
            return NotImplemented
        onum, oden = pair
        return Rational(onum * self._den - self._num * oden, self._den * oden)

    def __mul__(self, other: object) -> Rational:
        pair = _other_pair(other)
        if pair is None:
            return NotImplemented
        onum, oden = pair
        return Rational(self._num * onum, self._den * oden)

    def __rmul__(self, other: object) -> Rational:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> Rational:
        pair = _other_pair(other)
        if pair is None:
            return NotImplemented
        onum, oden = pair
        if onum == 0:
            raise ZeroDivisionError("division by zero")
        return Rational(self._num * oden, self._den * onum)

    def __rtruediv__(self, other: object) -> Rational:
        pair = _other_pair(other)
        if pair is None:
            return NotImplemented
        onum, oden = pair
        if self._num == 0:
            raise ZeroDivisionError("division by zero")
        return Rational(onum * self._den, oden * self._num)

    def __pow__(self, exponent: object) -> Rational:
        """Raise to an integer power (an int or an integer-valued Rational)."""
        if isinstance(exponent, Rational):
            if exponent._den != 1:
                raise ValueError("Rational powers need an integer exponent")
            n = exponent._num
        elif isinstance(exponent, int):
            n = exponent
        else:
            return NotImplemented
        if n >= 0:
            return Rational._raw(self._num**n, self._den**n)
        if self._num == 0:
            raise ZeroDivisionError("zero cannot be raised to a negative power")
        num, den = self._den ** (-n), self._num ** (-n)
        if den < 0:
            num, den = -num, -den
        return Rational._raw(num, den)

    def __rpow__(self, base: object) -> Rational:
        if isinstance(base, int):
            return Rational(base) ** self
        return NotImplemented

    def __neg__(self) -> Rational:
        return Rational._raw(-self._num, self._den)

    def __pos__(self) -> Rational:
        return self

    def __abs__(self) -> Rational:
        return Rational._raw(abs(self._num), self._den)

    # ------------------------------------------------------------ comparisons

    def _cmp(self, other: object) -> int | None:
        pair = _other_pair(other)
        if pair is None:
            return None
        onum, oden = pair
        lhs, rhs = self._num * oden, onum * self._den
        return (lhs > rhs) - (lhs < rhs)

    def __eq__(self, other: object) -> bool:
        pair = _other_pair(other)
        if pair is None:
            return NotImplemented
        return self._num == pair[0] and self._den == pair[1]

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
        # Same scheme as the numeric tower, so hash(Rational(n)) == hash(n).
        try:
            dinv = pow(self._den, -1, _HASH_MODULUS)
        except ValueError:
            h = _HASH_INF
        else:
            h = hash(hash(abs(self._num)) * dinv)
        result = h if self._num >= 0 else -h
        return -2 if result == -1 else result

    # ------------------------------------------------------------ conversions

    def __bool__(self) -> bool:
        return self._num != 0

    def __int__(self) -> int:
        """Truncate toward zero, like ``int(float)``."""
        if self._num < 0:
            return -(-self._num // self._den)
        return self._num // self._den

    def __float__(self) -> float:
        return self._num / self._den

    def __floor__(self) -> int:
        return self.floor()

    def __ceil__(self) -> int:
        return self.ceil()

    def __repr__(self) -> str:
        return f"Rational({self._num}, {self._den})"

    def __str__(self) -> str:
        if self._den == 1:
            return str(self._num)
        return f"{self._num}/{self._den}"


def _as_pair(value: object) -> tuple[int, int]:
    if isinstance(value, Rational):
        return value._num, value._den
    if isinstance(value, int):
        return int(value), 1
    raise TypeError(f"Rational components must be int or Rational, not {type(value).__name__}")


def _other_pair(value: object) -> tuple[int, int] | None:
    if isinstance(value, Rational):
        return value._num, value._den
    if isinstance(value, int):
        return int(value), 1
    return None
