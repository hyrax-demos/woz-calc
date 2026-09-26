"""Decimal formatting of :class:`~woz.rational.Rational` values.

Every routine here is exact: a value is rounded to ``N`` significant
digits using round-half-even on the true rational value, never on a
binary float approximation.
"""

from __future__ import annotations

from woz.rational import Rational


def _check_digits(digits: int) -> None:
    if isinstance(digits, bool) or not isinstance(digits, int):
        raise TypeError("digits must be an int")
    if digits < 1:
        raise ValueError("digits must be >= 1")


def exponent10(x: Rational) -> int:
    """Return ``floor(log10(|x|))`` for a non-zero ``x``, exactly."""
    if not x:
        raise ValueError("exponent10 of zero is undefined")
    n, d = abs(x.numerator), x.denominator
    # log10(2) ~ 0.30103; the bit-length estimate is within a digit or two.
    e = (n.bit_length() - d.bit_length()) * 30103 // 100000

    def at_least(k: int) -> bool:  # |x| >= 10**k, in exact integers
        return n >= d * 10**k if k >= 0 else n * 10**-k >= d

    while not at_least(e):
        e -= 1
    while at_least(e + 1):
        e += 1
    return e


def _pow10(e: int) -> Rational:
    return Rational(10**e) if e >= 0 else Rational(1, 10**-e)


def decompose(x: Rational, digits: int) -> tuple[bool, int, int]:
    """Round ``x`` to ``digits`` significant digits (round-half-even).

    Returns ``(negative, coefficient, exponent)`` where ``coefficient`` has
    exactly ``digits`` decimal digits (or is ``0``) and ``exponent`` is the
    power of ten of the leading digit, so the rounded value equals
    ``(-1)**negative * coefficient * 10**(exponent - digits + 1)``.
    """
    _check_digits(digits)
    if not x:
        return False, 0, 0
    negative = x < 0
    e = exponent10(x)
    scaled = abs(x) / _pow10(e - digits + 1)
    q, r = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * r
    if twice > scaled.denominator or (twice == scaled.denominator and q % 2 == 1):
        q += 1
    if q == 10**digits:
        q //= 10
        e += 1
    return negative, q, e


def round_sig(x: Rational, digits: int) -> Rational:
    """Return ``x`` rounded to ``digits`` significant digits, as a Rational."""
    negative, coef, e = decompose(x, digits)
    value = Rational(coef) * _pow10(e - digits + 1)
    return -value if negative else value


def format_sig(
    x: Rational,
    digits: int,
    *,
    scientific: bool | None = None,
    strip_zeros: bool = False,
) -> str:
    """Format ``x`` to ``digits`` significant digits.

    With ``scientific=None`` the ``%g`` rule picks the notation: scientific
    when the decimal exponent is below ``-4`` or at least ``digits``.
    Trailing zeros are kept (they are significant) unless ``strip_zeros``.
    """
    negative, coef, e = decompose(x, digits)
    body = str(coef).rjust(digits, "0") if coef else "0" * digits
    if scientific is None:
        scientific = coef != 0 and (e < -4 or e >= digits)
    if scientific:
        mantissa = body[0]
        frac = body[1:]
        if strip_zeros:
            frac = frac.rstrip("0")
        if frac:
            mantissa += "." + frac
        sign = "-" if e < 0 else "+"
        text = f"{mantissa}e{sign}{abs(e):02d}"
    else:
        if coef == 0:
            int_part, frac = "0", body[1:]
        elif e >= 0:
            int_part = body[: e + 1].ljust(e + 1, "0")
            frac = body[e + 1 :]
        else:
            int_part = "0"
            frac = "0" * (-e - 1) + body
        if strip_zeros:
            frac = frac.rstrip("0")
        text = int_part + ("." + frac if frac else "")
    return ("-" if negative else "") + text
