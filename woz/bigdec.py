"""Round a Rational to N significant decimal digits (round-half-even)."""

from __future__ import annotations

from woz.rational import Rational, RationalLike

__all__ = [
    "MAX_DIGITS",
    "decimal_exponent",
    "format_significant",
    "round_significant",
    "round_to_rational",
]

# Upper bound on the digit count accepted here; far above what the
# functions module allows, but it stops a typo from allocating gigabytes.
MAX_DIGITS = 100_000


def _check_digits(digits: int) -> None:
    if isinstance(digits, bool) or not isinstance(digits, int):
        raise TypeError("digits must be an int")
    if not 1 <= digits <= MAX_DIGITS:
        raise ValueError(f"digits must be between 1 and {MAX_DIGITS}, got {digits}")


def decimal_exponent(x: RationalLike) -> int:
    """Return ``k`` with ``10**k <= |x| < 10**(k+1)``; ``x`` must be non-zero."""
    x = Rational(x)
    num, den = abs(x.numerator), x.denominator
    if num == 0:
        raise ValueError("zero has no decimal exponent")
    # Estimate from bit lengths (avoids str() on huge ints), then correct.
    k = ((num.bit_length() - den.bit_length()) * 301029995663981) // 10**15
    while _at_least(num, den, k + 1):
        k += 1
    while not _at_least(num, den, k):
        k -= 1
    return k


def _at_least(num: int, den: int, k: int) -> bool:
    """Return whether ``num / den >= 10**k``."""
    if k >= 0:
        return num >= den * 10**k
    return num * 10**-k >= den


def round_significant(x: RationalLike, digits: int) -> tuple[int, int]:
    """Round ``x`` to ``digits`` significant digits.

    Returns ``(mantissa, exponent)`` with ``x ~= mantissa * 10**exponent`` and
    ``10**(digits-1) <= |mantissa| < 10**digits``. Zero gives ``(0, 0)``.
    Ties go to the even mantissa.
    """
    _check_digits(digits)
    x = Rational(x)
    if x.numerator == 0:
        return 0, 0
    negative = x.numerator < 0
    num, den = abs(x.numerator), x.denominator
    k = decimal_exponent(x)
    shift = digits - 1 - k
    if shift >= 0:
        num *= 10**shift
    else:
        den *= 10**-shift
    q, r = divmod(num, den)
    twice = 2 * r
    if twice > den or (twice == den and q % 2 == 1):
        q += 1
    exponent = -shift
    if q == 10**digits:
        q //= 10
        exponent += 1
    return (-q if negative else q), exponent


def round_to_rational(x: RationalLike, digits: int) -> Rational:
    """Return ``x`` rounded to ``digits`` significant digits, as a Rational."""
    mantissa, exponent = round_significant(x, digits)
    if exponent >= 0:
        return Rational(mantissa * 10**exponent)
    return Rational(mantissa, 10**-exponent)


def format_significant(x: RationalLike, digits: int, *, scientific: bool | None = None) -> str:
    """Format ``x`` with exactly ``digits`` significant digits.

    Plain notation is used when the decimal exponent is in ``[-6, digits)``,
    scientific (``1.234e-9``) otherwise; pass ``scientific`` to force either.
    Zero prints as ``0`` followed by ``digits - 1`` zero decimals.
    """
    mantissa, exponent = round_significant(x, digits)
    if mantissa == 0:
        return "0" if digits == 1 else "0." + "0" * (digits - 1)
    sign = "-" if mantissa < 0 else ""
    text = str(abs(mantissa))
    k = exponent + digits - 1  # decimal exponent of the leading digit
    use_sci = scientific if scientific is not None else not (-6 <= k < digits)
    if use_sci:
        body = text[0] + ("." + text[1:] if digits > 1 else "")
        return f"{sign}{body}e{k}"
    if k < 0:
        return f"{sign}0.{'0' * (-k - 1)}{text}"
    if k >= digits - 1:
        return sign + text + "0" * (k - digits + 1)
    return f"{sign}{text[: k + 1]}.{text[k + 1 :]}"
