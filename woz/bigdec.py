"""Round a :class:`~woz.rational.Rational` to N significant digits (half-even)."""

from __future__ import annotations

import math

from woz.rational import Rational, RationalLike, as_rational

MAX_DIGITS = 10_000
#: Integers longer than this many bits are never converted with ``str``.
MAX_STR_BITS = 12_000
_LOG10_2 = math.log10(2)


def _check_digits(digits: int) -> None:
    if isinstance(digits, bool) or not isinstance(digits, int):
        raise TypeError("digits must be an int")
    if not 1 <= digits <= MAX_DIGITS:
        raise ValueError(f"digits must be between 1 and {MAX_DIGITS}")


def decimal_exponent(value: Rational) -> int:
    """Return ``k`` with ``10**k <= |value| < 10**(k + 1)`` (``value != 0``)."""
    num, den = abs(value.numerator), value.denominator
    if num == 0:
        raise ValueError("decimal_exponent() of zero is undefined")
    # Estimate from bit lengths (off by at most one or two), then settle exactly.
    k = math.floor((num.bit_length() - den.bit_length()) * _LOG10_2)
    while _pow10_cmp(num, den, k) < 0:
        k -= 1
    while _pow10_cmp(num, den, k + 1) >= 0:
        k += 1
    return k


def _pow10_cmp(num: int, den: int, k: int) -> int:
    """Compare ``num / den`` with ``10**k``; return -1, 0 or 1."""
    lhs, rhs = (num, den * 10**k) if k >= 0 else (num * 10**-k, den)
    return (lhs > rhs) - (lhs < rhs)


def _div_round_half_even(num: int, den: int) -> int:
    """Round ``num / den`` (``den > 0``) to the nearest int, ties to even."""
    q, r = divmod(num, den)
    twice = 2 * r
    if twice > den or (twice == den and q % 2 == 1):
        q += 1
    return q


def round_significant(value: RationalLike, digits: int) -> tuple[int, int]:
    """Round to ``digits`` significant digits.

    Returns ``(mantissa, exponent)`` with ``value ~= mantissa * 10**exponent``
    and ``10**(digits-1) <= |mantissa| < 10**digits`` (or ``(0, 0)`` for zero).
    Ties are broken towards an even last digit.
    """
    _check_digits(digits)
    x = as_rational(value)
    if not x:
        return 0, 0
    exponent = decimal_exponent(x) - digits + 1
    num, den = abs(x.numerator), x.denominator
    if exponent >= 0:
        den *= 10**exponent
    else:
        num *= 10**-exponent
    mantissa = _div_round_half_even(num, den)
    if mantissa == 10**digits:
        mantissa //= 10
        exponent += 1
    return (-mantissa if x < 0 else mantissa), exponent


def round_to_rational(value: RationalLike, digits: int) -> Rational:
    """Round to ``digits`` significant digits and return the result exactly."""
    mantissa, exponent = round_significant(value, digits)
    if exponent >= 0:
        return Rational(mantissa * 10**exponent)
    return Rational(mantissa, 10**-exponent)


def format_significant(
    value: RationalLike, digits: int, *, scientific: bool | None = None
) -> str:
    """Format with exactly ``digits`` significant digits, rounding half-even.

    ``scientific=None`` picks plain notation for moderate magnitudes and
    ``d.ddde±x`` notation otherwise; ``True``/``False`` force a style.
    """
    mantissa, exponent = round_significant(value, digits)
    sign = "-" if mantissa < 0 else ""
    body = str(abs(mantissa)).rjust(digits, "0")
    if mantissa == 0:
        body = "0" * digits
    # Decimal exponent of the leading digit.
    lead = exponent + digits - 1
    if scientific is None:
        scientific = not (-7 < lead < max(digits, 21))
    if scientific:
        head = body[0] + ("." + body[1:] if digits > 1 else "")
        return f"{sign}{head}e{'+' if lead >= 0 else '-'}{abs(lead)}"
    if exponent >= 0:
        return sign + body + "0" * exponent
    point = len(body) + exponent
    if point > 0:
        return f"{sign}{body[:point]}.{body[point:]}"
    return f"{sign}0.{'0' * -point}{body}"


def to_decimal_string(value: RationalLike, max_digits: int = 60) -> str | None:
    """Return the exact plain decimal expansion if it terminates.

    ``None`` is returned when the expansion does not terminate or would need
    more than ``max_digits`` significant digits.
    """
    x = as_rational(value)
    den = x.denominator
    twos = fives = 0
    while den % 2 == 0:
        den //= 2
        twos += 1
    while den % 5 == 0:
        den //= 5
        fives += 1
    if den != 1:
        return None
    if not x:
        return "0"
    scale = max(twos, fives)
    scaled = abs(x.numerator) * 10**scale // x.denominator
    if scaled.bit_length() > MAX_STR_BITS:
        return None
    text = str(scaled)
    sig = len(text.strip("0")) if scale == 0 else len(text.lstrip("0"))
    if sig > max_digits:
        return None
    sign = "-" if x < 0 else ""
    if scale == 0:
        return sign + text
    text = text.rjust(scale + 1, "0")
    return f"{sign}{text[:-scale]}.{text[-scale:]}"
