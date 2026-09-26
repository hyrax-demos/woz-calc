"""Interactive REPL: ``python -m woz``."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TextIO

from woz.bigdec import decimal_exponent, format_significant, round_to_rational
from woz.evaluator import Evaluator
from woz.lexer import WozError
from woz.rational import Rational

__all__ = ["format_value", "handle_line", "main", "run_repl"]

PROMPT = "woz> "
HELP = """\
Enter an expression, e.g.  1/3 + 1/6   2**-3   sqrt(2)   x = pi/4   sin(x)
Operators: + - * / ** (or ^), unary -, parentheses.
Functions: sqrt exp ln sin cos atan.  Constants: pi e.
Commands:  :prec [N]  show or set digits (1-60)
           :vars      list variables
           :help      show this help
           :quit      exit (Ctrl-D also works)"""

# Show an exact fraction alongside its decimal when it's at most this long.
_MAX_FRACTION_CHARS = 40


def _strip_zeros(text: str) -> str:
    if "e" in text:
        mantissa, exponent = text.split("e")
        return _strip_zeros(mantissa) + "e" + exponent
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def format_value(value: Rational, precision: int) -> str:
    """Render ``value`` for the REPL.

    Integers print exactly. A value that's exact at ``precision`` digits
    prints as a decimal without trailing zeros; any other fraction prints
    as ``num/den ~= decimal`` (or just the decimal, if the fraction is long).
    """
    if value.is_integer() and (
        value == 0 or decimal_exponent(value) < max(precision, _MAX_FRACTION_CHARS)
    ):
        return str(value.numerator)
    decimal = format_significant(value, precision)
    if round_to_rational(value, precision) == value:
        return _strip_zeros(decimal)
    fraction = str(value)
    if len(fraction) <= _MAX_FRACTION_CHARS:
        return f"{fraction} ~= {decimal}"
    return f"~= {decimal}"


def handle_line(evaluator: Evaluator, line: str) -> str | None:
    """Process one input line and return the text to print.

    Returns ``None`` when the user asked to quit, ``""`` for blank lines.
    """
    text = line.strip()
    if not text or text.startswith("#"):
        return ""
    if text.startswith(":"):
        return _command(evaluator, text[1:].split())
    try:
        value = evaluator.evaluate(text)
    except WozError as exc:
        return f"error: {type(exc).__name__}: {exc}"
    return format_value(value, evaluator.precision)


def _command(evaluator: Evaluator, words: list[str]) -> str | None:
    if not words:
        return "error: empty command (try :help)"
    name, args = words[0], words[1:]
    if name in ("q", "quit", "exit"):
        return None
    if name in ("h", "help"):
        return HELP
    if name == "vars":
        if not evaluator.variables:
            return "(no variables)"
        return "\n".join(
            f"{key} = {format_value(val, evaluator.precision)}"
            for key, val in sorted(evaluator.variables.items())
        )
    if name == "prec":
        if not args:
            return f"precision = {evaluator.precision}"
        try:
            digits = int(args[0])
        except ValueError:
            return "error: precision must be an integer"
        try:
            evaluator.precision = digits
        except ValueError as exc:
            return f"error: {exc}"
        return f"precision = {evaluator.precision}"
    return f"error: unknown command :{name} (try :help)"


def run_repl(lines: Iterable[str], out: TextIO) -> int:
    """Run the REPL over ``lines``, writing results to ``out``."""
    evaluator = Evaluator()
    for line in lines:
        result = handle_line(evaluator, line)
        if result is None:
            break
        if result:
            out.write(result + "\n")
            out.flush()
    return 0


def _interactive_lines() -> Iterable[str]:
    while True:
        try:
            yield input(PROMPT)
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()
            continue


def main(argv: list[str] | None = None) -> int:
    """Entry point. With arguments, evaluate each one; else start the REPL."""
    args = sys.argv[1:] if argv is None else argv
    if args:
        evaluator = Evaluator()
        status = 0
        for arg in args:
            result = handle_line(evaluator, arg)
            if result is None:
                break
            if result.startswith("error:"):
                status = 1
            if result:
                print(result)
        return status
    if sys.stdin.isatty():
        print("woz exact calculator - :help for help, :quit to exit")
        return run_repl(_interactive_lines(), sys.stdout)
    return run_repl(sys.stdin, sys.stdout)
