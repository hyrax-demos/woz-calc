"""Interactive REPL for the woz calculator: ``python -m woz``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from typing import TextIO

from woz import functions
from woz.bigdec import MAX_STR_BITS, format_significant, to_decimal_string
from woz.evaluator import Evaluator
from woz.lexer import WozError
from woz.rational import Rational

PROMPT = "woz> "

HELP = """\
Enter an expression, e.g.  1/3 + 1/6   2**-3   sqrt(2)   x = pi/4   sin(x)
Operators: + - * / ** (or ^), unary -, parentheses, assignment with =
Functions: sqrt exp ln sin cos atan     Constants: pi e
Commands:  :help  :vars  :digits N  :quit"""


def format_value(value: Rational, digits: int) -> str:
    """Show exact values exactly, anything else to ``digits`` significant digits."""
    if value.is_integer() and value.numerator.bit_length() <= MAX_STR_BITS:
        return str(value)
    exact = to_decimal_string(value, digits)
    if exact is not None:
        return exact
    return format_significant(value, digits)


class Repl:
    def __init__(self, out: TextIO, precision: int = 30) -> None:
        self.out = out
        self.evaluator = Evaluator(precision)

    def _print(self, text: str) -> None:
        print(text, file=self.out)

    def handle(self, line: str) -> bool:
        """Process one input line; return False when the session should end."""
        line = line.strip()
        if not line or line.startswith("#"):
            return True
        if line.startswith(":"):
            return self._command(line[1:].split())
        try:
            value = self.evaluator.evaluate(line)
        except WozError as exc:
            self._print(f"error: {exc}")
        else:
            self.evaluator.variables["_"] = value
            self._print(format_value(value, self.evaluator.precision))
        return True

    def _command(self, words: list[str]) -> bool:
        name = words[0] if words else ""
        if name in ("q", "quit", "exit"):
            return False
        if name in ("h", "help"):
            self._print(HELP)
        elif name == "vars":
            for key, value in sorted(self.evaluator.variables.items()):
                self._print(f"{key} = {format_value(value, self.evaluator.precision)}")
        elif name == "digits":
            if len(words) == 1:
                self._print(str(self.evaluator.precision))
            else:
                try:
                    self.evaluator.precision = int(words[1])
                except ValueError:
                    self._print(
                        f"error: digits must be an integer from "
                        f"{functions.MIN_DIGITS} to {functions.MAX_DIGITS}"
                    )
        else:
            self._print(f"error: unknown command {':' + name!r} (try :help)")
        return True

    def run(self, lines: Iterable[str]) -> None:
        for line in lines:
            if not self.handle(line):
                break


def _interactive(repl: Repl) -> None:
    repl._print("woz exact calculator -- :help for help, :quit to exit")
    while True:
        try:
            line = input(PROMPT)
        except (EOFError, KeyboardInterrupt):
            repl._print("")
            return
        if not repl.handle(line):
            return


def main(argv: list[str] | None = None, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m woz", description="Exact-arithmetic calculator.")
    parser.add_argument("-d", "--digits", type=int, default=30, help="significant digits (1-60)")
    parser.add_argument("-e", "--expr", action="append", default=[], help="evaluate and exit")
    args = parser.parse_args(argv)
    out = stdout if stdout is not None else sys.stdout
    source = stdin if stdin is not None else sys.stdin
    if not functions.MIN_DIGITS <= args.digits <= functions.MAX_DIGITS:
        parser.error(f"--digits must be between {functions.MIN_DIGITS} and {functions.MAX_DIGITS}")
    repl = Repl(out, args.digits)
    if args.expr:
        repl.run(args.expr)
    elif stdin is None and source.isatty():
        _interactive(repl)
    else:
        repl.run(source)
    return 0
