"""An interactive read-eval-print loop: ``python -m woz``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from typing import TextIO

from woz.evaluator import Evaluator
from woz.lexer import WozError

PROMPT: str = "woz> "

HELP_TEXT: str = """\
Exact rational calculator.
  expressions   1/3 + 1/6, 2**-3, (1 + 2) * 3, -2**2
  variables     x = 3/4   then   x * 4
  functions     sqrt exp ln sin cos atan abs; constants pi e
  commands      :digits N (1-60)   :vars   :help   :quit
"""


def handle_line(evaluator: Evaluator, line: str) -> tuple[str | None, bool]:
    """Process one line of input.

    Returns ``(output, keep_going)``; ``output`` is None for blank lines.
    """
    text = line.strip()
    if not text or text.startswith("#"):
        return None, True
    if text.startswith(":"):
        return _command(evaluator, text[1:].split())
    try:
        value = evaluator.evaluate(text)
    except WozError as exc:
        return f"error: {exc}", True
    return evaluator.format(value), True


def _command(evaluator: Evaluator, parts: list[str]) -> tuple[str | None, bool]:
    if not parts:
        return "error: empty command", True
    name, args = parts[0], parts[1:]
    if name in ("q", "quit", "exit"):
        return None, False
    if name in ("h", "help"):
        return HELP_TEXT.rstrip("\n"), True
    if name == "digits":
        if not args:
            return f"digits = {evaluator.digits}", True
        try:
            evaluator.digits = int(args[0])
        except ValueError:
            return f"error: not an integer: {args[0]!r}", True
        except WozError as exc:
            return f"error: {exc}", True
        return f"digits = {evaluator.digits}", True
    if name == "vars":
        if not evaluator.variables:
            return "(no variables)", True
        return "\n".join(
            f"{k} = {evaluator.format(v)}"
            for k, v in sorted(evaluator.variables.items())
        ), True
    return f"error: unknown command :{name}", True


def run(
    lines: Iterable[str], out: TextIO, *, digits: int = 30, prompt: str = ""
) -> int:
    """Feed ``lines`` through a fresh evaluator, writing results to ``out``."""
    evaluator = Evaluator(digits)
    for line in lines:
        if prompt:
            out.write(prompt)
        output, keep_going = handle_line(evaluator, line)
        if output is not None:
            out.write(output + "\n")
        if not keep_going:
            break
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


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="woz", description="Exact rational calculator")
    ap.add_argument("-d", "--digits", type=int, default=30, help="precision (1-60)")
    ap.add_argument(
        "-e",
        "--eval",
        dest="exprs",
        action="append",
        default=[],
        metavar="EXPR",
        help="evaluate EXPR and exit (repeatable)",
    )
    ns = ap.parse_args(argv)
    if not 1 <= ns.digits <= 60:
        ap.error("--digits must be between 1 and 60")
    if ns.exprs:
        return run(ns.exprs, sys.stdout, digits=ns.digits)
    if sys.stdin.isatty():
        print("woz exact calculator - :help for help, :quit to exit")
        return run(_interactive_lines(), sys.stdout, digits=ns.digits)
    return run(sys.stdin, sys.stdout, digits=ns.digits)
