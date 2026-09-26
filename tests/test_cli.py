"""Tests for woz.cli (the REPL front end)."""

from __future__ import annotations

import io
import subprocess
import sys

import pytest

from woz.cli import handle_line, main, run
from woz.evaluator import Evaluator


def feed(*lines: str, digits: int = 30) -> str:
    out = io.StringIO()
    run(lines, out, digits=digits)
    return out.getvalue()


def test_basic_session() -> None:
    assert feed("1/3 + 1/6", "x = 2", "x ** 3") == "1/2\n2\n8\n"


def test_blank_and_comment_lines() -> None:
    assert feed("", "   ", "# note", "1") == "1\n"


def test_errors_are_reported_not_raised() -> None:
    out = feed("1/0", "nope", "1 +", "2")
    lines = out.splitlines()
    assert lines[0].startswith("error: division by zero")
    assert lines[1].startswith("error: undefined name")
    assert lines[2].startswith("error: ")
    assert lines[3] == "2"


def test_quit_stops_processing() -> None:
    assert feed("1", ":quit", "2") == "1\n"


@pytest.mark.parametrize("cmd", [":q", ":quit", ":exit"])
def test_quit_aliases(cmd: str) -> None:
    assert handle_line(Evaluator(), cmd) == (None, False)


def test_digits_command() -> None:
    out = feed(":digits", ":digits 5", "pi", ":digits 99", ":digits x")
    assert out.splitlines() == [
        "digits = 30",
        "digits = 5",
        "3.1416",
        "error: digits must be between 1 and 60",
        "error: not an integer: 'x'",
    ]


def test_vars_and_help() -> None:
    assert feed(":vars") == "(no variables)\n"
    assert feed("b = 2", "a = 1/2", ":vars") == "2\n1/2\na = 1/2\nb = 2\n"
    assert "functions" in feed(":help")


@pytest.mark.parametrize("cmd", [":", ":bogus"])
def test_bad_commands(cmd: str) -> None:
    out, keep = handle_line(Evaluator(), cmd)
    assert keep and out is not None and out.startswith("error:")


def test_prompt_written() -> None:
    out = io.StringIO()
    run(["1"], out, prompt="> ")
    assert out.getvalue() == "> 1\n"


def test_main_eval(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-d", "5", "-e", "sqrt(2)", "-e", "1/2"]) == 0
    assert capsys.readouterr().out == "1.4142\n1/2\n"


def test_main_bad_digits() -> None:
    with pytest.raises(SystemExit):
        main(["-d", "0", "-e", "1"])


def test_python_dash_m() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "woz"],
        input="x = 3/4\nx * 4\n:quit\n",
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert proc.stdout == "3/4\n3\n"
