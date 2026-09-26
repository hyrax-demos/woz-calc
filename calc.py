"""A tiny calculator module (intentionally un-refactored for a WoZ program run)."""


def _validate(a, b):
    if a is None or b is None:
        raise ValueError("inputs must not be None")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("inputs must be numbers")


def add(a, b):
    _validate(a, b)
    return a + b


def subtract(a, b):
    _validate(a, b)
    return a - b


def multiply(a, b):
    _validate(a, b)
    return a * b
