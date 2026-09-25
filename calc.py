"""A tiny calculator module (intentionally un-refactored for a WoZ program run)."""


def add(a, b):
    if a is None or b is None:
        raise ValueError("inputs must not be None")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("inputs must be numbers")
    return a + b


def subtract(a, b):
    if a is None or b is None:
        raise ValueError("inputs must not be None")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("inputs must be numbers")
    return a - b


def multiply(a, b):
    if a is None or b is None:
        raise ValueError("inputs must not be None")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("inputs must be numbers")
    return a * b


def divide(a, b):
    if a is None or b is None:
        raise ValueError("inputs must not be None")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("inputs must be numbers")
    if b == 0:
        raise ValueError("cannot divide by zero")
    return a / b
