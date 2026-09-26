import pytest

import calc


@pytest.mark.parametrize(
    "fn, a, b, expected",
    [
        (calc.add, 2, 3, 5),
        (calc.add, 1.5, 2.5, 4.0),
        (calc.subtract, 5, 3, 2),
        (calc.subtract, 0, 4, -4),
        (calc.multiply, 3, 4, 12),
        (calc.multiply, -2, 2.5, -5.0),
    ],
)
def test_arithmetic(fn, a, b, expected):
    assert fn(a, b) == expected


@pytest.mark.parametrize("fn", [calc.add, calc.subtract, calc.multiply])
def test_none_raises_value_error(fn):
    with pytest.raises(ValueError):
        fn(None, 1)
    with pytest.raises(ValueError):
        fn(1, None)


@pytest.mark.parametrize("fn", [calc.add, calc.subtract, calc.multiply])
def test_non_number_raises_type_error(fn):
    with pytest.raises(TypeError):
        fn("1", 2)
    with pytest.raises(TypeError):
        fn(1, [2])


def test_divide():
    assert calc.divide(6, 3) == 2


def test_divide_by_zero_raises():
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        calc.divide(1, 0)


def test_divide_none_raises_value_error():
    with pytest.raises(ValueError, match="inputs must not be None"):
        calc.divide(None, 1)
