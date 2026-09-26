"""Tests for woz.matrix exact linear algebra."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from woz.matrix import Matrix, MatrixError, SingularMatrixError
from woz.rational import Rational

R = Rational


def test_construction_and_shape() -> None:
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    assert m.shape == (2, 3)
    assert m[1, 2] == 6
    assert Matrix([["1/2", "0.25"]])[0, 1] == R(1, 4)


@pytest.mark.parametrize("rows", [[], [[]], [[1, 2], [3]]])
def test_bad_shapes(rows: list[list[int]]) -> None:
    with pytest.raises(MatrixError):
        Matrix(rows)


def test_bad_entry_type() -> None:
    with pytest.raises(TypeError):
        Matrix([[1.5]])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        Matrix([[True]])


def test_identity_zeros_transpose() -> None:
    assert Matrix.identity(2) == Matrix([[1, 0], [0, 1]])
    assert Matrix.zeros(1, 2) == Matrix([[0, 0]])
    assert Matrix([[1, 2]]).transpose() == Matrix([[1], [2]])
    with pytest.raises(MatrixError):
        Matrix.identity(0)


def test_arithmetic() -> None:
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    assert a + b == Matrix([[6, 8], [10, 12]])
    assert b - a == Matrix([[4, 4], [4, 4]])
    assert -a == Matrix([[-1, -2], [-3, -4]])
    assert a.scale(R(1, 2)) == Matrix([["1/2", 1], ["3/2", 2]])
    assert a @ b == Matrix([[19, 22], [43, 50]])
    with pytest.raises(MatrixError):
        a + Matrix([[1, 2]])
    with pytest.raises(MatrixError):
        Matrix([[1, 2]]) @ Matrix([[1, 2]])


def test_hash_and_repr() -> None:
    a = Matrix([[1, "1/2"]])
    assert hash(a) == hash(Matrix([[1, R(1, 2)]]))
    assert repr(a) == "Matrix([[1, 1/2]])"


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([[3]], R(3)),
        ([[1, 2], [3, 4]], R(-2)),
        ([[0, 1], [1, 0]], R(-1)),
        ([[2, 0, 0], [0, 3, 0], [0, 0, 4]], R(24)),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], R(0)),
        ([["1/2", "1/3"], ["1/4", "1/5"]], R(1, 60)),
        ([[0, 0, 1], [0, 1, 0], [1, 0, 0]], R(-1)),
        ([[1, 1, 1, 1], [1, 2, 4, 8], [1, 3, 9, 27], [1, 4, 16, 64]], R(12)),
    ],
)
def test_det(rows: list[list[object]], expected: Rational) -> None:
    assert Matrix(rows).det() == expected  # type: ignore[arg-type]


def test_det_non_square() -> None:
    with pytest.raises(MatrixError):
        Matrix([[1, 2]]).det()


def test_hilbert_inverse_is_exact() -> None:
    n = 5
    h = Matrix([[R(1, i + j + 1) for j in range(n)] for i in range(n)])
    inv = h.inverse()
    assert h @ inv == Matrix.identity(n)
    assert inv[0, 0] == 25
    assert inv[4, 4] == 44100


def test_inverse_2x2() -> None:
    assert Matrix([[4, 7], [2, 6]]).inverse() == Matrix(
        [["3/5", "-7/10"], ["-1/5", "2/5"]]
    )


@pytest.mark.parametrize("rows", [[[1, 2], [2, 4]], [[0, 0], [0, 0]]])
def test_inverse_singular(rows: list[list[int]]) -> None:
    with pytest.raises(SingularMatrixError):
        Matrix(rows).inverse()


def test_inverse_non_square() -> None:
    with pytest.raises(MatrixError):
        Matrix([[1, 2]]).inverse()


def test_solve() -> None:
    a = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
    x = a.solve([8, -11, -3])
    assert x == Matrix([[2], [3], [-1]])
    two = a.solve(Matrix([[8, 1], [-11, 0], [-3, 0]]))
    assert two.shape == (3, 2)
    assert a @ two == Matrix([[8, 1], [-11, 0], [-3, 0]])


def test_solve_needs_pivot_swap() -> None:
    assert Matrix([[0, 1], [1, 0]]).solve(["1/2", 3]) == Matrix([[3], ["1/2"]])


def test_solve_errors() -> None:
    with pytest.raises(SingularMatrixError):
        Matrix([[1, 1], [2, 2]]).solve([1, 2])
    with pytest.raises(MatrixError):
        Matrix([[1, 0], [0, 1]]).solve([1, 2, 3])
    with pytest.raises(MatrixError):
        Matrix([[1, 2]]).solve([1])


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([[0, 0], [0, 0]], 0),
        ([[1, 2], [2, 4]], 1),
        ([[1, 2], [3, 4]], 2),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2),
        ([[1, 2, 3, 4]], 1),
        ([[1], [2], [3]], 1),
        ([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], 3),
    ],
)
def test_rank(rows: list[list[int]], expected: int) -> None:
    assert Matrix(rows).rank() == expected


def test_rref() -> None:
    assert Matrix([[2, 4], [1, 3]]).rref() == Matrix.identity(2)
    assert Matrix([[1, 2], [2, 4]]).rref() == Matrix([[1, 2], [0, 0]])


def _fraction_det(rows: list[list[Fraction]]) -> Fraction:
    # Independent Laplace expansion for small matrices.
    if len(rows) == 1:
        return rows[0][0]
    total = Fraction(0)
    for j, v in enumerate(rows[0]):
        minor = [r[:j] + r[j + 1 :] for r in rows[1:]]
        total += (-1) ** j * v * _fraction_det(minor)
    return total


@pytest.mark.parametrize("seed", range(15))
def test_random_det_inverse_solve(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 4)
    raw = [
        [(rng.randint(-9, 9), rng.randint(1, 5)) for _ in range(n)] for _ in range(n)
    ]
    m = Matrix([[R(a, b) for a, b in row] for row in raw])
    d = _fraction_det([[Fraction(a, b) for a, b in row] for row in raw])
    det = m.det()
    assert (det.numerator, det.denominator) == (d.numerator, d.denominator)
    if d:
        assert m @ m.inverse() == Matrix.identity(n)
        b = [R(rng.randint(-9, 9)) for _ in range(n)]
        x = m.solve(b)
        assert m @ x == Matrix([v] for v in b)
        assert m.rank() == n
    else:
        assert m.rank() < n
