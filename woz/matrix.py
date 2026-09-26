"""Exact matrices over :class:`~woz.rational.Rational`.

All algorithms use fraction-exact Gaussian elimination, so results are
exact: there is no pivoting tolerance and no rounding error.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from woz.rational import Rational, RationalLike


class MatrixError(ValueError):
    """Raised for shape mismatches and singular systems."""


class SingularMatrixError(MatrixError):
    """Raised when an inverse or unique solution does not exist."""


def _to_rational(x: RationalLike | str) -> Rational:
    if isinstance(x, bool):
        raise TypeError("booleans are not matrix entries")
    if isinstance(x, Rational):
        return x
    if isinstance(x, (int, str)):
        return Rational(x)
    raise TypeError(f"unsupported matrix entry: {type(x).__name__}")


class Matrix:
    """An immutable ``rows x cols`` matrix of Rationals."""

    __slots__ = ("_ncols", "_rows")

    _rows: tuple[tuple[Rational, ...], ...]
    _ncols: int

    def __init__(self, rows: Iterable[Iterable[RationalLike | str]]) -> None:
        data = tuple(tuple(_to_rational(v) for v in row) for row in rows)
        if not data or not data[0]:
            raise MatrixError("a matrix needs at least one row and one column")
        width = len(data[0])
        if any(len(row) != width for row in data):
            raise MatrixError("all rows must have the same length")
        self._rows = data
        self._ncols = width

    # -- constructors ----------------------------------------------------
    @classmethod
    def identity(cls, n: int) -> Matrix:
        if n < 1:
            raise MatrixError("identity size must be >= 1")
        return cls([[1 if i == j else 0 for j in range(n)] for i in range(n)])

    @classmethod
    def zeros(cls, rows: int, cols: int) -> Matrix:
        if rows < 1 or cols < 1:
            raise MatrixError("matrix dimensions must be >= 1")
        return cls([[0] * cols for _ in range(rows)])

    # -- basic protocol --------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return len(self._rows), self._ncols

    @property
    def rows(self) -> tuple[tuple[Rational, ...], ...]:
        return self._rows

    def __getitem__(self, index: tuple[int, int]) -> Rational:
        i, j = index
        return self._rows[i][j]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self._rows == other._rows

    def __hash__(self) -> int:
        return hash(self._rows)

    def __repr__(self) -> str:
        body = ", ".join("[" + ", ".join(str(v) for v in r) + "]" for r in self._rows)
        return f"Matrix([{body}])"

    def is_square(self) -> bool:
        return len(self._rows) == self._ncols

    def transpose(self) -> Matrix:
        return Matrix(zip(*self._rows))

    # -- arithmetic ------------------------------------------------------
    def _same_shape(self, other: Matrix) -> None:
        if self.shape != other.shape:
            raise MatrixError(f"shape mismatch: {self.shape} vs {other.shape}")

    def __add__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented
        self._same_shape(other)
        return Matrix(
            [a + b for a, b in zip(r, s)] for r, s in zip(self._rows, other._rows)
        )

    def __sub__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented
        self._same_shape(other)
        return Matrix(
            [a - b for a, b in zip(r, s)] for r, s in zip(self._rows, other._rows)
        )

    def __neg__(self) -> Matrix:
        return Matrix([-a for a in r] for r in self._rows)

    def scale(self, k: RationalLike) -> Matrix:
        k = _to_rational(k)
        return Matrix([k * a for a in r] for r in self._rows)

    def __matmul__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented
        if self._ncols != len(other._rows):
            raise MatrixError(f"cannot multiply {self.shape} by {other.shape}")
        cols = list(zip(*other._rows))
        return Matrix(
            [sum((a * b for a, b in zip(r, c)), Rational(0)) for c in cols]
            for r in self._rows
        )

    # -- elimination -----------------------------------------------------
    def _echelon(self) -> tuple[list[list[Rational]], int, int]:
        """Row-reduce a copy to reduced echelon form.

        Returns ``(rows, rank, swaps)`` where ``swaps`` counts row exchanges.
        """
        m = [list(r) for r in self._rows]
        nrows, ncols = len(m), self._ncols
        rank = 0
        swaps = 0
        for col in range(ncols):
            pivot = next((i for i in range(rank, nrows) if m[i][col]), None)
            if pivot is None:
                continue
            if pivot != rank:
                m[rank], m[pivot] = m[pivot], m[rank]
                swaps += 1
            inv = 1 / m[rank][col]
            m[rank] = [v * inv for v in m[rank]]
            for i in range(nrows):
                if i != rank and m[i][col]:
                    f = m[i][col]
                    m[i] = [a - f * b for a, b in zip(m[i], m[rank])]
            rank += 1
            if rank == nrows:
                break
        return m, rank, swaps

    def rref(self) -> Matrix:
        """Return the reduced row echelon form."""
        return Matrix(self._echelon()[0])

    def rank(self) -> int:
        return self._echelon()[1]

    def det(self) -> Rational:
        """Exact determinant via elimination (product of pivots)."""
        if not self.is_square():
            raise MatrixError("determinant requires a square matrix")
        m = [list(r) for r in self._rows]
        n = len(m)
        result = Rational(1)
        for col in range(n):
            pivot = next((i for i in range(col, n) if m[i][col]), None)
            if pivot is None:
                return Rational(0)
            if pivot != col:
                m[col], m[pivot] = m[pivot], m[col]
                result = -result
            p = m[col][col]
            result *= p
            for i in range(col + 1, n):
                if m[i][col]:
                    f = m[i][col] / p
                    m[i] = [a - f * b for a, b in zip(m[i], m[col])]
        return result

    def inverse(self) -> Matrix:
        if not self.is_square():
            raise MatrixError("inverse requires a square matrix")
        n = len(self._rows)
        aug = Matrix(
            list(r) + [Rational(1 if i == j else 0) for j in range(n)]
            for i, r in enumerate(self._rows)
        )
        reduced, _, _ = aug._echelon()
        for i in range(n):
            if reduced[i][i] != 1 or any(reduced[i][j] for j in range(n) if j != i):
                raise SingularMatrixError("matrix is singular")
        return Matrix(r[n:] for r in reduced)

    def solve(self, b: Matrix | Sequence[RationalLike | str]) -> Matrix:
        """Solve ``self @ x == b`` for a square, non-singular ``self``.

        ``b`` may be a Matrix (one or more right-hand-side columns) or a
        flat sequence, treated as a single column.  Returns a Matrix with
        the same number of columns as ``b``.
        """
        if not self.is_square():
            raise MatrixError("solve requires a square matrix")
        rhs = b if isinstance(b, Matrix) else Matrix([v] for v in b)
        n = len(self._rows)
        if len(rhs._rows) != n:
            raise MatrixError("right-hand side has the wrong number of rows")
        aug = Matrix(list(r) + list(s) for r, s in zip(self._rows, rhs._rows))
        reduced, _, _ = aug._echelon()
        for i in range(n):
            if reduced[i][i] != 1:
                raise SingularMatrixError("system has no unique solution")
        return Matrix(r[n:] for r in reduced)
