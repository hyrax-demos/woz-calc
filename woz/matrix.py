"""Exact matrices of Rationals, using fraction-exact Gaussian elimination."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from woz.rational import Rational, RationalLike

__all__ = ["Matrix", "MatrixError", "SingularMatrixError"]


class MatrixError(ValueError):
    """Raised for shape mismatches and malformed matrices."""


class SingularMatrixError(MatrixError, ZeroDivisionError):
    """Raised when an inverse or a unique solution does not exist."""


def _coerce(value: object) -> Rational:
    if isinstance(value, Rational):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Rational(value)
    if isinstance(value, str):
        return Rational.from_str(value)
    raise TypeError(f"matrix entries must be Rational, int or str, not {type(value).__name__}")


class Matrix:
    """An immutable ``rows x cols`` matrix of Rationals."""

    __slots__ = ("_rows", "_nrows", "_ncols")

    _rows: tuple[tuple[Rational, ...], ...]
    _nrows: int
    _ncols: int

    def __init__(self, rows: Iterable[Iterable[RationalLike | str]]) -> None:
        data = tuple(tuple(_coerce(v) for v in row) for row in rows)
        if not data or not data[0]:
            raise MatrixError("a matrix needs at least one row and one column")
        width = len(data[0])
        if any(len(row) != width for row in data):
            raise MatrixError("all rows must have the same length")
        self._rows = data
        self._nrows = len(data)
        self._ncols = width

    @classmethod
    def identity(cls, n: int) -> Matrix:
        if n < 1:
            raise MatrixError("identity size must be at least 1")
        return cls([[1 if i == j else 0 for j in range(n)] for i in range(n)])

    @classmethod
    def zeros(cls, rows: int, cols: int) -> Matrix:
        if rows < 1 or cols < 1:
            raise MatrixError("matrix dimensions must be at least 1")
        return cls([[0] * cols for _ in range(rows)])

    # ------------------------------------------------------------ accessors

    @property
    def shape(self) -> tuple[int, int]:
        return self._nrows, self._ncols

    @property
    def is_square(self) -> bool:
        return self._nrows == self._ncols

    def __getitem__(self, index: tuple[int, int]) -> Rational:
        i, j = index
        return self._rows[i][j]

    def row(self, i: int) -> tuple[Rational, ...]:
        return self._rows[i]

    def column(self, j: int) -> tuple[Rational, ...]:
        return tuple(row[j] for row in self._rows)

    def to_lists(self) -> list[list[Rational]]:
        return [list(row) for row in self._rows]

    def __iter__(self) -> Iterator[tuple[Rational, ...]]:
        return iter(self._rows)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self._rows == other._rows

    def __hash__(self) -> int:
        return hash(self._rows)

    def __repr__(self) -> str:
        body = ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in self._rows)
        return f"Matrix([{body}])"

    def __str__(self) -> str:
        cells = [[str(v) for v in row] for row in self._rows]
        width = max(len(c) for row in cells for c in row)
        return "\n".join("[ " + "  ".join(c.rjust(width) for c in row) + " ]" for row in cells)

    # ----------------------------------------------------------- arithmetic

    def _check_same_shape(self, other: Matrix) -> None:
        if self.shape != other.shape:
            raise MatrixError(f"shape mismatch: {self.shape} vs {other.shape}")

    def __add__(self, other: object) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented
        self._check_same_shape(other)
        return Matrix(
            [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(self._rows, other._rows)]
        )

    def __sub__(self, other: object) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented
        self._check_same_shape(other)
        return Matrix(
            [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(self._rows, other._rows)]
        )

    def __neg__(self) -> Matrix:
        return Matrix([[-v for v in row] for row in self._rows])

    def __mul__(self, other: object) -> Matrix:
        if isinstance(other, (Rational, int)) and not isinstance(other, bool):
            return Matrix([[v * other for v in row] for row in self._rows])
        return NotImplemented

    def __rmul__(self, other: object) -> Matrix:
        return self.__mul__(other)

    def __matmul__(self, other: object) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented
        if self._ncols != other._nrows:
            raise MatrixError(f"cannot multiply {self.shape} by {other.shape}")
        cols = [other.column(j) for j in range(other._ncols)]
        return Matrix(
            [[sum((a * b for a, b in zip(row, col)), Rational(0)) for col in cols] for row in self._rows]
        )

    def transpose(self) -> Matrix:
        return Matrix([self.column(j) for j in range(self._ncols)])

    # ---------------------------------------------------------- elimination

    def _echelon(self) -> tuple[list[list[Rational]], list[int], int]:
        """Reduce to reduced row-echelon form.

        Returns ``(rows, pivot_columns, swaps)``.
        """
        rows = self.to_lists()
        pivots: list[int] = []
        swaps = 0
        r = 0
        for c in range(self._ncols):
            if r == self._nrows:
                break
            pivot = next((i for i in range(r, self._nrows) if rows[i][c]), None)
            if pivot is None:
                continue
            if pivot != r:
                rows[r], rows[pivot] = rows[pivot], rows[r]
                swaps += 1
            lead = rows[r][c]
            rows[r] = [v / lead for v in rows[r]]
            for i in range(self._nrows):
                factor = rows[i][c]
                if i != r and factor:
                    rows[i] = [a - factor * b for a, b in zip(rows[i], rows[r])]
            pivots.append(c)
            r += 1
        return rows, pivots, swaps

    def rref(self) -> Matrix:
        return Matrix(self._echelon()[0])

    def rank(self) -> int:
        return len(self._echelon()[1])

    def det(self) -> Rational:
        if not self.is_square:
            raise MatrixError("determinant needs a square matrix")
        rows = self.to_lists()
        n = self._nrows
        result = Rational(1)
        for c in range(n):
            pivot = next((i for i in range(c, n) if rows[i][c]), None)
            if pivot is None:
                return Rational(0)
            if pivot != c:
                rows[c], rows[pivot] = rows[pivot], rows[c]
                result = -result
            lead = rows[c][c]
            result *= lead
            for i in range(c + 1, n):
                factor = rows[i][c] / lead
                if factor:
                    rows[i] = [a - factor * b for a, b in zip(rows[i], rows[c])]
        return result

    def inverse(self) -> Matrix:
        if not self.is_square:
            raise MatrixError("only square matrices have inverses")
        n = self._nrows
        augmented = Matrix(
            [list(row) + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(self._rows)]
        )
        rows, pivots, _ = augmented._echelon()
        if pivots[:n] != list(range(n)):
            raise SingularMatrixError("matrix is singular")
        return Matrix([row[n:] for row in rows])

    def solve(self, b: Matrix | Sequence[RationalLike | str]) -> Matrix | tuple[Rational, ...]:
        """Solve ``self @ x == b`` for a unique ``x``.

        ``b`` may be a Matrix (giving a Matrix) or a flat sequence (giving
        a tuple). Raises SingularMatrixError when there is no solution or
        infinitely many.
        """
        vector = not isinstance(b, Matrix)
        rhs = Matrix([[v] for v in b]) if vector else b
        assert isinstance(rhs, Matrix)
        if rhs._nrows != self._nrows:
            raise MatrixError(f"right-hand side has {rhs._nrows} rows, expected {self._nrows}")
        n = self._ncols
        augmented = Matrix([list(a) + list(r) for a, r in zip(self._rows, rhs._rows)])
        rows, pivots, _ = augmented._echelon()
        if any(p >= n for p in pivots):
            raise SingularMatrixError("system is inconsistent")
        if len(pivots) != n:
            raise SingularMatrixError("system has infinitely many solutions")
        solution = Matrix([rows[i][n:] for i in range(n)])
        if vector:
            return solution.column(0)
        return solution
