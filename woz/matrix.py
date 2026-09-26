"""Exact matrices of :class:`~woz.rational.Rational` entries."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from woz.rational import Rational, RationalLike, as_rational

Entry = RationalLike | str


class MatrixError(ValueError):
    """Raised for shape mismatches and singular systems."""


class SingularMatrixError(MatrixError):
    """Raised when an inverse or unique solution does not exist."""


class Matrix:
    """An immutable ``rows x cols`` matrix with exact Rational entries."""

    __slots__ = ("_rows", "_shape")

    def __init__(self, rows: Iterable[Iterable[Entry]]) -> None:
        data = tuple(tuple(as_rational(v) for v in row) for row in rows)
        if not data or not data[0]:
            raise MatrixError("a matrix needs at least one row and one column")
        width = len(data[0])
        if any(len(row) != width for row in data):
            raise MatrixError("all rows must have the same length")
        self._rows: tuple[tuple[Rational, ...], ...] = data
        self._shape = (len(data), width)

    # ------------------------------------------------------------ construction
    @classmethod
    def identity(cls, n: int) -> Matrix:
        return cls([[1 if i == j else 0 for j in range(n)] for i in range(n)])

    @classmethod
    def zeros(cls, rows: int, cols: int) -> Matrix:
        return cls([[0] * cols for _ in range(rows)])

    @classmethod
    def column(cls, values: Iterable[Entry]) -> Matrix:
        return cls([[v] for v in values])

    # --------------------------------------------------------------- accessors
    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def rows(self) -> int:
        return self._shape[0]

    @property
    def cols(self) -> int:
        return self._shape[1]

    def is_square(self) -> bool:
        return self.rows == self.cols

    def __getitem__(self, index: tuple[int, int]) -> Rational:
        i, j = index
        return self._rows[i][j]

    def row(self, i: int) -> tuple[Rational, ...]:
        return self._rows[i]

    def tolist(self) -> list[list[Rational]]:
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
        return "\n".join("[" + "  ".join(c.rjust(width) for c in row) + "]" for row in cells)

    # -------------------------------------------------------------- arithmetic
    def _same_shape(self, other: Matrix) -> None:
        if self.shape != other.shape:
            raise MatrixError(f"shape mismatch: {self.shape} vs {other.shape}")

    def __add__(self, other: object) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented
        self._same_shape(other)
        return Matrix(
            [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(self._rows, other._rows)]
        )

    def __sub__(self, other: object) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented
        self._same_shape(other)
        return Matrix(
            [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(self._rows, other._rows)]
        )

    def __neg__(self) -> Matrix:
        return Matrix([[-a for a in row] for row in self._rows])

    def scale(self, factor: Entry) -> Matrix:
        k = as_rational(factor)
        return Matrix([[k * a for a in row] for row in self._rows])

    def __matmul__(self, other: object) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.cols != other.rows:
            raise MatrixError(f"cannot multiply {self.shape} by {other.shape}")
        cols = list(zip(*other._rows))
        return Matrix(
            [[sum((a * b for a, b in zip(row, col)), Rational(0)) for col in cols] for row in self._rows]
        )

    def __mul__(self, other: object) -> Matrix:
        if isinstance(other, Matrix):
            return self.__matmul__(other)
        if isinstance(other, (int, Rational)):
            return self.scale(other)
        return NotImplemented

    def __rmul__(self, other: object) -> Matrix:
        if isinstance(other, (int, Rational)):
            return self.scale(other)
        return NotImplemented

    def __pow__(self, n: int) -> Matrix:
        if not isinstance(n, int):
            return NotImplemented
        if not self.is_square():
            raise MatrixError("only square matrices can be raised to a power")
        base = self if n >= 0 else self.inverse()
        result = Matrix.identity(self.rows)
        for _ in range(abs(n)):
            result = result @ base
        return result

    def transpose(self) -> Matrix:
        return Matrix(zip(*self._rows))

    # ------------------------------------------------------ Gaussian elimination
    def _echelon(self) -> tuple[list[list[Rational]], list[int], int]:
        """Reduced row-echelon form.

        Returns ``(rows, pivot_columns, swaps)``; ``swaps`` counts row swaps so
        the determinant sign can be recovered.
        """
        m = [list(row) for row in self._rows]
        pivots: list[int] = []
        swaps = 0
        r = 0
        for c in range(self.cols):
            pivot = next((i for i in range(r, self.rows) if m[i][c]), None)
            if pivot is None:
                continue
            if pivot != r:
                m[r], m[pivot] = m[pivot], m[r]
                swaps += 1
            lead = m[r][c]
            m[r] = [v / lead for v in m[r]]
            for i in range(self.rows):
                if i != r and m[i][c]:
                    f = m[i][c]
                    m[i] = [a - f * b for a, b in zip(m[i], m[r])]
            pivots.append(c)
            r += 1
            if r == self.rows:
                break
        return m, pivots, swaps

    def rref(self) -> Matrix:
        return Matrix(self._echelon()[0])

    def rank(self) -> int:
        return len(self._echelon()[1])

    def det(self) -> Rational:
        """Determinant by fraction-exact elimination."""
        if not self.is_square():
            raise MatrixError("determinant needs a square matrix")
        m = [list(row) for row in self._rows]
        n = self.rows
        result = Rational(1)
        for c in range(n):
            pivot = next((i for i in range(c, n) if m[i][c]), None)
            if pivot is None:
                return Rational(0)
            if pivot != c:
                m[c], m[pivot] = m[pivot], m[c]
                result = -result
            lead = m[c][c]
            result = result * lead
            for i in range(c + 1, n):
                if m[i][c]:
                    f = m[i][c] / lead
                    m[i] = [a - f * b for a, b in zip(m[i], m[c])]
        return result

    def inverse(self) -> Matrix:
        if not self.is_square():
            raise MatrixError("only square matrices are invertible")
        n = self.rows
        augmented = Matrix(
            [list(row) + [Rational(int(i == j)) for j in range(n)] for i, row in enumerate(self._rows)]
        )
        reduced, pivots, _ = augmented._echelon()
        if pivots[:n] != list(range(n)) or len(pivots) < n:
            raise SingularMatrixError("matrix is singular")
        return Matrix([row[n:] for row in reduced])

    def solve(self, rhs: Matrix | Sequence[Entry]) -> Matrix:
        """Solve ``self @ x == rhs`` for the unique ``x``.

        ``rhs`` may be a Matrix (several right-hand sides as columns) or a
        flat sequence (one right-hand side); the result has matching shape.
        """
        b = rhs if isinstance(rhs, Matrix) else Matrix.column(rhs)
        if b.rows != self.rows:
            raise MatrixError(f"right-hand side has {b.rows} rows, expected {self.rows}")
        n = self.cols
        augmented = Matrix([list(a) + list(r) for a, r in zip(self._rows, b._rows)])
        reduced, pivots, _ = augmented._echelon()
        if any(p >= n for p in pivots):
            raise SingularMatrixError("system is inconsistent")
        if len(pivots) < n:
            raise SingularMatrixError("system has infinitely many solutions")
        return Matrix([reduced[i][n:] for i in range(n)])
