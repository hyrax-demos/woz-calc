"""woz: an exact-arithmetic calculator package.

Modules:

* :mod:`woz.rational` -- the exact :class:`Rational` number type.
* :mod:`woz.bigdec` -- round-half-even significant-digit formatting.
* :mod:`woz.functions` -- correctly rounded sqrt/exp/ln/sin/cos/atan/pi.
* :mod:`woz.matrix` -- exact Rational matrices.
* :mod:`woz.lexer`, :mod:`woz.parser`, :mod:`woz.evaluator` -- the expression
  language.
* :mod:`woz.cli` -- the ``python -m woz`` REPL.
"""

from woz.rational import Rational

__all__ = ["Rational"]
