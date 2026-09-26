"""Tests for woz.functions: every function at precisions 1, 10, 30 and 60."""

from __future__ import annotations

import pytest

from woz import functions as F
from woz.bigdec import format_sig, round_sig
from woz.rational import Rational

PRECISIONS = [1, 10, 30, 60]

# 60-significant-digit reference values (correctly rounded).
PI_60 = "3.14159265358979323846264338327950288419716939937510582097494"
E_60 = "2.71828182845904523536028747135266249775724709369995957496697"
SQRT2_60 = "1.41421356237309504880168872420969807856967187537694807317668"
LN2_60 = "0.693147180559945309417232121458176568075500134360255254120680"

# 70-digit references (computed independently) for further arguments.
REFERENCE: dict[tuple[str, str], str] = {
    (
        "sqrt",
        "2",
    ): "1.414213562373095048801688724209698078569671875376948073176679737990732",
    (
        "sqrt",
        "3/7",
    ): "0.6546536707079771437982924562468583555692080823954245575153203034152669",
    (
        "sqrt",
        "1e-9",
    ): "0.00003162277660168379331998893544432718533719555139325216826857504852792594",
    (
        "sqrt",
        "12345",
    ): "111.1080555135405112450044387430752414899113774596977299764856731617826",
    (
        "exp",
        "1",
    ): "2.718281828459045235360287471352662497757247093699959574966967627724077",
    (
        "exp",
        "-5/2",
    ): "0.08208499862389879516952867446715980783780412101543664884575841051522476",
    (
        "exp",
        "1e-20",
    ): "1.000000000000000000010000000000000000000050000000000000000000166666667",
    (
        "exp",
        "50",
    ): "5184705528587072464087.453322933485384827469100583846401904056933806857",
    (
        "ln",
        "2",
    ): "0.6931471805599453094172321214581765680755001343602552541206800094933936",
    (
        "ln",
        "10",
    ): "2.302585092994045684017991454684364207601101488628772976033327900967573",
    (
        "ln",
        "1/3",
    ): "-1.098612288668109691395245236922525704647490557822749451734694333637494",
    (
        "ln",
        "1000001/1000000",
    ): "9.999995000003333330833335333331666668095236845239206348206350115439282e-7",
    (
        "sin",
        "1",
    ): "0.8414709848078965066525023216302989996225630607983710656727517099919104",
    (
        "sin",
        "-7/3",
    ): "-0.7230858817383246167978879286163673263801434704086676930443758555181676",
    (
        "sin",
        "1e-10",
    ): "9.999999999999999999983333333333333333333341666666666666666666664682540e-11",
    (
        "sin",
        "355/113",
    ): "-2.667641890624191484063745288734688868221054266212710251192493875921745e-7",
    (
        "cos",
        "1",
    ): "0.5403023058681397174009366074429766037323104206179222276700972553811004",
    (
        "cos",
        "-7/3",
    ): "-0.6907581397498762927279716947563487870100274864336181898199648430055743",
    (
        "cos",
        "1e-10",
    ): "0.9999999999999999999950000000000000000000041666666666666666666652777778",
    (
        "cos",
        "22/7",
    ): "-0.9999992005335529032683357396565749515758288206610964906755434814480029",
    (
        "atan",
        "1",
    ): "0.7853981633974483096156608458198757210492923498437764552437361480769541",
    (
        "atan",
        "-7/3",
    ): "-1.165904540509813195919248762630308825546698063501877292820041770401155",
    (
        "atan",
        "1e-10",
    ): "9.999999999999999999966666666666666666666866666666666666666665238095238e-11",
    (
        "atan",
        "1000",
    ): "1.569796327128229752564797882004830898086963765133284897396041247966263",
}

FUNCS = {
    "sqrt": F.sqrt,
    "exp": F.exp,
    "ln": F.ln,
    "sin": F.sin,
    "cos": F.cos,
    "atan": F.atan,
}


def _expected(ref: str, digits: int) -> str:
    # The references carry >= 69 digits, so rounding them to <= 60 digits
    # is exact unless the dropped tail is a tie, which never happens here.
    return format_sig(Rational.from_str(ref), digits)


@pytest.mark.parametrize("digits", PRECISIONS)
@pytest.mark.parametrize("key", sorted(REFERENCE))
def test_reference_values(key: tuple[str, str], digits: int) -> None:
    name, arg = key
    got = FUNCS[name](Rational.from_str(arg), digits)
    assert format_sig(got, digits) == _expected(REFERENCE[key], digits)


@pytest.mark.parametrize("digits", PRECISIONS)
def test_pi(digits: int) -> None:
    assert format_sig(F.pi(digits), digits) == format_sig(
        Rational.from_str(PI_60), digits
    )


def test_pi_60_exact_string() -> None:
    assert format_sig(F.pi(60), 60) == PI_60


def test_e_60_exact_string() -> None:
    assert format_sig(F.exp(1, 60), 60) == E_60


def test_sqrt2_60_exact_string() -> None:
    assert format_sig(F.sqrt(2, 60), 60) == SQRT2_60


def test_ln2_60_exact_string() -> None:
    assert format_sig(F.ln(2, 60), 60) == LN2_60


@pytest.mark.parametrize("digits", PRECISIONS)
def test_exact_special_values(digits: int) -> None:
    assert F.sqrt(Rational(9, 4), digits) == round_sig(Rational(3, 2), digits)
    assert F.sqrt(0, digits) == 0
    assert F.exp(0, digits) == 1
    assert F.ln(1, digits) == 0
    assert F.sin(0, digits) == 0
    assert F.cos(0, digits) == 1
    assert F.atan(0, digits) == 0


@pytest.mark.parametrize("digits", PRECISIONS)
def test_identities(digits: int) -> None:
    # atan(1) * 4 == pi, compared at the precision requested.
    four_atan = format_sig(4 * F.atan(1, 60), digits)
    assert four_atan == format_sig(F.pi(60), digits)
    # sin and cos of pi/2 approximations
    assert abs(F.cos(F.pi(60) / 2, digits)) < Rational(1, 10**58)


@pytest.mark.parametrize("digits", [0, 61, -3])
@pytest.mark.parametrize("name", sorted(FUNCS))
def test_precision_out_of_range(name: str, digits: int) -> None:
    with pytest.raises(ValueError):
        FUNCS[name](Rational(1, 2), digits)


def test_pi_precision_out_of_range() -> None:
    with pytest.raises(ValueError):
        F.pi(61)


@pytest.mark.parametrize("fn, arg", [(F.sqrt, -1), (F.ln, 0), (F.ln, -2)])
def test_domain_errors(fn: object, arg: int) -> None:
    with pytest.raises(F.DomainError):
        fn(arg, 10)  # type: ignore[operator]


def test_overflow_guards() -> None:
    with pytest.raises(OverflowError):
        F.exp(10**6, 10)
    with pytest.raises(OverflowError):
        F.sin(10**40, 10)


def test_type_errors() -> None:
    with pytest.raises(TypeError):
        F.sqrt(2.0, 10)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        F.exp(True, 10)
    with pytest.raises(TypeError):
        F.pi(10.0)  # type: ignore[arg-type]


def test_large_trig_argument() -> None:
    got = F.sin(Rational(10**20), 30)
    assert format_sig(got, 30) == "-0.645251285265780844205811711313"
