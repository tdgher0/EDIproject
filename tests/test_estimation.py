import pytest

from src.estimation.labour import labour_cost
from src.estimation.paint import paint_cost
from src.estimation.primer import primer_cost
from src.estimation.putty import putty_cost
from src.estimation.total_cost import total_cost


def test_paint_cost():
    quantity, cost = paint_cost(100, 2, 10, 30, 5)
    assert quantity == pytest.approx(21.0)
    assert cost == pytest.approx(630.0)


def test_primer_cost():
    quantity, cost = primer_cost(100, 1, 10, 20, 5)
    assert quantity == pytest.approx(10.5)
    assert cost == pytest.approx(210.0)


def test_putty_cost():
    quantity, cost = putty_cost(100, 1, 4, 25, 5)
    assert quantity == pytest.approx(26.25)
    assert cost == pytest.approx(656.25)


def test_labour_cost():
    assert labour_cost(100, 20) == pytest.approx(2000.0)


def test_total_cost():
    result = total_cost(
        100,
        2, 10, 30,
        1, 10, 20,
        1, 4, 25,
        20,
        5, 5, 5,
    )
    assert result == pytest.approx(3496.25)


@pytest.mark.parametrize(
    "function,args",
    [
        (paint_cost, (0, 2, 10, 30)),
        (primer_cost, (0, 1, 10, 20)),
        (putty_cost, (0, 1, 4, 25)),
        (labour_cost, (0, 20)),
    ],
)
def test_estimation_rejects_zero_area(function, args):
    with pytest.raises(ValueError):
        function(*args)


def test_total_cost_matches_component_costs():
    area = 100

    paint = paint_cost(100, 2, 10, 30, 5)[1]
    primer = primer_cost(100, 1, 10, 20, 5)[1]
    putty = putty_cost(100, 1, 4, 25, 5)[1]
    labour = labour_cost(100, 20)

    total = total_cost(
        area,
        2, 10, 30,
        1, 10, 20,
        1, 4, 25,
        20,
        5, 5, 5,
    )

    assert total == pytest.approx(paint + primer + putty + labour)
