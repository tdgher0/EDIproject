"""Paint cost estimation helpers.

The public function :func:`paint_cost` calculates the amount of paint
required in litres and the corresponding cost based on the supplied
parameters.

The implementation follows the formulas described in the project
specification and includes input validation that raises
:class:`TypeError` or :class:`ValueError` with clear messages.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["paint_cost"]


def paint_cost(
    area_m2: float,
    coats: float,
    coverage_m2_per_litre: float,
    price_per_litre: float,
    wastage_percent: float = 0.0,
) -> Tuple[float, float]:
    """Calculate paint quantity and cost.

    Parameters
    ----------
    area_m2:
        Surface area to paint in square metres.
    coats:
        Number of paint coats.
    coverage_m2_per_litre:
        Paint coverage in m² per litre.
    price_per_litre:
        Paint price per litre in local currency.
    wastage_percent:
        Expected wastage as a percentage (default ``0``).

    Returns
    -------
    quantity_litres, cost
        ``quantity_litres`` - paint required in litres (float).
        ``cost`` - total paint cost in local currency (float).
    """
    # Validation
    if not isinstance(area_m2, (int, float, np.integer, np.floating)):
        raise TypeError("area_m2 must be a numeric scalar")
    if not np.isfinite(area_m2) or area_m2 <= 0:
        raise ValueError("area_m2 must be a positive, finite number")
    if not isinstance(coats, (int, float, np.integer, np.floating)):
        raise TypeError("coats must be a numeric scalar")
    if not np.isfinite(coats) or coats <= 0:
        raise ValueError("coats must be a positive, finite number")
    if not isinstance(coverage_m2_per_litre, (int, float, np.integer, np.floating)):
        raise TypeError("coverage_m2_per_litre must be a numeric scalar")
    if not np.isfinite(coverage_m2_per_litre) or coverage_m2_per_litre <= 0:
        raise ValueError("coverage_m2_per_litre must be a positive, finite number")
    if not isinstance(price_per_litre, (int, float, np.integer, np.floating)):
        raise TypeError("price_per_litre must be a numeric scalar")
    if not np.isfinite(price_per_litre) or price_per_litre < 0:
        raise ValueError("price_per_litre must be a non-negative, finite number")
    if not isinstance(wastage_percent, (int, float, np.integer, np.floating)):
        raise TypeError("wastage_percent must be a numeric scalar")
    if not np.isfinite(wastage_percent) or wastage_percent < 0:
        raise ValueError("wastage_percent must be a non-negative, finite number")

    # Core calculation
    quantity_litres = (area_m2 * coats / coverage_m2_per_litre) * (1 + wastage_percent / 100)
    cost = quantity_litres * price_per_litre

    return quantity_litres, cost

# End of file
