"""Primer cost estimation helpers.

This module provides a :func:`primer_cost` function that mirrors the calculation
logic of :func:`paint_cost` but operates on primer products.  The public API and
validation rules are identical to :mod:`paint`.

The implementation follows the same formula used for paint:

``quantity_litres = (area_m2 * coats / coverage_m2_per_litre) * (1 + wastage_percent / 100)``

and returns the calculated quantity and cost.
"""

from __future__ import annotations


import numpy as np

__all__ = ["primer_cost"]


def primer_cost(
    area_m2: float,
    coats: float,
    coverage_m2_per_litre: float,
    price_per_litre: float,
    wastage_percent: float = 0.0,
) -> Tuple[float, float]:
    """Calculate primer quantity and cost.

    Parameters
    ----------
    area_m2:
        Surface area to paint in square metres.
    coats:
        Number of paint coats.
    coverage_m2_per_litre:
        Primer coverage in m² per litre.
    price_per_litre:
        Primer price per litre in local currency.
    wastage_percent:
        Expected wastage as a percentage (default ``0``).

    Returns
    -------
    quantity_litres, cost
        ``quantity_litres`` – primer required in litres (float).
        ``cost`` – total primer cost in local currency (float).

    Raises
    ------
    TypeError
        If any argument is not a numeric scalar.
    ValueError
        If numeric values are non-finite or violate positivity/non-negative constraints.
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
    quantity_litres = (
        area_m2 * coats / coverage_m2_per_litre
    ) * (1 + wastage_percent / 100)
    cost = quantity_litres * price_per_litre

    return quantity_litres, cost

# End of file
