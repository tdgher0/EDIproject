"""Total cost estimation helpers.

This module exposes :func:`total_cost` which aggregates the costs from paint, primer, putty and labour calculations.
"""

from __future__ import annotations

import numpy as np

from .paint import paint_cost
from .primer import primer_cost
from .putty import putty_cost
from .labour import labour_cost

__all__ = ["total_cost"]


def total_cost(
    area_m2: float,
    paint_coats: float,
    paint_coverage_m2_per_litre: float,
    paint_price_per_litre: float,
    primer_coats: float,
    primer_coverage_m2_per_litre: float,
    primer_price_per_litre: float,
    putty_coats: float,
    putty_coverage_m2_per_kg: float,
    putty_price_per_kg: float,
    labour_rate_per_m2: float,
    paint_wastage_percent: float = 0.0,
    primer_wastage_percent: float = 0.0,
    putty_wastage_percent: float = 0.0,
) -> float:
    """Calculate the total cost of a painting project.

    The calculation is the sum of the individual component costs:

    ``total_cost = paint_val + primer_val + putty_val + labour_val``.

    Parameters
    ----------
    area_m2:
        Surface area to paint in square metres.
    paint_coats:
        Number of paint coats.
    paint_coverage_m2_per_litre:
        Paint coverage in m² per litre.
    paint_price_per_litre:
        Paint price per litre.
    paint_wastage_percent:
        Expected wastage for paint as a percentage (default ``0``).
    primer_coats:
        Number of primer coats.
    primer_coverage_m2_per_litre:
        Primer coverage in m² per litre.
    primer_price_per_litre:
        Primer price per litre.
    primer_wastage_percent:
        Expected wastage for primer as a percentage (default ``0``).
    putty_coats:
        Number of putty coats.
    putty_coverage_m2_per_kg:
        Putty coverage in m² per kilogram.
    putty_price_per_kg:
        Putty price per kilogram.
    putty_wastage_percent:
        Expected wastage for putty as a percentage (default ``0``).
    labour_rate_per_m2:
        Labour rate per square metre.

    Returns
    -------
    float
        The aggregate cost of paint, primer, putty and labour.

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

    # paint
    if not isinstance(paint_coats, (int, float, np.integer, np.floating)):
        raise TypeError("paint_coats must be a numeric scalar")
    if not np.isfinite(paint_coats) or paint_coats <= 0:
        raise ValueError("paint_coats must be a positive, finite number")
    if not isinstance(paint_coverage_m2_per_litre, (int, float, np.integer, np.floating)):
        raise TypeError("paint_coverage_m2_per_litre must be a numeric scalar")
    if not np.isfinite(paint_coverage_m2_per_litre) or paint_coverage_m2_per_litre <= 0:
        raise ValueError("paint_coverage_m2_per_litre must be a positive, finite number")
    if not isinstance(paint_price_per_litre, (int, float, np.integer, np.floating)):
        raise TypeError("paint_price_per_litre must be a numeric scalar")
    if not np.isfinite(paint_price_per_litre) or paint_price_per_litre < 0:
        raise ValueError("paint_price_per_litre must be a non-negative, finite number")
    if not isinstance(paint_wastage_percent, (int, float, np.integer, np.floating)):
        raise TypeError("paint_wastage_percent must be a numeric scalar")
    if not np.isfinite(paint_wastage_percent) or paint_wastage_percent < 0:
        raise ValueError("paint_wastage_percent must be a non-negative, finite number")

    # primer
    if not isinstance(primer_coats, (int, float, np.integer, np.floating)):
        raise TypeError("primer_coats must be a numeric scalar")
    if not np.isfinite(primer_coats) or primer_coats <= 0:
        raise ValueError("primer_coats must be a positive, finite number")
    if not isinstance(primer_coverage_m2_per_litre, (int, float, np.integer, np.floating)):
        raise TypeError("primer_coverage_m2_per_litre must be a numeric scalar")
    if not np.isfinite(primer_coverage_m2_per_litre) or primer_coverage_m2_per_litre <= 0:
        raise ValueError("primer_coverage_m2_per_litre must be a positive, finite number")
    if not isinstance(primer_price_per_litre, (int, float, np.integer, np.floating)):
        raise TypeError("primer_price_per_litre must be a numeric scalar")
    if not np.isfinite(primer_price_per_litre) or primer_price_per_litre < 0:
        raise ValueError("primer_price_per_litre must be a non-negative, finite number")
    if not isinstance(primer_wastage_percent, (int, float, np.integer, np.floating)):
        raise TypeError("primer_wastage_percent must be a numeric scalar")
    if not np.isfinite(primer_wastage_percent) or primer_wastage_percent < 0:
        raise ValueError("primer_wastage_percent must be a non-negative, finite number")

    # putty
    if not isinstance(putty_coats, (int, float, np.integer, np.floating)):
        raise TypeError("putty_coats must be a numeric scalar")
    if not np.isfinite(putty_coats) or putty_coats <= 0:
        raise ValueError("putty_coats must be a positive, finite number")
    if not isinstance(putty_coverage_m2_per_kg, (int, float, np.integer, np.floating)):
        raise TypeError("putty_coverage_m2_per_kg must be a numeric scalar")
    if not np.isfinite(putty_coverage_m2_per_kg) or putty_coverage_m2_per_kg <= 0:
        raise ValueError("putty_coverage_m2_per_kg must be a positive, finite number")
    if not isinstance(putty_price_per_kg, (int, float, np.integer, np.floating)):
        raise TypeError("putty_price_per_kg must be a numeric scalar")
    if not np.isfinite(putty_price_per_kg) or putty_price_per_kg < 0:
        raise ValueError("putty_price_per_kg must be a non-negative, finite number")
    if not isinstance(putty_wastage_percent, (int, float, np.integer, np.floating)):
        raise TypeError("putty_wastage_percent must be a numeric scalar")
    if not np.isfinite(putty_wastage_percent) or putty_wastage_percent < 0:
        raise ValueError("putty_wastage_percent must be a non-negative, finite number")

    # labour
    if not isinstance(labour_rate_per_m2, (int, float, np.integer, np.floating)):
        raise TypeError("labour_rate_per_m2 must be a numeric scalar")
    if not np.isfinite(labour_rate_per_m2) or labour_rate_per_m2 < 0:
        raise ValueError("labour_rate_per_m2 must be a non-negative, finite number")

    # Compute individual costs
    paint_val = paint_cost(
        area_m2,
        paint_coats,
        paint_coverage_m2_per_litre,
        paint_price_per_litre,
        paint_wastage_percent,
    )[1]

    primer_val = primer_cost(
        area_m2,
        primer_coats,
        primer_coverage_m2_per_litre,
        primer_price_per_litre,
        primer_wastage_percent,
    )[1]

    putty_val = putty_cost(
        area_m2,
        putty_coats,
        putty_coverage_m2_per_kg,
        putty_price_per_kg,
        putty_wastage_percent,
    )[1]

    labour_val = labour_cost(area_m2, labour_rate_per_m2)

    return paint_val + primer_val + putty_val + labour_val

# End of file
