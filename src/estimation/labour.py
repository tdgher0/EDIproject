"""Labour cost estimation helpers.

This module provides a :func:`labour_cost` function that calculates the cost of
labour for a given area and rate.
"""

from __future__ import annotations

import numpy as np

__all__ = ["labour_cost"]


def labour_cost(area_m2: float, rate_per_m2: float) -> float:
    """Calculate the labour cost.

    Parameters
    ----------
    area_m2:
        Surface area in square metres.
    rate_per_m2:
        Labour rate per square metre in local currency.

    Returns
    -------
    float
        The total labour cost.

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
    if not isinstance(rate_per_m2, (int, float, np.integer, np.floating)):
        raise TypeError("rate_per_m2 must be a numeric scalar")
    if not np.isfinite(rate_per_m2) or rate_per_m2 < 0:
        raise ValueError("rate_per_m2 must be a non-negative, finite number")

    return area_m2 * rate_per_m2

# End of file
