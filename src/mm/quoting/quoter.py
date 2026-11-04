"""Two-sided quotes around a skewed centre, quoted in vol and rounded to the tick.

    centre_i      = fair_i - shift_i                       (shift: quoting.skew)
    half_spread_i = max(min_half_ticks * tick, w * vega_i)
    bid_i         = floor((centre_i - half_spread_i) / tick) * tick
    ask_i         = ceil((centre_i + half_spread_i) / tick) * tick

The width is set in vol (w, e.g. 1 vol point either side) and converted to price
with the option's coin vega, as options desks quote; it replaces Avellaneda-
Stoikov's fill-intensity term, which needs fill data to estimate. Rounding is
outwards (bids down, asks up) so it never tightens the market shown. A bid at or
below zero is not shown (NaN).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from pricing_core._types import FloatArray, as_float_array

_GRID_EPS = 1e-9  # in ticks: keeps 0.0185 / 0.0001 = 184.999... on its grid point


@dataclass(frozen=True)
class QuoteParams:
    half_width_vol: float = 0.01  # w: half-spread in vol (0.01 = 1 vol point)
    tick: float = 0.0001  # price increment, coin
    min_half_ticks: int = 1  # minimum half-spread, in ticks


def make_quotes(
    fair: ArrayLike, vega: ArrayLike, shift: ArrayLike, p: QuoteParams
) -> tuple[FloatArray, FloatArray]:
    """Bid and ask per option (coin), NaN bid where it would be <= 0."""
    centre = as_float_array(fair) - as_float_array(shift)
    half = np.maximum(
        p.min_half_ticks * p.tick, p.half_width_vol * as_float_array(vega)
    )
    bid = np.floor((centre - half) / p.tick + _GRID_EPS) * p.tick
    ask = np.ceil((centre + half) / p.tick - _GRID_EPS) * p.tick
    return np.where(bid > 0, bid, np.nan), ask
