"""Model fair value per option, compared with the market.

    fair  = V_coin = Black-76(F, K, T, r = 0, sigma) / F     (BTC)
    mid   = (bid + ask) / 2, NaN if either side of the book is empty
    edge  = mid - fair: > 0 the market is rich vs the model, < 0 cheap

sigma defaults to Deribit's mark_iv, so for now fair value is Deribit's own mark
(the exchange's view of fair, not an independent one). The `vol` argument is the
hook for an own surface (e.g. per-expiry SVI fitted to bids and offers), which is
what gives a market maker a view that can differ from the exchange's.

delta_pa is the premium-adjusted delta (mm.pricing.inverse): the hedge ratio for a
book whose PnL is counted in the coin.
"""

from dataclasses import dataclass

from numpy.typing import ArrayLike
from pricing_core._types import FloatArray, as_float_array
from pricing_core.market.adapters.deribit import OptionChain

from mm.pricing.inverse import premium_adjusted_delta, price_coin


@dataclass(frozen=True)
class FairValues:
    """Per-option fair value, market mid and edge (BTC), aligned with the chain."""

    instrument: tuple[str, ...]
    fair: FloatArray
    mid: FloatArray
    edge: FloatArray
    delta_pa: FloatArray


def fair_values(chain: OptionChain, vol: ArrayLike | None = None) -> FairValues:
    """Fair value of every option in `chain` at `vol` (default: Deribit mark_iv)."""
    sigma = chain.mark_iv if vol is None else as_float_array(vol)
    args = (chain.forward, chain.strike, chain.T, sigma)
    fair = as_float_array(price_coin(*args, is_call=chain.is_call))
    mid = (chain.bid + chain.ask) / 2.0  # NaN propagates if a side is missing
    return FairValues(
        instrument=chain.instrument,
        fair=fair,
        mid=mid,
        edge=mid - fair,
        delta_pa=as_float_array(premium_adjusted_delta(*args, is_call=chain.is_call)),
    )
