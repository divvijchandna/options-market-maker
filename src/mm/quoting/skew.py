"""Inventory skew: shift the quote centre to lean against the book's risk.

Avellaneda and Stoikov (2008), one asset with inventory q:
    reservation price r = s - q * gamma * sigma^2 * tau

Generalised to an option book through its Greeks. Over the horizon tau, option i
has coin PnL ~ Delta_i * x + nu_i * d_sigma, with x = dF / F (futures return,
annual vol sigma_F) and d_sigma the change in implied vol (annual vol of vol
sigma_v), taken as independent. With a quadratic penalty gamma/2 * Var(book PnL),
the marginal risk cost of trading one more unit of option i is
gamma * Cov(book PnL, option i PnL):

    shift_i = gamma * tau * (Delta_book * Delta_i * sigma_F^2
                             + nu_book * nu_i * sigma_v^2)
    centre_i = fair_i - shift_i

One asset (Delta_i = 1, Delta_book = q, no vega) gives back Avellaneda-Stoikov.
A long-delta book lowers the centre of options that add delta (sell them more
readily); a short-vega book raises every centre (buy vega back).

Units: Delta in coin (premium-adjusted), vega in coin per 1.00 of vol, gamma in
1/coin, so the shift is in coin.

Reference: Avellaneda and Stoikov (2008), "High-frequency trading in a limit
order book", Quantitative Finance 8(3); Gueant (2017), The Financial Mathematics
of Market Liquidity, ch. on multi-asset market making.
"""

from dataclasses import dataclass

from numpy.typing import ArrayLike
from pricing_core._types import FloatOrArray, as_float_array, to_output


@dataclass(frozen=True)
class SkewParams:
    risk_aversion: float  # gamma, per coin
    spot_vol: float  # sigma_F, annualised vol of futures returns (e.g. 0.5)
    vol_of_vol: float  # sigma_v, annualised std of implied-vol changes (e.g. 0.8)
    horizon: float  # tau, years until inventory is expected to be worked off


def reservation_shift(
    book_delta: ArrayLike,
    book_vega: ArrayLike,
    delta: ArrayLike,
    vega: ArrayLike,
    p: SkewParams,
) -> FloatOrArray:
    """Amount to subtract from fair value for each option's quote centre (coin)."""
    risk = as_float_array(book_delta) * as_float_array(delta) * p.spot_vol**2 + (
        as_float_array(book_vega) * as_float_array(vega) * p.vol_of_vol**2
    )
    return to_output(p.risk_aversion * p.horizon * risk)
