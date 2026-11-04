"""Coin-settled (inverse) options, as listed on Deribit.

A BTC option on 1 BTC has a USD strike but is paid for and settled in BTC.

    Price in coin:   V_coin = V_usd / F,  V_usd = Black-76(F, K, T, r = 0, sigma)
    Payoff in coin:  call max(S_T - K, 0) / S_T,  put max(K - S_T, 0) / S_T
    Delta (coin):    Delta_pa = F * dV_coin/dF = Delta_usd - V_coin
    Vega (coin):     dV_coin/dsigma = vega_usd / F   (F held fixed)

Delta_pa is the premium-adjusted delta: the premium is itself paid in the coin,
so a coin-valued position already carries V_coin of coin exposure and needs that
much less hedge. It is the FX premium-adjusted delta with the coin as the
currency the premium is paid in. Identity (r = 0, C - P = F - K in USD):
Delta_pa,call - Delta_pa,put = K / F.

Deribit conventions, verified on live data (2026-10-18): the mark price equals
V_usd / F at mark_iv with r = 0, F = the expiry's forward (median error 3e-6 BTC
over 938 options); the delta Deribit displays is the plain Delta_usd, not
premium-adjusted.

Reference: Clark, Foreign Exchange Option Pricing (2011), ch. 3 (premium-adjusted
delta); Deribit API v2 documentation (inverse options).
"""

import numpy as np
from numpy.typing import ArrayLike
from pricing_core._types import FloatOrArray, as_float_array, to_output
from pricing_core.analytic.black_scholes import black76_price
from pricing_core.analytic.greeks import delta, vega


def price_coin(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    *,
    is_call: bool | ArrayLike,
) -> FloatOrArray:
    """Option price in coin per 1 coin of notional: Black-76 USD price (r = 0) / F."""
    usd = as_float_array(black76_price(F, K, T, 0.0, sigma, is_call=is_call))
    return to_output(usd / as_float_array(F))


def premium_adjusted_delta(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    *,
    is_call: bool | ArrayLike,
) -> FloatOrArray:
    """Delta_pa = Delta_usd - V_coin, in coin per 1 coin of notional."""
    usd_delta = as_float_array(delta(F, K, T, 0.0, 0.0, sigma, is_call=is_call))
    v_coin = as_float_array(price_coin(F, K, T, sigma, is_call=is_call))
    return to_output(usd_delta - v_coin)


def payoff_coin(
    S_T: ArrayLike, K: ArrayLike, *, is_call: bool | ArrayLike
) -> FloatOrArray:
    """Expiry payoff in coin: the USD payoff converted at the settlement price."""
    s_t, k = as_float_array(S_T), as_float_array(K)
    usd = np.where(is_call, np.maximum(s_t - k, 0.0), np.maximum(k - s_t, 0.0))
    return to_output(usd / s_t)


def vega_coin(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike
) -> FloatOrArray:
    """dV_coin/dsigma = vega_usd / F, in coin per 1.00 of vol (calls and puts)."""
    usd_vega = as_float_array(vega(F, K, T, 0.0, 0.0, sigma))
    return to_output(usd_vega / as_float_array(F))
