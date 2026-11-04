"""Tests for coin-settled (inverse) option pricing, mm.pricing.inverse."""

import numpy as np
import pytest
from pricing_core.analytic.black_scholes import black76_price
from pricing_core.analytic.greeks import delta
from pricing_core.market.adapters.deribit import parse_book_summary

from mm.pricing.inverse import (
    payoff_coin,
    premium_adjusted_delta,
    price_coin,
    vega_coin,
)

# Two rows from a live Deribit get_book_summary_by_currency snapshot (2026-10-18).
DERIBIT_ROWS = {
    "jsonrpc": "2.0",
    "result": [
        {
            "instrument_name": "BTC-25DEC26-100000-P",
            "bid_price": 0.2185,
            "ask_price": 0.225,
            "mark_price": 0.22202042,
            "mark_iv": 36.72,
            "underlying_price": 82469.6,
            "creation_timestamp": 1792304331015,
        },
        {
            "instrument_name": "BTC-25JUN27-100000-C",
            "bid_price": 0.063,
            "ask_price": 0.0645,
            "mark_price": 0.0636774,
            "mark_iv": 37.84,
            "underlying_price": 84671.31,
            "creation_timestamp": 1792304331016,
        },
    ],
}

F, K, T, SIGMA = 80000.0, 90000.0, 0.25, 0.6


def test_price_coin_matches_deribit_mark():
    # Deribit convention: mark (BTC) = Black-76 USD price at mark_iv, r = 0, / F.
    # Tolerance: mark_iv is published to 2 d.p., ~1e-5 BTC of price.
    chain = parse_book_summary(DERIBIT_ROWS)
    model = price_coin(
        chain.forward, chain.strike, chain.T, chain.mark_iv, is_call=chain.is_call
    )
    np.testing.assert_allclose(model, chain.mark, atol=5e-5)


@pytest.mark.parametrize("is_call", [True, False])
def test_price_coin_is_usd_price_over_forward(is_call):
    usd = black76_price(F, K, T, 0.0, SIGMA, is_call=is_call)
    assert price_coin(F, K, T, SIGMA, is_call=is_call) == pytest.approx(usd / F)


@pytest.mark.parametrize("is_call", [True, False])
def test_premium_adjusted_delta_is_usd_delta_minus_coin_price(is_call):
    usd_delta = delta(F, K, T, 0.0, 0.0, SIGMA, is_call=is_call)
    v_coin = price_coin(F, K, T, SIGMA, is_call=is_call)
    assert premium_adjusted_delta(F, K, T, SIGMA, is_call=is_call) == pytest.approx(
        usd_delta - v_coin
    )


@pytest.mark.parametrize("is_call", [True, False])
def test_premium_adjusted_delta_is_f_times_coin_price_slope(is_call):
    # Delta_pa = F * dV_coin / dF: the sensitivity of a coin-valued position.
    h = 1e-4 * F
    slope = (
        price_coin(F + h, K, T, SIGMA, is_call=is_call)
        - price_coin(F - h, K, T, SIGMA, is_call=is_call)
    ) / (2 * h)
    assert premium_adjusted_delta(F, K, T, SIGMA, is_call=is_call) == pytest.approx(
        F * slope, rel=1e-6
    )


def test_call_minus_put_premium_adjusted_delta_is_k_over_f():
    # r = 0: C - P = F - K (USD), so Delta_pa,c - Delta_pa,p = 1 - (F - K)/F = K/F.
    strikes = np.array([60000.0, 80000.0, 100000.0])
    diff = premium_adjusted_delta(F, strikes, T, SIGMA, is_call=True) - (
        premium_adjusted_delta(F, strikes, T, SIGMA, is_call=False)
    )
    np.testing.assert_allclose(diff, strikes / F, rtol=1e-12)


def test_premium_adjustment_reduces_call_delta_and_extends_put_delta():
    assert premium_adjusted_delta(F, K, T, SIGMA, is_call=True) < delta(
        F, K, T, 0.0, 0.0, SIGMA, is_call=True
    )
    assert premium_adjusted_delta(F, K, T, SIGMA, is_call=False) < delta(
        F, K, T, 0.0, 0.0, SIGMA, is_call=False
    )


@pytest.mark.parametrize(
    ("s_t", "is_call", "expected"),
    [
        (120000.0, True, 30000.0 / 120000.0),  # ITM call: (S_T - K) / S_T BTC
        (80000.0, True, 0.0),
        (60000.0, False, 30000.0 / 60000.0),  # ITM put: (K - S_T) / S_T BTC
        (100000.0, False, 0.0),
    ],
)
def test_payoff_coin(s_t, is_call, expected):
    assert payoff_coin(s_t, K, is_call=is_call) == pytest.approx(expected)


def test_scalar_in_float_out():
    for value in (
        price_coin(F, K, T, SIGMA, is_call=True),
        premium_adjusted_delta(F, K, T, SIGMA, is_call=True),
        payoff_coin(F, K, is_call=True),
    ):
        assert type(value) is float


@pytest.mark.parametrize("is_call", [True, False])
def test_vega_coin_is_slope_of_coin_price_in_vol(is_call):
    h = 1e-4
    slope = (
        price_coin(F, K, T, SIGMA + h, is_call=is_call)
        - price_coin(F, K, T, SIGMA - h, is_call=is_call)
    ) / (2 * h)
    assert vega_coin(F, K, T, SIGMA) == pytest.approx(slope, rel=1e-6)
