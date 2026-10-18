"""Tests for mm.fair_value: model fair value vs the market, per option."""

import numpy as np
import pytest
from pricing_core.market.adapters.deribit import parse_book_summary

from mm.fair_value import FairValues, fair_values
from mm.pricing.inverse import premium_adjusted_delta, price_coin

# Three real rows from the live snapshot of 2026-10-18: both sides quoted,
# bid missing, both sides missing.
CHAIN = parse_book_summary(
    {
        "jsonrpc": "2.0",
        "result": [
            {
                "instrument_name": "BTC-19OCT26-80000-C",
                "bid_price": 0.0185,
                "ask_price": 0.0235,
                "mark_price": 0.02099432,
                "mark_iv": 32.43,
                "underlying_price": 81623.64,
                "creation_timestamp": 1792304331015,
            },
            {
                "instrument_name": "BTC-18OCT26-72000-P",
                "bid_price": None,
                "ask_price": 0.0001,
                "mark_price": 0.0,
                "mark_iv": 63.56,
                "underlying_price": 81612.53,
                "creation_timestamp": 1792304331017,
            },
            {
                "instrument_name": "BTC-19OCT26-88000-C",
                "bid_price": None,
                "ask_price": None,
                "mark_price": 2.29e-06,
                "mark_iv": 41.26,
                "underlying_price": 81623.64,
                "creation_timestamp": 1792304331016,
            },
        ],
    }
)


def test_aligned_with_chain():
    fv = fair_values(CHAIN)
    assert isinstance(fv, FairValues)
    assert fv.instrument == CHAIN.instrument
    for arr in (fv.fair, fv.mid, fv.edge, fv.delta_pa):
        assert arr.shape == (len(CHAIN.instrument),)


def test_fair_at_mark_vol_reproduces_deribit_mark():
    # With Deribit's own mark_iv the model fair value is Deribit's mark
    # (to the 2 d.p. rounding of mark_iv).
    np.testing.assert_allclose(fair_values(CHAIN).fair, CHAIN.mark, atol=5e-5)


def test_mid_and_edge():
    fv = fair_values(CHAIN)
    assert fv.mid[0] == pytest.approx((0.0185 + 0.0235) / 2)
    assert fv.edge[0] == pytest.approx(fv.mid[0] - fv.fair[0])


def test_one_sided_or_empty_book_has_no_mid():
    fv = fair_values(CHAIN)
    assert np.isnan(fv.mid[1]) and np.isnan(fv.edge[1])  # no bid
    assert np.isnan(fv.mid[2]) and np.isnan(fv.edge[2])  # no bid, no ask
    assert np.all(np.isfinite(fv.fair))  # fair value exists regardless


def test_own_vol_overrides_mark_vol():
    # Hook for the future SVI surface: any vol per option can replace mark_iv.
    vol = np.array([0.40, 0.70, 0.50])
    fv = fair_values(CHAIN, vol=vol)
    expected = price_coin(
        CHAIN.forward, CHAIN.strike, CHAIN.T, vol, is_call=CHAIN.is_call
    )
    np.testing.assert_allclose(fv.fair, expected, rtol=1e-14)


def test_delta_is_premium_adjusted():
    fv = fair_values(CHAIN)
    expected = premium_adjusted_delta(
        CHAIN.forward, CHAIN.strike, CHAIN.T, CHAIN.mark_iv, is_call=CHAIN.is_call
    )
    np.testing.assert_allclose(fv.delta_pa, expected, rtol=1e-14)
