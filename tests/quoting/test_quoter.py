"""Tests for mm.quoting.quoter: two-sided quotes around a centre, on the tick grid."""

import numpy as np
import pytest

from mm.quoting.quoter import QuoteParams, make_quotes

P = QuoteParams(half_width_vol=0.01, tick=0.0001, min_half_ticks=1)
FAIR = np.array([0.0210, 0.1500, 0.00030])
VEGA = np.array([0.0300, 0.0800, 0.00050])  # coin vega per 1.00 of vol


def _on_grid(x, tick):
    k = x / tick
    return np.all(np.abs(k - np.round(k)) < 1e-9)


def test_quotes_bracket_the_centre_on_the_tick_grid():
    bid, ask = make_quotes(FAIR, VEGA, shift=0.0, p=P)
    assert np.all(bid <= FAIR) and np.all(ask >= FAIR)
    assert _on_grid(bid, P.tick) and _on_grid(ask, P.tick)


def test_width_is_vol_width_times_vega():
    # Half-spread = 1 vol point x vega; rounding outwards adds at most 1 tick a side.
    bid, ask = make_quotes(FAIR, VEGA, shift=0.0, p=P)
    width = ask - bid
    target = 2 * P.half_width_vol * VEGA
    assert np.all(width >= target - 1e-12)
    assert np.all(width <= np.maximum(target, 2 * P.tick) + 2 * P.tick + 1e-12)


def test_minimum_width_when_vega_is_tiny():
    bid, ask = make_quotes(np.array([0.05]), np.array([1e-6]), shift=0.0, p=P)
    assert ask[0] - bid[0] == pytest.approx(2 * P.tick)


def test_positive_shift_lowers_both_quotes():
    b0, a0 = make_quotes(FAIR, VEGA, shift=0.0, p=P)
    b1, a1 = make_quotes(FAIR, VEGA, shift=0.0005, p=P)
    # A bid pushed to or below zero disappears (NaN); that also counts as lower.
    assert np.all((b1 <= b0) | np.isnan(b1)) and np.all(a1 <= a0)
    assert np.any(b1 < b0)


def test_no_bid_at_or_below_zero():
    # Deep OTM: centre minus half-spread <= 0, so no bid; the offer still shows.
    bid, ask = make_quotes(np.array([0.00005]), np.array([0.002]), shift=0.0, p=P)
    assert np.isnan(bid[0])
    assert ask[0] > 0


def test_rounding_never_tightens():
    # A centre exactly on the grid still gets at least the minimum width.
    bid, ask = make_quotes(np.array([0.0200]), np.array([0.0]), shift=0.0, p=P)
    assert bid[0] == pytest.approx(0.0199) and ask[0] == pytest.approx(0.0201)


def test_rounding_is_outwards_not_nearest():
    # Half-spread 1 vol point x vega 0.014 = 0.00014 around 0.0200: raw quotes
    # 0.01986 / 0.02014. Outwards: 0.0198 / 0.0202. Nearest would tighten to
    # 0.0199 / 0.0201.
    bid, ask = make_quotes(np.array([0.0200]), np.array([0.014]), shift=0.0, p=P)
    assert bid[0] == pytest.approx(0.0198) and ask[0] == pytest.approx(0.0202)
