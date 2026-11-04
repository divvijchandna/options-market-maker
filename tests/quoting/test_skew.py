"""Tests for mm.quoting.skew: inventory skew of the quote centre."""

import numpy as np
import pytest

from mm.quoting.skew import SkewParams, reservation_shift

P = SkewParams(risk_aversion=2.0, spot_vol=0.5, vol_of_vol=0.8, horizon=1 / 365)


def test_reduces_to_avellaneda_stoikov_for_one_asset():
    # One asset: delta 1 per unit, inventory q, no vega -> shift = q gamma sigma^2 tau.
    q = 3.0
    shift = reservation_shift(book_delta=q, book_vega=0.0, delta=1.0, vega=0.0, p=P)
    assert shift == pytest.approx(q * 2.0 * 0.5**2 * (1 / 365))


def test_flat_book_has_no_skew():
    assert reservation_shift(0.0, 0.0, delta=0.4, vega=0.02, p=P) == 0.0


def test_long_delta_lowers_the_centre_for_positive_delta_options():
    # Long delta: lean to sell options that add delta, i.e. lower both quotes
    # (centre = fair - shift, so a positive shift lowers the centre).
    assert reservation_shift(5.0, 0.0, delta=0.5, vega=0.0, p=P) > 0


def test_short_vega_raises_the_centre():
    # Short vega (sold options): bid higher for any option, buying vega back.
    assert reservation_shift(0.0, -1.0, delta=0.0, vega=0.02, p=P) < 0


def test_vectorised_and_linear_in_risk_aversion():
    deltas = np.array([0.2, 0.5, -0.3])
    vegas = np.array([0.01, 0.03, 0.02])
    s1 = reservation_shift(2.0, -0.5, deltas, vegas, P)
    p2 = SkewParams(risk_aversion=4.0, spot_vol=0.5, vol_of_vol=0.8, horizon=1 / 365)
    s2 = reservation_shift(2.0, -0.5, deltas, vegas, p2)
    assert s1.shape == (3,)
    np.testing.assert_allclose(s2, 2 * s1, rtol=1e-14)


def test_vega_term_size():
    # Pure vega book: shift = gamma * tau * nu_book * nu_i * sigma_v^2.
    shift = reservation_shift(0.0, -2.0, delta=0.0, vega=0.03, p=P)
    assert shift == pytest.approx(2.0 * (1 / 365) * (-2.0) * 0.03 * 0.8**2)
