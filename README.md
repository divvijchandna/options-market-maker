# options-market-maker

Research simulator for making two-sided markets in Deribit BTC options, built on
[pricing-core](https://github.com/divvijchandna/pricing-core). The goal is to quote
around a fair value, lean the quotes against inventory risk, hedge, and attribute
the resulting PnL, all measured on recorded market data with the fill assumptions
stated explicitly.

> Status: early. The pricing and quoting layer below is built and tested; the
> simulation loop, hedging and PnL attribution are not built yet (see Roadmap).

## What is built

| Module | Contents |
|---|---|
| `pricing/inverse.py` | Coin-settled option pricing: price, payoff, premium-adjusted delta and vega, all in BTC |
| `fair_value.py` | Fair value, market mid and edge per option, aligned with a Deribit option chain |
| `quoting/skew.py` | Inventory skew: Avellaneda-Stoikov reservation price, generalised to a book's delta and vega |
| `quoting/quoter.py` | Bid and ask from fair value, skew and a width set in vol, rounded to the tick |

Market data comes from pricing-core's Deribit adapter (public REST API, no keys).

### Coin-settled options

Deribit BTC options have a USD strike but are paid for and settled in BTC, so
prices, Greeks and PnL are in coin:

    V_coin   = Black-76(F, K, T, r = 0, σ) / F
    Δ_pa     = Δ_usd − V_coin          (premium-adjusted delta)
    vega_coin = vega_usd / F

The premium is itself paid in BTC, so a long option already carries V_coin of BTC
exposure and needs that much less hedge: the hedge ratio for a coin-valued book is
the premium-adjusted delta, not the USD delta.

Checked against Deribit on a live snapshot (18 Oct 2026): across all 938 listed BTC
options, this formula at the exchange's mark IV matches its mark price with a median
absolute error of 3e-6 BTC. The delta Deribit displays is the plain USD delta, not
premium-adjusted.

### Quoting

    shift_i  = γ τ (Δ_book Δ_i σ_F² + ν_book ν_i σ_v²)
    centre_i = fair_i − shift_i
    half_i   = max(1 tick, w · vega_i)
    bid_i    = floor((centre_i − half_i) / tick) · tick
    ask_i    = ceil((centre_i + half_i) / tick) · tick

- **Skew from portfolio risk.** The shift is the marginal cost, under a quadratic
  risk penalty, of adding option i to the book: γ times the covariance of its PnL
  with the book's PnL, with futures returns and vol changes as the risk factors.
  For one asset with inventory q it reduces exactly to Avellaneda-Stoikov's
  r = s − q γ σ² τ. A long-delta book lowers its quotes on options that add delta;
  a short-vega book raises every quote, to buy vega back.
- **Width in vol, not price.** The half-spread is set in vol points (e.g. 1 vol
  point) and converted to price with each option's vega, as options desks quote.
  This replaces Avellaneda-Stoikov's fill-intensity term, which needs fill data
  to estimate.
- **Rounding outwards.** Bids round down and asks round up, so tick rounding never
  shows a tighter market than intended. A bid at or below zero is not shown.

## Example

```python
from pricing_core.market.adapters.deribit import fetch_option_chain
from mm.fair_value import fair_values
from mm.pricing.inverse import vega_coin
from mm.quoting.skew import SkewParams, reservation_shift
from mm.quoting.quoter import QuoteParams, make_quotes

chain = fetch_option_chain("BTC")                 # live, public endpoint
fv = fair_values(chain)                           # fair, mid, edge, delta_pa
vega = vega_coin(chain.forward, chain.strike, chain.T, chain.mark_iv)

# Book: long 0.5 BTC of delta, short 2 BTC of vega (per 1.00 of vol).
risk = SkewParams(risk_aversion=2.0, spot_vol=0.5, vol_of_vol=0.8, horizon=1 / 365)
shift = reservation_shift(0.5, -2.0, fv.delta_pa, vega, risk)
bid, ask = make_quotes(fv.fair, vega, shift, QuoteParams(half_width_vol=0.01))
```

On a recorded snapshot from 18 Oct 2026 (pricing-core's Deribit test fixture), three
of the resulting quotes (BTC):

| Option | Fair | Our bid / ask | Market bid / ask |
|---|---|---|---|
| BTC-19OCT26-80000-C | 0.02098 | 0.0203 / 0.0206 | 0.0185 / 0.0235 |
| BTC-25DEC26-100000-P | 0.22204 | 0.2225 / 0.2244 | 0.2185 / 0.2250 |
| BTC-18OCT26-72000-P | 0.00000 | none / 0.0001 | none / 0.0001 |

The 100000 put is quoted entirely above fair: the book is short vega, and the skew
raises the quotes to buy vega back. The deep out-of-the-money put shows no bid.

## Design choices

- **Fair value is Deribit's mark for now.** `fair_values` prices at the exchange's
  mark IV by default, so the edge is measured against the exchange's own view. The
  `vol` argument is the hook for an own fitted surface, which is what would give a
  view that can differ from the exchange's.
- **Everything is vectorised** over the whole option chain, so quoting all listed
  options is one call per step.
- **Assumptions are parameters, not constants**: risk aversion, horizon, vol of
  vol, quote width and tick are all explicit.

## Testing

36 tests. They check the coin price against Deribit marks on recorded data, the
identity Δ_pa,call − Δ_pa,put = K/F, premium-adjusted delta and coin vega against
the slope of the coin price, the reduction of the skew to Avellaneda-Stoikov for one
asset and its signs for delta and vega books, and the quoter's width, tick grid,
minimum width and outward rounding.

```bash
ruff check .
ruff format --check .
pytest
```

## Roadmap

Planned, in order:

1. Deribit recorder (order book snapshots and trades to parquet) and replay.
2. Hard vega and gamma limits on the book.
3. An explicit fill model: a trade printing through our quote fills us, with a
   queue-position haircut.
4. Delta hedging with perpetuals and futures, band versus schedule, with fees and
   funding.
5. PnL attribution: edge captured at fill, inventory PnL, hedging cost, and BTC/USD
   translation.
6. Informed versus uninformed flow, and markouts at 1 s, 10 s and 60 s on real
   Deribit prints.

## Install

Requires Python 3.11+ and a sibling checkout of
[pricing-core](https://github.com/divvijchandna/pricing-core).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "../pricing-core[deribit]"
pip install -e ".[dev]"
```

## Data

Market-data recordings are never committed. Once the recorder exists, a small
sample will be committed under `data/sample/`; the tests use recorded Deribit
responses embedded in the test files.

## References

Avellaneda and Stoikov (2008), "High-frequency trading in a limit order book",
*Quantitative Finance* 8(3); Guéant, *The Financial Mathematics of Market
Liquidity* (2017); Clark, *Foreign Exchange Option Pricing* (2011), ch. 3
(premium-adjusted delta); Deribit API v2 documentation.
