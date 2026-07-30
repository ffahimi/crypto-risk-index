# crypto-risk-index

A few indices to measure momentary and mid-term tail risk on BTC.

This package contains pure pandas/numpy implementations of the recent BTC risk indices:

- Tail Fragility Index (TFI)
- Order-Flow Toxicity / VPIN proxy (OTI)
- Directional Jump Risk (JRI-style composition)
- Leverage Fragility Index (LFI)

The core package does not require Postgres, ZMQ, Telegram, Polymarket, or any production server paths. Postgres is an optional debug input adapter.

## Install

```bash
pip install -e ".[postgres,dev]"
```

For metric-only use:

```bash
pip install -e .
```

## Public API

```python
from crypto_risk_index.tail import compute_surface_features, PipelineConfig
from crypto_risk_index.flow import compute_oti_proxy, OTIProxyConfig
from crypto_risk_index.jump import compute_directional_jump_risk, JumpRiskConfig
from crypto_risk_index.leverage import compute_leverage_fragility, LFIConfig
```

All compute functions accept `pandas.DataFrame` inputs and return `pandas.DataFrame` outputs.

## Debug Run From Postgres

The first debug mode reads `derivatives_venue_metrics`, computes leverage fragility, prints rows, and inserts nothing.

```bash
export PM_DB_DSN='postgresql://...'

crypto-risk-index leverage \
  --dsn "$PM_DB_DSN" \
  --asset BTC \
  --lookback-minutes 1800 \
  --debug \
  --tail-rows 8
```

Equivalent module call:

```bash
python -m crypto_risk_index.cli.main leverage \
  --dsn "$PM_DB_DSN" \
  --asset BTC \
  --debug
```

## Required Data Shapes

### Leverage Fragility

Required columns:

```text
timestamp, venue, asset, perp_mid, mark_price, index_price,
funding_rate_native, funding_interval_hours,
open_interest_usd, volume_usd_24h
```

Optional columns:

```text
spot_reference_price, predicted_funding_rate, volume_usd_1h,
long_liquidation_usd, short_liquidation_usd, data_quality_score
```

### Order-Flow Toxicity Proxy

Expected venue columns where available:

```text
timestamp
coinbase_mid, coinbase_bid, coinbase_ask
okx_mid, okx_bid, okx_ask
bybit_mid, bybit_bid, bybit_ask
```

### Tail Fragility

Expected canonical raw columns:

```text
timestamp
bybit_mid, bybit_age_ms
okx_mid, okx_age_ms
coinbase_mid, coinbase_age_ms
```

## Notes

These indices are diagnostics, not direct trade recommendations. They are best used as state variables for market stress, direction bias, and volatility risk.
