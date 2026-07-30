from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_risk_index.leverage import LFIConfig, compute_leverage_fragility


def test_leverage_smoke_produces_scores() -> None:
    rows = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for i in range(220):
        ts = start + pd.Timedelta(minutes=i)
        price = 100_000 + i * 2
        for venue, oi_offset, funding in (
            ("okx_perp", 0.0, 0.00005),
            ("bybit_perp", 250_000_000.0, 0.00006),
        ):
            rows.append(
                {
                    "timestamp": ts,
                    "venue": venue,
                    "asset": "BTC",
                    "perp_mid": price,
                    "mark_price": price,
                    "index_price": price - 10,
                    "funding_rate_native": funding,
                    "funding_interval_hours": 8.0,
                    "open_interest_usd": 1_000_000_000 + oi_offset + i * 1_000_000,
                    "volume_usd_24h": 10_000_000_000,
                    "long_liquidation_usd": 0.0,
                    "short_liquidation_usd": 0.0,
                    "data_quality_score": 1.0,
                }
            )
    out = compute_leverage_fragility(
        pd.DataFrame(rows),
        LFIConfig(normalization_lookback=80, normalization_min_history=20),
    )
    assert not out.empty
    assert np.isfinite(out["long_fragility_score_0_100"].dropna()).any()
    assert np.isfinite(out["short_fragility_score_0_100"].dropna()).any()
