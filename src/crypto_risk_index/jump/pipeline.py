from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class JumpRiskConfig:
    """Directional jump-risk composition.

    The input may be a merged tail/flow table or separate tail and flow tables.
    Scores are 0-100. Downside combines downside tail structure with sell
    toxicity; upside combines upside/resilient structure with buy toxicity.
    """

    tail_weight: float = 0.45
    flow_weight: float = 0.45
    interaction_weight: float = 0.10
    clean_spread: float = 10.0


def finite(value: Any) -> bool:
    try:
        return np.isfinite(float(value))
    except Exception:
        return False


def _pct01(series: pd.Series) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce") / 100.0).clip(0.0, 1.0)


def _col(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in df:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def jump_state(prefix: str, score: Any) -> str:
    if not finite(score):
        return f"{prefix}_UNKNOWN"
    s = float(score)
    if s >= 97.5:
        regime = "EXTREME"
    elif s >= 90:
        regime = "SEVERE"
    elif s >= 75:
        regime = "HIGH"
    elif s >= 50:
        regime = "ELEVATED"
    else:
        regime = "NORMAL"
    return f"{prefix}_{regime}"


def compute_directional_jump_risk(
    tail: pd.DataFrame,
    flow: pd.DataFrame | None = None,
    *,
    cfg: JumpRiskConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or JumpRiskConfig()
    if tail.empty:
        return pd.DataFrame()

    left = tail.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
    if flow is not None and not flow.empty:
        right = flow.copy()
        right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True)
        keep = [
            c
            for c in (
                "timestamp",
                "oti_sell_percentile",
                "oti_buy_percentile",
                "oti_unsigned_percentile",
                "vpin_proxy_50",
                "sell_tox_proxy_50",
                "buy_tox_proxy_50",
                "perp_dispersion_bps",
            )
            if c in right
        ]
        df = pd.merge_asof(
            left.sort_values("timestamp"),
            right[keep].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
    else:
        df = left.sort_values("timestamp").copy()

    out = pd.DataFrame({"timestamp": df["timestamp"]})
    for name in ("asset", "canonical_perp_price", "source_count"):
        if name in df:
            out[name] = df[name]

    # Tail structure: TFI percentile is downside stress. For upside, use the
    # inverse of downside asymmetry when available plus buy toxicity.
    downside_tail_raw = _col(df, "tfi_percentile_rate_only")
    if downside_tail_raw.isna().all() and "tfi_percentile" in df:
        downside_tail_raw = _col(df, "tfi_percentile")
    downside_tail = _pct01(downside_tail_raw)
    asym = _col(df, "lf_asymmetry_3_24h")
    upside_tail_raw = pd.Series(np.nan, index=df.index, dtype=float)
    if asym.notna().any():
        rank = asym.rolling(501, min_periods=50).apply(
            lambda arr: 100.0 * np.mean(arr[:-1][np.isfinite(arr[:-1])] <= arr[-1])
            if np.isfinite(arr[-1]) and np.count_nonzero(np.isfinite(arr[:-1])) >= 50
            else np.nan,
            raw=True,
        )
        # Negative asymmetry was logged as upside-cheaper in the original tool,
        # so high upside tail structure is low asymmetry rank.
        upside_tail_raw = 100.0 - rank
    upside_tail = _pct01(upside_tail_raw)

    sell_flow = _pct01(_col(df, "oti_sell_percentile"))
    buy_flow = _pct01(_col(df, "oti_buy_percentile"))
    if sell_flow.isna().all() and "sell_tox_proxy_50" in df:
        sell_flow = pd.to_numeric(df["sell_tox_proxy_50"], errors="coerce").clip(0, 1)
    if buy_flow.isna().all() and "buy_tox_proxy_50" in df:
        buy_flow = pd.to_numeric(df["buy_tox_proxy_50"], errors="coerce").clip(0, 1)

    down = 100.0 * (
        cfg.tail_weight * downside_tail
        + cfg.flow_weight * sell_flow
        + cfg.interaction_weight * downside_tail * sell_flow
    )
    up = 100.0 * (
        cfg.tail_weight * upside_tail
        + cfg.flow_weight * buy_flow
        + cfg.interaction_weight * upside_tail * buy_flow
    )

    out["jump_risk_downside_score_0_100"] = down
    out["jump_risk_upside_score_0_100"] = up
    out["dominant_side"] = np.where(down >= up, "downside", "upside")
    out["dominant_score_0_100"] = np.maximum(down, up)
    out["opposite_score_0_100"] = np.minimum(down, up)
    out["dominance_spread"] = out["dominant_score_0_100"] - out["opposite_score_0_100"]
    out["dominance_ratio"] = out["dominant_score_0_100"] / out["opposite_score_0_100"].clip(lower=1e-9)
    out["clean_side"] = out["dominance_spread"] >= cfg.clean_spread
    out["downside_state"] = [jump_state("DOWNSIDE", x) for x in out["jump_risk_downside_score_0_100"]]
    out["upside_state"] = [jump_state("UPSIDE", x) for x in out["jump_risk_upside_score_0_100"]]
    out["calculation_version"] = "directional_jump_risk_v0"
    return out
