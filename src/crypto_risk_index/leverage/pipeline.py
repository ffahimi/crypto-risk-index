from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

import warnings

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in log", category=RuntimeWarning)


CALC_VERSION = "leverage_fragility_v0"
EPS = 1e-12


@dataclass(frozen=True)
class LFIConfig:
    asset: str = "BTC"
    calculation_version: str = CALC_VERSION
    normalization_lookback: int = 500
    normalization_min_history: int = 50
    z_clip: float = 4.0
    crowding_threshold_pct: float = 75.0
    trigger_threshold_pct: float = 75.0
    agreement_threshold_pct: float = 75.0
    funding_interval_default_h: float = 8.0


REQUIRED_CANONICAL_COLUMNS = (
    "timestamp",
    "venue",
    "asset",
    "perp_mid",
    "mark_price",
    "index_price",
    "funding_rate_native",
    "funding_interval_hours",
    "open_interest_usd",
    "volume_usd_24h",
)


def finite(v: Any) -> bool:
    try:
        return np.isfinite(float(v))
    except Exception:
        return False


def regime_from_percentile(pct: Any) -> str | None:
    if not finite(pct):
        return None
    p = float(pct)
    if p < 50:
        return "NORMAL"
    if p < 75:
        return "ELEVATED"
    if p < 90:
        return "FRAGILE"
    if p < 97.5:
        return "SEVERE"
    return "EXTREME"


def robust_trailing(series: pd.Series, lookback: int, min_history: int, clip: float) -> pd.DataFrame:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full((len(vals), 6), np.nan, dtype=float)

    for i, cur in enumerate(vals):
        start = max(0, i - lookback)
        hist = vals[start:i]
        hist = hist[np.isfinite(hist)]
        out[i, 4] = float(hist.size)
        if not np.isfinite(cur) or hist.size < min_history:
            continue
        med = float(np.median(hist))
        mad = float(np.median(np.abs(hist - med)))
        scale = max(1.4826 * mad, EPS)
        raw_z = (float(cur) - med) / scale
        out[i, 0] = raw_z
        out[i, 1] = float(np.clip(raw_z, -clip, clip))
        out[i, 2] = med
        out[i, 3] = scale
        out[i, 5] = 100.0 * float(np.count_nonzero(hist <= cur)) / float(hist.size)

    return pd.DataFrame(
        out,
        columns=("raw_z", "clipped_z", "median", "mad", "count", "percentile"),
        index=series.index,
    )


def percentile_trailing(series: pd.Series, lookback: int, min_history: int) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")

    def calc(arr: np.ndarray) -> float:
        cur = arr[-1]
        hist = arr[:-1]
        hist = hist[np.isfinite(hist)]
        if not np.isfinite(cur) or hist.size < min_history:
            return np.nan
        return 100.0 * float(np.count_nonzero(hist <= cur)) / float(hist.size)

    return vals.rolling(lookback + 1, min_periods=min_history + 1).apply(calc, raw=True)


def lag_by_time(df: pd.DataFrame, col: str, minutes: int) -> pd.Series:
    if df.empty or col not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    idx = pd.DatetimeIndex(df["timestamp"])
    targets = idx - pd.Timedelta(minutes=minutes)
    positions = idx.get_indexer(targets, method="ffill")
    out = np.full(len(df), np.nan, dtype=float)
    valid = positions >= 0
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    out[valid] = arr[positions[valid]]
    return pd.Series(out, index=df.index)


def add_time_changes(df: pd.DataFrame, col: str, prefix: str, horizons: tuple[int, ...]) -> None:
    for h in horizons:
        lag = lag_by_time(df, col, h)
        cur = pd.to_numeric(df[col], errors="coerce")
        df[f"{prefix}_change_{h}m"] = cur - lag
        df[f"{prefix}_pct_change_{h}m"] = df[f"{prefix}_change_{h}m"] / (lag.abs() + EPS)
        df[f"{prefix}_log_change_{h}m"] = np.where((cur > 0) & (lag > 0), np.log((cur + EPS) / (lag + EPS)), np.nan)


def weighted_mean(df: pd.DataFrame, value: str, weight: str) -> float:
    vals = pd.to_numeric(df[value], errors="coerce")
    weights = pd.to_numeric(df[weight], errors="coerce")
    mask = vals.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float((vals[mask] * weights[mask]).sum() / (weights[mask].sum() + EPS))


def weighted_dispersion(df: pd.DataFrame, value: str, weight: str) -> float:
    vals = pd.to_numeric(df[value], errors="coerce")
    weights = pd.to_numeric(df[weight], errors="coerce")
    mask = vals.notna() & weights.notna() & (weights > 0)
    if mask.sum() < 2:
        return np.nan
    mean = float((vals[mask] * weights[mask]).sum() / (weights[mask].sum() + EPS))
    return float(np.sqrt((weights[mask] * (vals[mask] - mean) ** 2).sum() / (weights[mask].sum() + EPS)))


def _state_label(side: str, crowd_pct: Any, trigger_pct: Any, cfg: LFIConfig) -> str:
    crowded = finite(crowd_pct) and float(crowd_pct) >= cfg.crowding_threshold_pct
    triggered = finite(trigger_pct) and float(trigger_pct) >= cfg.trigger_threshold_pct
    if side == "long":
        if crowded and triggered:
            return "LONG_CROWDED_AND_UNWINDING"
        if crowded:
            return "LONG_CROWDED"
        if triggered:
            return "LONG_UNWINDING"
        return "LONG_NORMAL"
    if crowded and triggered:
        return "SHORT_CROWDED_AND_SQUEEZING"
    if crowded:
        return "SHORT_CROWDED"
    if triggered:
        return "SHORT_SQUEEZING"
    return "SHORT_NORMAL"


def _state_code(state: str | None) -> int | None:
    if state is None:
        return None
    if state.endswith("_CROWDED_AND_UNWINDING") or state.endswith("_CROWDED_AND_SQUEEZING"):
        return 3
    if state.endswith("_UNWINDING") or state.endswith("_SQUEEZING"):
        return 2
    if state.endswith("_CROWDED"):
        return 1
    if state.endswith("_NORMAL"):
        return 0
    return None


def _price_oi_state(ret: Any, oi_change: Any) -> str | None:
    if not finite(ret) or not finite(oi_change):
        return None
    if float(ret) >= 0 and float(oi_change) >= 0:
        return "price_up_oi_up_new_leverage"
    if float(ret) < 0 and float(oi_change) >= 0:
        return "price_down_oi_up_short_build"
    if float(ret) >= 0 and float(oi_change) < 0:
        return "price_up_oi_down_short_covering"
    return "price_down_oi_down_long_deleveraging"


def _prepare_venue_rows(df: pd.DataFrame, cfg: LFIConfig) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["asset"] = out["asset"].astype(str).str.upper()
    out = out[out["asset"] == cfg.asset.upper()].copy()
    for col in (
        "perp_mid",
        "mark_price",
        "index_price",
        "funding_rate_native",
        "funding_interval_hours",
        "predicted_funding_rate",
        "open_interest_usd",
        "volume_usd_1h",
        "volume_usd_24h",
        "long_liquidation_usd",
        "short_liquidation_usd",
        "data_quality_score",
    ):
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["funding_interval_hours"] = out["funding_interval_hours"].fillna(cfg.funding_interval_default_h)
    out["funding_rate_8h_equivalent"] = out["funding_rate_native"] * (8.0 / out["funding_interval_hours"])
    out["funding_rate_annualized"] = out["funding_rate_native"] * (24.0 * 365.0 / out["funding_interval_hours"])
    out["spot_reference_price"] = pd.to_numeric(out.get("spot_reference_price", np.nan), errors="coerce")
    out["basis_reference"] = out.get("basis_reference", "venue_index")
    out["spot_reference_price"] = out["spot_reference_price"].fillna(out["index_price"])
    out["basis_raw"] = (out["perp_mid"] - out["spot_reference_price"]) / (out["spot_reference_price"] + EPS)
    out["basis_bps"] = 10000.0 * out["basis_raw"]
    out["data_quality_score"] = out["data_quality_score"].fillna(1.0)
    out["weight"] = out["open_interest_usd"] * out["data_quality_score"]
    out.loc[~np.isfinite(out["weight"]) | (out["weight"] <= 0), "weight"] = np.nan
    return out.sort_values(["venue", "timestamp"]).reset_index(drop=True)


def _add_venue_history(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("venue", sort=False):
        g = g.sort_values("timestamp").copy()
        g["price_return_15m"] = np.log((g["perp_mid"] + EPS) / (lag_by_time(g, "perp_mid", 15) + EPS))
        g["price_return_1h"] = np.log((g["perp_mid"] + EPS) / (lag_by_time(g, "perp_mid", 60) + EPS))
        add_time_changes(g, "open_interest_usd", "open_interest", (15, 60, 240, 1440))
        add_time_changes(g, "basis_bps", "basis", (15, 60, 240, 1440))
        g["oi_to_volume_1h"] = g["open_interest_usd"] / (g["volume_usd_1h"] + EPS)
        g["oi_to_volume_24h"] = g["open_interest_usd"] / (g["volume_usd_24h"] + EPS)
        g["price_oi_state_15m"] = [
            _price_oi_state(r, o)
            for r, o in zip(g["price_return_15m"], g["open_interest_pct_change_15m"], strict=False)
        ]
        g["price_oi_state_1h"] = [
            _price_oi_state(r, o)
            for r, o in zip(g["price_return_1h"], g["open_interest_pct_change_60m"], strict=False)
        ]
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else df


def _build_composite(venue_df: pd.DataFrame, cfg: LFIConfig) -> pd.DataFrame:
    rows = []
    for ts, g in venue_df.groupby("timestamp", sort=True):
        row: dict[str, Any] = {
            "asset": cfg.asset.upper(),
            "timestamp": ts,
            "calculation_version": cfg.calculation_version,
            "source_count": int(g["venue"].nunique()),
            "data_quality_score": float(pd.to_numeric(g["data_quality_score"], errors="coerce").mean()),
            "venues_available": ",".join(sorted(g["venue"].dropna().astype(str).unique())),
        }
        for venue in ("okx_perp", "bybit_perp"):
            vg = g[g["venue"] == venue]
            if not vg.empty:
                last = vg.iloc[-1]
                for col in (
                    "perp_mid",
                    "mark_price",
                    "index_price",
                    "funding_rate_8h_equivalent",
                    "funding_rate_annualized",
                    "basis_bps",
                    "open_interest_usd",
                    "open_interest_pct_change_60m",
                    "oi_to_volume_24h",
                    "long_liquidation_usd",
                    "short_liquidation_usd",
                ):
                    row[f"{venue}_{col}"] = last.get(col)
        row["canonical_perp_price"] = weighted_mean(g, "perp_mid", "weight")
        if not finite(row["canonical_perp_price"]):
            row["canonical_perp_price"] = float(pd.to_numeric(g["perp_mid"], errors="coerce").mean())
        row["funding_composite_8h"] = weighted_mean(g, "funding_rate_8h_equivalent", "weight")
        row["funding_composite_annualized"] = weighted_mean(g, "funding_rate_annualized", "weight")
        row["basis_composite_bps"] = weighted_mean(g, "basis_bps", "weight")
        row["open_interest_composite_usd"] = float(pd.to_numeric(g["open_interest_usd"], errors="coerce").sum(min_count=1))
        row["volume_usd_1h_composite"] = float(pd.to_numeric(g["volume_usd_1h"], errors="coerce").sum(min_count=1))
        row["volume_usd_24h_composite"] = float(pd.to_numeric(g["volume_usd_24h"], errors="coerce").sum(min_count=1))
        row["oi_to_volume_1h"] = row["open_interest_composite_usd"] / (row["volume_usd_1h_composite"] + EPS) if finite(row["volume_usd_1h_composite"]) else np.nan
        row["oi_to_volume_24h"] = row["open_interest_composite_usd"] / (row["volume_usd_24h_composite"] + EPS) if finite(row["volume_usd_24h_composite"]) else np.nan
        row["long_liquidation_usd_1h"] = float(pd.to_numeric(g["long_liquidation_usd"], errors="coerce").sum(min_count=1))
        row["short_liquidation_usd_1h"] = float(pd.to_numeric(g["short_liquidation_usd"], errors="coerce").sum(min_count=1))
        row["funding_cross_venue_dispersion"] = weighted_dispersion(g, "funding_rate_8h_equivalent", "weight")
        row["basis_cross_venue_dispersion"] = weighted_dispersion(g, "basis_bps", "weight")
        row["oi_change_cross_venue_dispersion"] = weighted_dispersion(g, "open_interest_pct_change_60m", "weight")
        if {"okx_perp", "bybit_perp"}.issubset(set(g["venue"])):
            okx = g[g["venue"] == "okx_perp"].iloc[-1]
            byb = g[g["venue"] == "bybit_perp"].iloc[-1]
            row["funding_gap_okx_bybit"] = abs(float(okx["funding_rate_8h_equivalent"]) - float(byb["funding_rate_8h_equivalent"])) if finite(okx["funding_rate_8h_equivalent"]) and finite(byb["funding_rate_8h_equivalent"]) else np.nan
            row["basis_gap_okx_bybit"] = abs(float(okx["basis_bps"]) - float(byb["basis_bps"])) if finite(okx["basis_bps"]) and finite(byb["basis_bps"]) else np.nan
            row["oi_growth_gap_okx_bybit"] = abs(float(okx["open_interest_pct_change_60m"]) - float(byb["open_interest_pct_change_60m"])) if finite(okx["open_interest_pct_change_60m"]) and finite(byb["open_interest_pct_change_60m"]) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    add_time_changes(out, "open_interest_composite_usd", "open_interest", (15, 60, 240, 1440))
    add_time_changes(out, "basis_composite_bps", "basis", (15, 60, 240, 1440))
    add_time_changes(out, "funding_composite_8h", "funding", (15, 60, 240, 1440))
    out["price_return_1h"] = np.log((out["canonical_perp_price"] + EPS) / (lag_by_time(out, "canonical_perp_price", 60) + EPS))
    return out


def _z(df: pd.DataFrame, col: str, cfg: LFIConfig, alias: str | None = None) -> pd.Series:
    name = alias or col
    stats = robust_trailing(df[col], cfg.normalization_lookback, cfg.normalization_min_history, cfg.z_clip)
    df[f"z_{name}"] = stats["clipped_z"]
    df[f"z_{name}_raw"] = stats["raw_z"]
    df[f"{name}_percentile"] = stats["percentile"]
    df[f"{name}_norm_median"] = stats["median"]
    df[f"{name}_norm_mad"] = stats["mad"]
    df[f"{name}_norm_count"] = stats["count"]
    return df[f"z_{name}"]


def _add_indices(df: pd.DataFrame, venue_df: pd.DataFrame, cfg: LFIConfig) -> pd.DataFrame:
    if df.empty:
        return df
    missing_required = {
        "funding": int(pd.to_numeric(df["funding_composite_8h"], errors="coerce").notna().sum()),
        "basis": int(pd.to_numeric(df["basis_composite_bps"], errors="coerce").notna().sum()),
        "open_interest": int(pd.to_numeric(df["open_interest_composite_usd"], errors="coerce").notna().sum()),
        "volume_24h": int(pd.to_numeric(df["volume_usd_24h_composite"], errors="coerce").notna().sum()),
    }
    enough = all(v > 0 for v in missing_required.values())
    if not enough:
        df["diagnostics"] = [{"publishable": False, "reason": "missing_required_lfi_inputs", "available_counts": missing_required} for _ in range(len(df))]
        return df

    df["funding_x_oi_1h"] = df["funding_composite_8h"] * df["open_interest_pct_change_60m"]
    df["basis_x_oi_1h"] = df["basis_composite_bps"] * df["open_interest_pct_change_60m"]
    df["neg_funding"] = -df["funding_composite_8h"]
    df["neg_basis"] = -df["basis_composite_bps"]
    df["neg_funding_x_oi_1h"] = -df["funding_x_oi_1h"]
    df["neg_basis_x_oi_1h"] = -df["basis_x_oi_1h"]
    df["neg_price_return_1h"] = -df["price_return_1h"]
    df["neg_oi_change_1h"] = -df["open_interest_pct_change_60m"]
    df["neg_basis_change_1h"] = -df["basis_change_60m"]
    df["pos_price_x_oi_contract"] = np.maximum(0, df["price_return_1h"]) * np.maximum(0, -df["open_interest_pct_change_60m"])
    df["neg_price_x_oi_contract"] = np.maximum(0, -df["price_return_1h"]) * np.maximum(0, -df["open_interest_pct_change_60m"])

    z_f_long = _z(df, "funding_composite_8h", cfg, "funding")
    z_b_long = _z(df, "basis_composite_bps", cfg, "basis")
    z_oi = _z(df, "open_interest_pct_change_60m", cfg, "open_interest_change_1h")
    z_l = _z(df, "oi_to_volume_24h", cfg, "oi_to_volume_24h")
    z_foi_long = _z(df, "funding_x_oi_1h", cfg)
    z_boi_long = _z(df, "basis_x_oi_1h", cfg)
    z_f_short = _z(df, "neg_funding", cfg)
    z_b_short = _z(df, "neg_basis", cfg)
    z_foi_short = _z(df, "neg_funding_x_oi_1h", cfg)
    z_boi_short = _z(df, "neg_basis_x_oi_1h", cfg)

    df["long_crowding_raw"] = 0.25 * z_f_long + 0.20 * z_b_long + 0.15 * z_oi + 0.20 * z_l + 0.10 * z_foi_long + 0.10 * z_boi_long
    df["short_crowding_raw"] = 0.25 * z_f_short + 0.20 * z_b_short + 0.15 * z_oi + 0.20 * z_l + 0.10 * z_foi_short + 0.10 * z_boi_short
    df["long_crowding_percentile"] = percentile_trailing(df["long_crowding_raw"], cfg.normalization_lookback, cfg.normalization_min_history)
    df["short_crowding_percentile"] = percentile_trailing(df["short_crowding_raw"], cfg.normalization_lookback, cfg.normalization_min_history)
    df["long_crowding_regime"] = df["long_crowding_percentile"].map(regime_from_percentile)
    df["short_crowding_regime"] = df["short_crowding_percentile"].map(regime_from_percentile)

    z_lu_price = _z(df, "neg_price_return_1h", cfg)
    z_lu_oi = _z(df, "neg_oi_change_1h", cfg)
    z_lu_basis = _z(df, "neg_basis_change_1h", cfg)
    z_long_liq = _z(df, "long_liquidation_usd_1h", cfg)
    z_lu_int = _z(df, "neg_price_x_oi_contract", cfg)
    z_ss_price = _z(df, "price_return_1h", cfg)
    z_ss_basis = _z(df, "basis_change_60m", cfg, "basis_strength_1h")
    z_short_liq = _z(df, "short_liquidation_usd_1h", cfg)
    z_ss_int = _z(df, "pos_price_x_oi_contract", cfg)
    df["long_unwind_pressure_raw"] = 0.25 * z_lu_price + 0.20 * z_lu_oi + 0.20 * z_lu_basis + 0.20 * z_long_liq + 0.15 * z_lu_int
    df["short_squeeze_pressure_raw"] = 0.25 * z_ss_price + 0.20 * z_lu_oi + 0.20 * z_ss_basis + 0.20 * z_short_liq + 0.15 * z_ss_int
    df["long_unwind_pressure_percentile"] = percentile_trailing(df["long_unwind_pressure_raw"], cfg.normalization_lookback, cfg.normalization_min_history)
    df["short_squeeze_pressure_percentile"] = percentile_trailing(df["short_squeeze_pressure_raw"], cfg.normalization_lookback, cfg.normalization_min_history)

    c_long = df["long_crowding_percentile"] / 100.0
    u_long = df["long_unwind_pressure_percentile"] / 100.0
    c_short = df["short_crowding_percentile"] / 100.0
    u_short = df["short_squeeze_pressure_percentile"] / 100.0
    df["long_fragility_raw"] = 0.45 * c_long + 0.35 * u_long + 0.20 * c_long * u_long
    df["short_fragility_raw"] = 0.45 * c_short + 0.35 * u_short + 0.20 * c_short * u_short
    df["long_fragility_score_0_100"] = 100.0 * df["long_fragility_raw"]
    df["short_fragility_score_0_100"] = 100.0 * df["short_fragility_raw"]
    df["long_fragility_percentile"] = percentile_trailing(df["long_fragility_raw"], cfg.normalization_lookback, cfg.normalization_min_history)
    df["short_fragility_percentile"] = percentile_trailing(df["short_fragility_raw"], cfg.normalization_lookback, cfg.normalization_min_history)
    df["long_fragility_regime"] = df["long_fragility_percentile"].map(regime_from_percentile)
    df["short_fragility_regime"] = df["short_fragility_percentile"].map(regime_from_percentile)

    df["funding_basis_gap"] = df["z_funding"] - df["z_basis"]
    df["funding_basis_divergence_long"] = np.maximum(0, df["z_funding"]) * np.maximum(0, df["z_funding"] - df["z_basis"])
    df["funding_basis_divergence_short"] = np.maximum(0, -df["z_funding"]) * np.maximum(0, df["z_basis"] - df["z_funding"])
    df["funding_oi_deterioration_long"] = np.maximum(0, df["z_funding"]) * np.maximum(0, -df["z_open_interest_change_1h"])
    df["funding_oi_deterioration_short"] = np.maximum(0, df["z_neg_funding"]) * np.maximum(0, -df["z_open_interest_change_1h"])
    df["basis_oi_deterioration_long"] = np.maximum(0, df["z_basis"]) * np.maximum(0, -df["z_open_interest_change_1h"])
    df["basis_oi_deterioration_short"] = np.maximum(0, df["z_neg_basis"]) * np.maximum(0, -df["z_open_interest_change_1h"])

    venue_latest = venue_df.merge(df[["timestamp"]], on="timestamp", how="inner")
    long_agree = []
    short_agree = []
    for ts, g in venue_latest.groupby("timestamp", sort=True):
        # Minimal venue agreement uses funding/basis direction plus local OI growth.
        w = pd.to_numeric(g["weight"], errors="coerce").fillna(0)
        long_sig = (pd.to_numeric(g["funding_rate_8h_equivalent"], errors="coerce") > 0) & (pd.to_numeric(g["basis_bps"], errors="coerce") > 0) & (pd.to_numeric(g["open_interest_pct_change_60m"], errors="coerce") > 0)
        short_sig = (pd.to_numeric(g["funding_rate_8h_equivalent"], errors="coerce") < 0) & (pd.to_numeric(g["basis_bps"], errors="coerce") < 0) & (pd.to_numeric(g["open_interest_pct_change_60m"], errors="coerce") > 0)
        long_agree.append((ts, float((w * long_sig.astype(float)).sum() / (w.sum() + EPS)) if w.sum() > 0 else np.nan))
        short_agree.append((ts, float((w * short_sig.astype(float)).sum() / (w.sum() + EPS)) if w.sum() > 0 else np.nan))
    df = df.merge(pd.DataFrame(long_agree, columns=["timestamp", "long_crowding_agreement"]), on="timestamp", how="left")
    df = df.merge(pd.DataFrame(short_agree, columns=["timestamp", "short_crowding_agreement"]), on="timestamp", how="left")

    for col in ("long_fragility_score_0_100", "short_fragility_score_0_100"):
        base = col.replace("_score_0_100", "")
        df[f"{base}_velocity_15m"] = df[col] - lag_by_time(df, col, 15)
        df[f"{base}_velocity_1h"] = df[col] - lag_by_time(df, col, 60)
        prev_vel = lag_by_time(pd.DataFrame({"timestamp": df["timestamp"], f"{base}_velocity_15m": df[f"{base}_velocity_15m"]}), f"{base}_velocity_15m", 15)
        df[f"{base}_acceleration_15m"] = df[f"{base}_velocity_15m"] - prev_vel
        df[f"{base}_impulse"] = df[col] - lag_by_time(df, col, 1440)

    df["leverage_instability_raw"] = (
        0.25 * _z(df, "funding_change_60m", cfg, "abs_funding_change_1h").abs()
        + 0.20 * _z(df, "basis_change_60m", cfg, "abs_basis_change_1h").abs()
        + 0.20 * _z(df, "open_interest_pct_change_60m", cfg, "abs_open_interest_change_1h").abs()
        + 0.15 * _z(df, "funding_cross_venue_dispersion", cfg).fillna(0)
        + 0.10 * _z(df, "basis_cross_venue_dispersion", cfg).fillna(0)
        + 0.10 * _z(df, "oi_change_cross_venue_dispersion", cfg).fillna(0)
    )
    df["leverage_instability_percentile"] = percentile_trailing(df["leverage_instability_raw"], cfg.normalization_lookback, cfg.normalization_min_history)
    df["leverage_fragility_unsigned"] = df[["long_fragility_score_0_100", "short_fragility_score_0_100"]].max(axis=1)
    df["leverage_fragility_direction"] = np.where(df["long_fragility_score_0_100"] >= df["short_fragility_score_0_100"], "long_downside", "short_upside")
    df["long_state"] = [_state_label("long", c, u, cfg) for c, u in zip(df["long_crowding_percentile"], df["long_unwind_pressure_percentile"], strict=False)]
    df["short_state"] = [_state_label("short", c, u, cfg) for c, u in zip(df["short_crowding_percentile"], df["short_squeeze_pressure_percentile"], strict=False)]
    df["long_state_code"] = df["long_state"].map(_state_code)
    df["short_state_code"] = df["short_state"].map(_state_code)
    df["leverage_stress_leader"] = np.where(
        pd.to_numeric(df.get("funding_gap_okx_bybit", np.nan), errors="coerce").fillna(0) > 0,
        "DIVERGENT",
        "NONE",
    )
    df["diagnostics"] = [{"publishable": True, "reason": "ok", "available_counts": missing_required} for _ in range(len(df))]
    return df


def compute_leverage_fragility(canonical_rows: pd.DataFrame, cfg: LFIConfig | None = None) -> pd.DataFrame:
    cfg = cfg or LFIConfig()
    missing = [c for c in REQUIRED_CANONICAL_COLUMNS if c not in canonical_rows.columns]
    if missing:
        raise ValueError(f"canonical derivatives rows missing columns: {missing}")
    venue = _prepare_venue_rows(canonical_rows, cfg)
    if venue.empty:
        return pd.DataFrame()
    venue = _add_venue_history(venue)
    comp = _build_composite(venue, cfg)
    return _add_indices(comp, venue, cfg)
