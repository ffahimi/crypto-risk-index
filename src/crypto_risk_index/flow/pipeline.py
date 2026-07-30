from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CALC_VERSION = "oti_proxy_v0_available_venue_metrics"


@dataclass(frozen=True)
class OTIProxyConfig:
    """Configuration for the debug-only available-data OTI proxy.

    This is not full equal-volume VPIN.  It uses signed venue price-flow,
    dispersion, and spread fields that are available in the current TFI input
    tables, and keeps all unavailable VPIN/depth concepts visibly absent.
    """

    fast_window: int = 20
    standard_window: int = 50
    slow_window: int = 200
    normalization_lookback: int = 500
    normalization_min_history: int = 50
    z_clip: float = 4.0


def _finite_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.where(np.isfinite(den) & (den.abs() > 1e-12))
    return num / den


def _rolling_robust_z(series: pd.Series, lookback: int, min_periods: int) -> pd.DataFrame:
    def calc(arr: np.ndarray) -> np.ndarray:
        cur = arr[-1]
        hist = arr[:-1]
        hist = hist[np.isfinite(hist)]
        if not np.isfinite(cur) or hist.size < min_periods:
            return np.asarray([np.nan, np.nan, np.nan, float(hist.size)], dtype=float)
        med = float(np.median(hist))
        mad = float(np.median(np.abs(hist - med)))
        scale = max(1.4826 * mad, 1e-9)
        return np.asarray([(float(cur) - med) / scale, med, scale, float(hist.size)], dtype=float)

    roll = series.rolling(lookback + 1, min_periods=min_periods + 1)
    raw_z = roll.apply(lambda arr: calc(arr)[0], raw=True)
    med = roll.apply(lambda arr: calc(arr)[1], raw=True)
    scale = roll.apply(lambda arr: calc(arr)[2], raw=True)
    n = series.rolling(lookback + 1, min_periods=1).apply(
        lambda arr: float(np.count_nonzero(np.isfinite(arr[:-1]))),
        raw=True,
    )
    return pd.DataFrame({"raw_z": raw_z, "median": med, "scale": scale, "n": n}, index=series.index)


def _rolling_percentile(series: pd.Series, lookback: int, min_periods: int) -> pd.Series:
    def calc(arr: np.ndarray) -> float:
        cur = arr[-1]
        hist = arr[:-1]
        hist = hist[np.isfinite(hist)]
        if not np.isfinite(cur) or hist.size < min_periods:
            return np.nan
        return 100.0 * float(np.count_nonzero(hist <= cur)) / float(hist.size)

    return series.rolling(lookback + 1, min_periods=min_periods + 1).apply(calc, raw=True)


def _regime(pct: Any) -> str | None:
    try:
        val = float(pct)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    if val >= 97.5:
        return "extreme"
    if val >= 90.0:
        return "severe"
    if val >= 75.0:
        return "toxic"
    if val >= 50.0:
        return "elevated"
    return "normal"


def _price_cols(raw: pd.DataFrame) -> list[str]:
    return [c for c in ("coinbase_mid", "okx_mid", "bybit_mid") if c in raw]


def _perp_cols(raw: pd.DataFrame) -> list[str]:
    return [c for c in ("okx_mid", "bybit_mid") if c in raw]


def _spread_bps(raw: pd.DataFrame) -> pd.Series:
    spreads = []
    for prefix in ("coinbase", "okx", "bybit"):
        bid = _finite_series(raw, f"{prefix}_bid")
        ask = _finite_series(raw, f"{prefix}_ask")
        mid = _finite_series(raw, f"{prefix}_mid")
        s = 10000.0 * (ask - bid) / mid.replace(0, np.nan)
        spreads.append(s.where(np.isfinite(s) & (s >= 0)))
    if not spreads:
        return pd.Series(np.nan, index=raw.index, dtype=float)
    return pd.concat(spreads, axis=1).median(axis=1, skipna=True)


def _dispersion_bps(raw: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=raw.index, dtype=float)
    mids = raw[cols].apply(pd.to_numeric, errors="coerce")
    med = mids.median(axis=1, skipna=True)
    disp = 10000.0 * (mids.max(axis=1, skipna=True) - mids.min(axis=1, skipna=True)) / med.replace(0, np.nan)
    return disp.where(np.isfinite(disp) & (disp >= 0))


def compute_oti_proxy(raw: pd.DataFrame, *, cfg: OTIProxyConfig | None = None, tfi_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute a debug-only OTI proxy from available minute venue metrics.

    This is causal and trailing-only, but it is not standard VPIN because the
    current input path does not expose canonical aggressor-side trades.  The
    proxy uses signed price-flow and venue agreement as a temporary Phase-II
    diagnostic until true trade buckets are wired.
    """

    cfg = cfg or OTIProxyConfig()
    if raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"timestamp": pd.to_datetime(raw["timestamp"], utc=True)})
    mids = _price_cols(raw)
    perps = _perp_cols(raw)
    out["source_count"] = raw[mids].notna().sum(axis=1) if mids else 0
    out["perp_source_count"] = raw[perps].notna().sum(axis=1) if perps else 0

    if perps:
        out["canonical_perp_price"] = raw[perps].apply(pd.to_numeric, errors="coerce").median(axis=1, skipna=True)
    else:
        out["canonical_perp_price"] = raw[mids].apply(pd.to_numeric, errors="coerce").median(axis=1, skipna=True) if mids else np.nan
    out["perp_dispersion_bps"] = _dispersion_bps(raw, perps)
    out["all_venue_dispersion_bps"] = _dispersion_bps(raw, mids)
    out["spread_bps"] = _spread_bps(raw)

    venue_rets = []
    venue_sell = []
    venue_buy = []
    for col in mids:
        ret_col = f"{col.removesuffix('_mid')}_ret_bp"
        mid = pd.to_numeric(raw[col], errors="coerce")
        ret = 10000.0 * np.log(mid / mid.shift(1))
        out[ret_col] = ret.replace([np.inf, -np.inf], np.nan)
        venue_rets.append(ret_col)
        venue_sell.append(out[ret_col] < 0)
        venue_buy.append(out[ret_col] > 0)

    if venue_rets:
        ret_frame = out[venue_rets]
        out["composite_return_bp"] = ret_frame.median(axis=1, skipna=True)
        out["venue_sell_agreement"] = pd.concat(venue_sell, axis=1).sum(axis=1) / out["source_count"].replace(0, np.nan)
        out["venue_buy_agreement"] = pd.concat(venue_buy, axis=1).sum(axis=1) / out["source_count"].replace(0, np.nan)
    else:
        out["composite_return_bp"] = np.nan
        out["venue_sell_agreement"] = np.nan
        out["venue_buy_agreement"] = np.nan

    flow_abs = out["composite_return_bp"].abs()
    flow_sell = (-out["composite_return_bp"]).clip(lower=0)
    flow_buy = out["composite_return_bp"].clip(lower=0)

    for win, name in ((cfg.fast_window, "20"), (cfg.standard_window, "50"), (cfg.slow_window, "200")):
        abs_sum = flow_abs.rolling(win, min_periods=max(3, min(win, 10))).sum()
        signed_sum = out["composite_return_bp"].rolling(win, min_periods=max(3, min(win, 10))).sum()
        out[f"vpin_proxy_{name}"] = _safe_div(signed_sum.abs(), abs_sum).clip(0, 1)
        out[f"signed_flow_proxy_{name}"] = _safe_div(signed_sum, abs_sum).clip(-1, 1)
        out[f"sell_tox_proxy_{name}"] = _safe_div(flow_sell.rolling(win, min_periods=max(3, min(win, 10))).sum(), abs_sum).clip(0, 1)
        out[f"buy_tox_proxy_{name}"] = _safe_div(flow_buy.rolling(win, min_periods=max(3, min(win, 10))).sum(), abs_sum).clip(0, 1)

    out["vpin_impulse_proxy_20_200"] = out["vpin_proxy_20"] - out["vpin_proxy_200"]
    out["sell_persistence_proxy_50"] = (out["sell_tox_proxy_50"] > out["sell_tox_proxy_50"].rolling(200, min_periods=50).quantile(0.9)).rolling(50, min_periods=10).mean()
    out["buy_persistence_proxy_50"] = (out["buy_tox_proxy_50"] > out["buy_tox_proxy_50"].rolling(200, min_periods=50).quantile(0.9)).rolling(50, min_periods=10).mean()
    out["flow_speed_proxy_20"] = flow_abs.rolling(cfg.fast_window, min_periods=5).mean()

    components = {
        "vpin": ("vpin_proxy_50", 0.15),
        "impulse": ("vpin_impulse_proxy_20_200", 0.10),
        "sell": ("sell_tox_proxy_50", 0.20),
        "buy": ("buy_tox_proxy_50", 0.20),
        "sell_persist": ("sell_persistence_proxy_50", 0.10),
        "buy_persist": ("buy_persistence_proxy_50", 0.10),
        "flow_speed": ("flow_speed_proxy_20", 0.10),
        "dispersion": ("perp_dispersion_bps", 0.10),
        "spread": ("spread_bps", 0.10),
    }

    zcols: dict[str, str] = {}
    for name, (col, _weight) in components.items():
        stats = _rolling_robust_z(pd.to_numeric(out[col], errors="coerce"), cfg.normalization_lookback, cfg.normalization_min_history)
        out[f"z_{name}_raw"] = stats["raw_z"]
        out[f"z_{name}"] = stats["raw_z"].clip(-cfg.z_clip, cfg.z_clip)
        out[f"z_{name}_norm_n"] = stats["n"]
        zcols[name] = f"z_{name}"

    def weighted_score(items: list[tuple[str, float]]) -> pd.Series:
        total = pd.Series(0.0, index=out.index)
        used = pd.Series(0.0, index=out.index)
        for name, weight in items:
            z = out[zcols[name]]
            mask = z.notna()
            total.loc[mask] += z.loc[mask] * weight
            used.loc[mask] += weight
        return total.where(used > 0) / used.where(used > 0)

    out["oti_sell_raw"] = weighted_score([
        ("vpin", 0.15),
        ("impulse", 0.10),
        ("sell", 0.25),
        ("sell_persist", 0.10),
        ("flow_speed", 0.10),
        ("dispersion", 0.15),
        ("spread", 0.15),
    ])
    out["oti_buy_raw"] = weighted_score([
        ("vpin", 0.15),
        ("impulse", 0.10),
        ("buy", 0.25),
        ("buy_persist", 0.10),
        ("flow_speed", 0.10),
        ("dispersion", 0.15),
        ("spread", 0.15),
    ])
    out["oti_unsigned_raw"] = weighted_score([
        ("vpin", 0.20),
        ("impulse", 0.15),
        ("flow_speed", 0.20),
        ("dispersion", 0.25),
        ("spread", 0.20),
    ])
    for col in ("oti_sell_raw", "oti_buy_raw", "oti_unsigned_raw"):
        pct_col = col.replace("_raw", "_percentile")
        out[pct_col] = _rolling_percentile(out[col], cfg.normalization_lookback, cfg.normalization_min_history)
    out["oti_sell_regime"] = out["oti_sell_percentile"].map(_regime)
    out["oti_buy_regime"] = out["oti_buy_percentile"].map(_regime)
    out["oti_unsigned_regime"] = out["oti_unsigned_percentile"].map(_regime)

    if tfi_features is not None and not tfi_features.empty and "timestamp" in tfi_features:
        tfi = tfi_features[["timestamp", "tfi_percentile_rate_only"]].copy()
        tfi["timestamp"] = pd.to_datetime(tfi["timestamp"], utc=True)
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            tfi.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        tfi01 = pd.to_numeric(out["tfi_percentile_rate_only"], errors="coerce") / 100.0
        oti01 = pd.to_numeric(out["oti_sell_percentile"], errors="coerce") / 100.0
        out["jri_proxy_sell"] = 100.0 * (0.35 * tfi01 + 0.35 * oti01 + 0.30 * tfi01 * oti01)
    else:
        out["tfi_percentile_rate_only"] = np.nan
        out["jri_proxy_sell"] = np.nan
    out["calculation_version"] = CALC_VERSION
    return out


def _fmt(v: Any, nd: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(f):
        return "NA"
    return f"{f:.{nd}f}"


def _signed(v: Any, nd: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(f):
        return "NA"
    return f"{f:+.{nd}f}"


def compact_oti_debug_line(row: dict[str, Any]) -> str:
    regime = str(row.get("oti_sell_regime") or "unknown").upper()
    return (
        f"[OTI-PROXY {row.get('timestamp')}] "
        f"px={_fmt(row.get('canonical_perp_price'), 2)} src={_fmt(row.get('source_count'), 0)} "
        f"perp_disp={_fmt(row.get('perp_dispersion_bps'), 2)}bp spread={_fmt(row.get('spread_bps'), 2)}bp | "
        f"vpinP20/50/200={_fmt(row.get('vpin_proxy_20'))}/{_fmt(row.get('vpin_proxy_50'))}/{_fmt(row.get('vpin_proxy_200'))} "
        f"imp={_signed(row.get('vpin_impulse_proxy_20_200'))} | "
        f"sellP20/50={_fmt(row.get('sell_tox_proxy_20'))}/{_fmt(row.get('sell_tox_proxy_50'))} "
        f"buyP20/50={_fmt(row.get('buy_tox_proxy_20'))}/{_fmt(row.get('buy_tox_proxy_50'))} "
        f"sellAgree={_fmt(row.get('venue_sell_agreement'), 2)} buyAgree={_fmt(row.get('venue_buy_agreement'), 2)} | "
        f"z vpin/imp/sell/buy/flow/disp/spread="
        f"{_signed(row.get('z_vpin'))}/{_signed(row.get('z_impulse'))}/"
        f"{_signed(row.get('z_sell'))}/{_signed(row.get('z_buy'))}/"
        f"{_signed(row.get('z_flow_speed'))}/{_signed(row.get('z_dispersion'))}/"
        f"{_signed(row.get('z_spread'))} | "
        f"otiSell={_fmt(row.get('oti_sell_raw'))} pct={_fmt(row.get('oti_sell_percentile'), 1)} risk={regime} "
        f"otiBuy={_fmt(row.get('oti_buy_raw'))} pct={_fmt(row.get('oti_buy_percentile'), 1)} "
        f"jriSell={_fmt(row.get('jri_proxy_sell'), 1)} "
        f"note=price-flow proxy, not true equal-volume VPIN"
    )
