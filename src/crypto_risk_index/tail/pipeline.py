from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .lf import (
    DEFAULT_SURFACES,
    RateSurface,
    SurfaceConfig,
    default_x_grid,
    left_area,
    legendre_rate_curve,
    theta_at,
    theta_grid,
    value_at,
)


CALC_VERSION = "lf_tfi_v1_rate_only"


@dataclass(frozen=True)
class PipelineConfig:
    asset: str
    price_source: str = "perp_composite"
    calculation_version: str = CALC_VERSION
    max_quote_age_seconds: float = 10.0
    max_return_gap_minutes: int = 1
    standardization: str = "robust"
    scale_floor: float = 1e-6
    z_clip: float = 12.0
    theta_min: float = -6.0
    theta_max: float = 6.0
    theta_step: float = 0.02
    output_cadence_minutes: int = 1
    normalization_lookback_minutes: int = 30 * 24 * 60
    normalization_min_history: int = 120
    component_z_clip: float = 4.0
    store_surface_cadence_minutes: int = 15


def _finite(x: Any) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def build_canonical_prices(raw: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    max_age_ms = cfg.max_quote_age_seconds * 1000.0
    rows: list[dict[str, Any]] = []
    for r in df.to_dict("records"):
        bybit_mid = float(r["bybit_mid"]) if _finite(r.get("bybit_mid")) else np.nan
        okx_mid = float(r["okx_mid"]) if _finite(r.get("okx_mid")) else np.nan
        coinbase_mid = float(r["coinbase_mid"]) if _finite(r.get("coinbase_mid")) else np.nan
        bybit_age = float(r["bybit_age_ms"]) if _finite(r.get("bybit_age_ms")) else np.nan
        okx_age = float(r["okx_age_ms"]) if _finite(r.get("okx_age_ms")) else np.nan
        coinbase_age = float(r["coinbase_age_ms"]) if _finite(r.get("coinbase_age_ms")) else np.nan
        bybit_ok = np.isfinite(bybit_mid) and (not np.isfinite(bybit_age) or bybit_age <= max_age_ms)
        okx_ok = np.isfinite(okx_mid) and (not np.isfinite(okx_age) or okx_age <= max_age_ms)
        coinbase_ok = np.isfinite(coinbase_mid) and (not np.isfinite(coinbase_age) or coinbase_age <= max_age_ms)
        perp_vals = [v for ok, v in ((bybit_ok, bybit_mid), (okx_ok, okx_mid)) if ok and np.isfinite(v)]
        spot = coinbase_mid if coinbase_ok else np.nan

        if cfg.price_source == "coinbase_spot":
            canonical = spot
            method = "coinbase_spot" if np.isfinite(canonical) else "missing"
        elif len(perp_vals) == 2:
            canonical = float(np.median(perp_vals))
            method = "median_bybit_okx"
        elif len(perp_vals) == 1:
            canonical = float(perp_vals[0])
            method = "single_bybit" if bybit_ok else "single_okx"
        else:
            canonical = np.nan
            method = "missing"

        source_count = int(bybit_ok) + int(okx_ok)
        dispersion = np.nan
        if bybit_ok and okx_ok and np.isfinite(canonical) and canonical != 0:
            dispersion = 1e4 * abs(bybit_mid - okx_mid) / canonical
        basis = np.nan
        if np.isfinite(canonical) and np.isfinite(spot) and spot != 0:
            basis = (canonical - spot) / spot
        data_quality = 0.0
        if np.isfinite(canonical):
            data_quality = 0.55 + 0.2 * source_count + (0.05 if coinbase_ok else 0.0)
            data_quality = min(data_quality, 1.0)
        rows.append(
            {
                "timestamp": r["timestamp"],
                "asset": cfg.asset.upper(),
                "price_source": cfg.price_source,
                "canonical_perp_price": canonical,
                "coinbase_spot_price": spot,
                "basis": basis,
                "cross_venue_dispersion_bps": dispersion,
                "data_quality_score": data_quality,
                "source_count": source_count,
                "canonical_price_method": method,
                "bybit_available": bybit_ok,
                "okx_available": okx_ok,
                "coinbase_available": coinbase_ok,
                "bybit_quote_age_ms": bybit_age,
                "okx_quote_age_ms": okx_age,
                "coinbase_quote_age_ms": coinbase_age,
            }
        )
    out = pd.DataFrame(rows)
    out = out[np.isfinite(out["canonical_perp_price"].astype(float))].copy()
    return out.reset_index(drop=True)


def add_gap_safe_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("timestamp").copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    price = out["canonical_perp_price"].astype(float)
    for minutes, col in ((1, "return_1m"), (5, "return_5m"), (15, "return_15m"), (60, "return_1h")):
        shifted_price = price.shift(minutes)
        shifted_ts = out["timestamp"].shift(minutes)
        expected = pd.to_timedelta(minutes, unit="m")
        exact_gap = out["timestamp"] - shifted_ts == expected
        # All intermediate rows must be one-minute apart.
        minute_ok = out["timestamp"].diff().eq(pd.Timedelta(minutes=1))
        full_path = minute_ok.rolling(minutes, min_periods=minutes).sum().eq(minutes)
        ret = np.log(price / shifted_price)
        out[col] = np.where(exact_gap & full_path & np.isfinite(ret), ret, np.nan)
    return out


def _window_returns(df: pd.DataFrame, idx: int, cfg: SurfaceConfig) -> np.ndarray:
    ts = df.at[idx, "timestamp"]
    start = ts - pd.Timedelta(minutes=cfg.lookback_minutes)
    mask = (df["timestamp"] > start) & (df["timestamp"] <= ts)
    return df.loc[mask, cfg.return_col].dropna().to_numpy(dtype=float)


def _surface_to_features(surface: RateSurface, suffix: str) -> dict[str, Any]:
    return {
        f"lf_i_neg_2_{suffix}": value_at(surface, -2.0),
        f"lf_i_neg_3_{suffix}": value_at(surface, -3.0),
        f"lf_i_neg_4_{suffix}": value_at(surface, -4.0),
        f"lf_theta_star_neg_3_{suffix}": theta_at(surface, -3.0),
    }


def _safe_sub(a: Any, b: Any) -> float:
    if _finite(a) and _finite(b):
        return float(a) - float(b)
    return np.nan


def _safe_div(a: Any, b: Any) -> float:
    if _finite(a) and _finite(b) and abs(float(b)) > 1e-12:
        return float(a) / float(b)
    return np.nan


def compute_surface_features(
    df: pd.DataFrame,
    *,
    cfg: PipelineConfig,
    surfaces: tuple[SurfaceConfig, ...] = DEFAULT_SURFACES,
    surface_warmup_returns: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if df.empty:
        return pd.DataFrame(), []
    x_values = default_x_grid()
    thetas = theta_grid(cfg.theta_min, cfg.theta_max, cfg.theta_step)
    rows: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []
    cadence = max(int(cfg.output_cadence_minutes), 1)
    surface_cadence = max(int(cfg.store_surface_cadence_minutes), 1)

    for idx in range(len(df)):
        ts = df.at[idx, "timestamp"]
        if ts.minute % cadence != 0:
            continue
        base = df.iloc[idx].to_dict()
        base["calculation_version"] = cfg.calculation_version
        diagnostics: dict[str, Any] = {
            "liquidation_features_available": False,
            "liquidation_reason": "No Bybit/OKX liquidation/OI tables found in the current mapped DB sources.",
            "surfaces": {},
        }
        computed: dict[str, RateSurface] = {}
        for surf_cfg in surfaces:
            sample = _surface_warmup_sample(surface_warmup_returns, ts, surf_cfg)
            if sample is None:
                sample = _window_returns(df, idx, surf_cfg)
            suffix = surf_cfg.name
            if len(sample) < surf_cfg.min_observations:
                diagnostics["surfaces"][suffix] = {
                    "ready": False,
                    "sample_count": int(len(sample)),
                    "min_observations": surf_cfg.min_observations,
                    "return_col": surf_cfg.return_col,
                }
                continue
            surf = legendre_rate_curve(
                sample,
                window_name=surf_cfg.name,
                return_col=surf_cfg.return_col,
                x_values=x_values,
                thetas=thetas,
                standardization=cfg.standardization,
                scale_floor=cfg.scale_floor,
                z_clip=cfg.z_clip,
            )
            computed[suffix] = surf
            base.update(_surface_to_features(surf, suffix))
            diagnostics["surfaces"][suffix] = {
                "ready": True,
                "sample_count": surf.sample_count,
                "location": surf.location,
                "scale": surf.scale,
                "clipped_count": surf.clipped_count,
                "clipped_pct": surf.clipped_pct,
                "theta_boundary_count": surf.theta_boundary_count,
                "support_warning_count": surf.support_warning_count,
                "convexity_min_second_diff": surf.convexity_min_second_diff,
                "min_rate_value": surf.min_rate_value,
            }
            if ts.minute % surface_cadence == 0:
                for x, rate, theta in zip(surf.x_grid, surf.rate_values, surf.theta_star):
                    surface_rows.append(
                        {
                            "asset": cfg.asset.upper(),
                            "timestamp": ts,
                            "window_name": surf.window_name,
                            "return_resolution": surf.return_col.replace("return_", ""),
                            "price_source": cfg.price_source,
                            "x_value": float(x),
                            "rate_value": float(rate),
                            "theta_star": float(theta),
                            "sample_count": surf.sample_count,
                            "location_estimate": surf.location,
                            "scale_estimate": surf.scale,
                            "data_quality_score": base.get("data_quality_score"),
                            "calculation_version": cfg.calculation_version,
                            "diagnostics": diagnostics["surfaces"][suffix],
                        }
                    )

        s24 = computed.get("24h")
        s6 = computed.get("6h")
        s7 = computed.get("7d")
        s30 = computed.get("30d")
        if s24 is not None:
            base["lf_asymmetry_2_24h"] = value_at(s24, 2.0) - value_at(s24, -2.0)
            base["lf_asymmetry_3_24h"] = value_at(s24, 3.0) - value_at(s24, -3.0)
            base["lf_asymmetry_4_24h"] = value_at(s24, 4.0) - value_at(s24, -4.0)
            base["lf_left_slope_24h"] = (value_at(s24, -4.0) - value_at(s24, -2.0)) / 2.0
            base["lf_left_curvature_24h"] = value_at(s24, -4.0) - 2.0 * value_at(s24, -3.0) + value_at(s24, -2.0)
            base["lf_left_area_24h"] = left_area(s24)
        if s30 is not None and s24 is not None:
            base["lf_collapse_3_30d_24h"] = value_at(s30, -3.0) - value_at(s24, -3.0)
            base["lf_collapse_4_30d_24h"] = value_at(s30, -4.0) - value_at(s24, -4.0)
            base["lf_area_collapse_30d_24h"] = left_area(s30) - left_area(s24)
        if s7 is not None and s24 is not None:
            base["lf_collapse_3_7d_24h"] = value_at(s7, -3.0) - value_at(s24, -3.0)
            base["lf_collapse_4_7d_24h"] = value_at(s7, -4.0) - value_at(s24, -4.0)
        if s24 is not None and s6 is not None:
            base["lf_collapse_3_24h_6h"] = value_at(s24, -3.0) - value_at(s6, -3.0)
            base["lf_collapse_4_24h_6h"] = value_at(s24, -4.0) - value_at(s6, -4.0)
        base["diagnostics"] = diagnostics
        rows.append(base)

    feat = pd.DataFrame(rows)
    if not feat.empty:
        feat = add_velocity_and_tfi(feat, cfg)
    return feat, surface_rows


def _surface_warmup_sample(
    warmup: pd.DataFrame | None,
    ts: pd.Timestamp,
    cfg: SurfaceConfig,
) -> np.ndarray | None:
    if warmup is None or warmup.empty:
        return None
    if not {"surface", "bucket_ts", "log_ret"}.issubset(warmup.columns):
        return None
    bucket_ts = pd.to_datetime(warmup["bucket_ts"], utc=True)
    start = ts - pd.Timedelta(minutes=cfg.lookback_minutes)
    mask = (warmup["surface"] == cfg.name) & (bucket_ts > start) & (bucket_ts <= ts)
    vals = pd.to_numeric(warmup.loc[mask, "log_ret"], errors="coerce").dropna().to_numpy(dtype=float)
    return vals


def _rolling_robust_z(series: pd.Series, lookback: int, min_periods: int) -> pd.Series:
    def calc(arr: np.ndarray) -> float:
        cur = arr[-1]
        hist = arr[:-1]
        hist = hist[np.isfinite(hist)]
        if not np.isfinite(cur) or hist.size < min_periods:
            return np.nan
        med = np.median(hist)
        mad = np.median(np.abs(hist - med))
        scale = max(1.4826 * mad, 1e-9)
        return float((cur - med) / scale)

    return series.rolling(lookback + 1, min_periods=min_periods + 1).apply(calc, raw=True)


def _rolling_robust_z_stats(series: pd.Series, lookback: int, min_periods: int, prefix: str) -> pd.DataFrame:
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

    vals = series.rolling(lookback + 1, min_periods=min_periods + 1).apply(
        lambda arr: calc(arr)[0],
        raw=True,
    )
    med = series.rolling(lookback + 1, min_periods=min_periods + 1).apply(
        lambda arr: calc(arr)[1],
        raw=True,
    )
    scale = series.rolling(lookback + 1, min_periods=min_periods + 1).apply(
        lambda arr: calc(arr)[2],
        raw=True,
    )
    n = series.rolling(lookback + 1, min_periods=1).apply(
        lambda arr: float(np.count_nonzero(np.isfinite(arr[:-1]))),
        raw=True,
    )
    return pd.DataFrame(
        {
            f"{prefix}_z_raw": vals,
            f"{prefix}_norm_median": med,
            f"{prefix}_norm_mad_scale": scale,
            f"{prefix}_norm_n": n,
        },
        index=series.index,
    )


def _rolling_percentile(series: pd.Series, lookback: int, min_periods: int) -> pd.Series:
    def calc(arr: np.ndarray) -> float:
        cur = arr[-1]
        hist = arr[:-1]
        hist = hist[np.isfinite(hist)]
        if not np.isfinite(cur) or hist.size < min_periods:
            return np.nan
        return float(np.mean(hist <= cur) * 100.0)

    return series.rolling(lookback + 1, min_periods=min_periods + 1).apply(calc, raw=True)


def add_velocity_and_tfi(feat: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = feat.sort_values("timestamp").reset_index(drop=True).copy()
    out = out.set_index("timestamp", drop=False)
    for col, lag_min, out_col, danger in (
        ("lf_i_neg_3_24h", 15, "lf_cost3_velocity_15m", True),
        ("lf_i_neg_3_24h", 60, "lf_cost3_velocity_1h", True),
        ("lf_i_neg_4_24h", 15, "lf_cost4_velocity_15m", True),
    ):
        if col in out:
            lagged = _asof_time_lag(out[col], minutes=lag_min)
            diff = out[col] - lagged
            out[f"{out_col}_raw"] = diff
            out[out_col] = -diff if danger else diff
    if "lf_cost3_velocity_15m" in out:
        out["lf_cost3_acceleration_15m"] = out["lf_cost3_velocity_15m"] - _asof_time_lag(out["lf_cost3_velocity_15m"], minutes=15)

    lookback = max(int(cfg.normalization_lookback_minutes / max(cfg.output_cadence_minutes, 1)), 1)
    min_hist = min(cfg.normalization_min_history, max(lookback - 1, 1))
    z_sources = {
        "lf_i_neg_3_danger": -out.get("lf_i_neg_3_24h", pd.Series(index=out.index, dtype=float)),
        "lf_collapse_3_danger": out.get("lf_collapse_3_30d_24h", pd.Series(index=out.index, dtype=float)),
        "lf_asymmetry_3_danger": out.get("lf_asymmetry_3_24h", pd.Series(index=out.index, dtype=float)),
        "lf_velocity_3_danger": out.get("lf_cost3_velocity_15m", pd.Series(index=out.index, dtype=float)),
    }
    for prefix, series in z_sources.items():
        stats = _rolling_robust_z_stats(pd.to_numeric(series, errors="coerce"), lookback, min_hist, prefix)
        for col in stats:
            out[col] = stats[col]
        raw_col = f"{prefix}_z_raw"
        out[f"{prefix}_z"] = out[raw_col].clip(-cfg.component_z_clip, cfg.component_z_clip)
    out["liquidation_pressure_danger_z"] = np.nan

    weights = {
        "lf_i_neg_3_danger_z": 0.25,
        "lf_collapse_3_danger_z": 0.20,
        "lf_asymmetry_3_danger_z": 0.15,
        "lf_velocity_3_danger_z": 0.15,
    }
    weighted = pd.Series(0.0, index=out.index)
    used_w = pd.Series(0.0, index=out.index)
    for col, w in weights.items():
        val = out[col]
        mask = np.isfinite(val)
        weighted.loc[mask] += val.loc[mask] * w
        used_w.loc[mask] += w
    out["tfi_raw_rate_only"] = np.where(used_w > 0, weighted / used_w, np.nan)
    out["tfi_percentile_rate_only"] = _rolling_percentile(out["tfi_raw_rate_only"], lookback, min_hist)
    out["tfi_raw"] = out["tfi_raw_rate_only"]
    out["tfi_percentile"] = out["tfi_percentile_rate_only"]
    out["tfi_regime"] = out["tfi_percentile"].map(regime_label)
    return out.reset_index(drop=True)


def _asof_time_lag(series: pd.Series, *, minutes: int) -> pd.Series:
    if series.empty:
        return pd.Series(index=series.index, dtype=float)
    idx = pd.DatetimeIndex(series.index)
    targets = idx - pd.Timedelta(minutes=minutes)
    positions = idx.get_indexer(targets, method="ffill")
    vals = np.full(len(series), np.nan, dtype=float)
    valid = positions >= 0
    if np.any(valid):
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        vals[valid] = arr[positions[valid]]
    return pd.Series(vals, index=series.index)


def regime_label(pct: Any) -> str | None:
    if not _finite(pct):
        return None
    p = float(pct)
    if p < 50:
        return "normal"
    if p < 75:
        return "elevated"
    if p < 90:
        return "fragile"
    if p < 97.5:
        return "severe"
    return "extreme"


ANSI = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "red": "\033[31m",
    "red_bg": "\033[41;97m",
    "cyan": "\033[36m",
}


def compact_debug_line(row: dict[str, Any], *, color: bool = False) -> str:
    regime = row.get("tfi_regime") or regime_label(row.get("tfi_percentile_rate_only")) or "unknown"
    regime_text = _regime_text(regime, color=color)
    pct = _fmt(row.get("tfi_percentile_rate_only"), nd=1)
    tfi = _fmt(row.get("tfi_raw_rate_only"), nd=3)
    collapse = row.get("lf_collapse_3_30d_24h")
    asym = row.get("lf_asymmetry_3_24h")
    collapse_hint = _collapse_hint(collapse)
    asym_hint = _asymmetry_hint(asym)
    z_text = "/".join(
        [
            _signed_fmt(row.get("lf_i_neg_3_danger_z"), color=color),
            _signed_fmt(row.get("lf_collapse_3_danger_z"), color=color),
            _signed_fmt(row.get("lf_asymmetry_3_danger_z"), color=color),
            _signed_fmt(row.get("lf_velocity_3_danger_z"), color=color),
        ]
    )
    raw_z_text = "/".join(
        [
            _signed_fmt(row.get("lf_i_neg_3_danger_z_raw"), color=color),
            _signed_fmt(row.get("lf_collapse_3_danger_z_raw"), color=color),
            _signed_fmt(row.get("lf_asymmetry_3_danger_z_raw"), color=color),
            _signed_fmt(row.get("lf_velocity_3_danger_z_raw"), color=color),
        ]
    )
    norm_n_text = "/".join(
        [
            _fmt(row.get("lf_i_neg_3_danger_norm_n"), nd=0),
            _fmt(row.get("lf_collapse_3_danger_norm_n"), nd=0),
            _fmt(row.get("lf_asymmetry_3_danger_norm_n"), nd=0),
            _fmt(row.get("lf_velocity_3_danger_norm_n"), nd=0),
        ]
    )
    norm_scale_text = "/".join(
        [
            _fmt(row.get("lf_i_neg_3_danger_norm_mad_scale"), nd=4),
            _fmt(row.get("lf_collapse_3_danger_norm_mad_scale"), nd=4),
            _fmt(row.get("lf_asymmetry_3_danger_norm_mad_scale"), nd=4),
            _fmt(row.get("lf_velocity_3_danger_norm_mad_scale"), nd=4),
        ]
    )
    head = _color(f"[TFI {row.get('asset')} {row.get('price_source')}]", "bold", color)
    return (
        f"{head} {row.get('timestamp')} px={_fmt(row.get('canonical_perp_price'), nd=2)} "
        f"src={row.get('source_count')} disp={_fmt(row.get('cross_venue_dispersion_bps'), nd=2)}bp q={_fmt(row.get('data_quality_score'), nd=2)} | "
        f"risk={regime_text} pct={pct} tfi={tfi} | "
        f"I(-3) 6h/24h/7d/30d={_fmt(row.get('lf_i_neg_3_6h'), nd=3)}/"
        f"{_fmt(row.get('lf_i_neg_3_24h'), nd=3)}/{_fmt(row.get('lf_i_neg_3_7d'), nd=3)}/"
        f"{_fmt(row.get('lf_i_neg_3_30d'), nd=3)} | "
        f"collapse30-24={_signed_fmt(collapse, color=color)} {collapse_hint} "
        f"asym24={_signed_fmt(asym, color=color)} {asym_hint} "
        f"costVel15Raw={_signed_fmt(row.get('lf_cost3_velocity_15m_raw'), color=color)} "
        f"fragVel15={_signed_fmt(row.get('lf_cost3_velocity_15m'), color=color)} | "
        f"z(clipped) cost/coll/asym/vel={z_text} rawz={raw_z_text} "
        f"norm_n={norm_n_text} norm_scale={norm_scale_text}"
    )


def _fmt(v: Any, nd: int = 4) -> str:
    if not _finite(v):
        return "NA"
    return f"{float(v):.{nd}f}"


def _signed_fmt(v: Any, *, color: bool, nd: int = 3) -> str:
    if not _finite(v):
        return "NA"
    val = float(v)
    txt = f"{val:+.{nd}f}"
    if val >= 1.0:
        return _color(txt, "red", color)
    if val >= 0.5:
        return _color(txt, "yellow", color)
    if val <= -1.0:
        return _color(txt, "green", color)
    return txt


def _regime_text(regime: str, *, color: bool) -> str:
    name = str(regime or "unknown").upper()
    palette = {
        "NORMAL": "green",
        "ELEVATED": "yellow",
        "FRAGILE": "magenta",
        "SEVERE": "red",
        "EXTREME": "red_bg",
    }
    return _color(name, palette.get(name, "dim"), color)


def _collapse_hint(v: Any) -> str:
    if not _finite(v):
        return "(collapse NA)"
    return "(24h tail cheaper)" if float(v) > 0 else "(24h tail costlier)"


def _asymmetry_hint(v: Any) -> str:
    if not _finite(v):
        return "(asym NA)"
    return "(downside cheaper)" if float(v) > 0 else "(upside cheaper)"


def _color(text: str, name: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{ANSI.get(name, '')}{text}{ANSI['reset']}"
