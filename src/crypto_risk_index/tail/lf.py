from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class SurfaceConfig:
    name: str
    return_col: str
    lookback_minutes: int
    min_observations: int
    return_resolution_minutes: int


@dataclass(frozen=True)
class RateSurface:
    window_name: str
    return_col: str
    sample_count: int
    location: float
    scale: float
    clipped_count: int
    clipped_pct: float
    x_grid: np.ndarray
    theta_grid: np.ndarray
    rate_values: np.ndarray
    theta_star: np.ndarray
    theta_boundary_count: int
    support_warning_count: int
    convexity_min_second_diff: float
    min_rate_value: float


DEFAULT_SURFACES = (
    SurfaceConfig("6h", "return_1m", 6 * 60, 300, 1),
    SurfaceConfig("24h", "return_1m", 24 * 60, 1200, 1),
    SurfaceConfig("7d", "return_5m", 7 * 24 * 60, 1500, 5),
    SurfaceConfig("30d", "return_1h", 30 * 24 * 60, 600, 60),
)


def parse_float_grid(raw: str) -> np.ndarray:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("grid must contain at least one value")
    return np.asarray(vals, dtype=float)


def theta_grid(theta_min: float = -6.0, theta_max: float = 6.0, theta_step: float = 0.02) -> np.ndarray:
    n = int(round((theta_max - theta_min) / theta_step))
    return np.linspace(theta_min, theta_max, n + 1, dtype=float)


def default_x_grid() -> np.ndarray:
    return np.asarray(
        [-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
         0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        dtype=float,
    )


def robust_location_scale(values: np.ndarray, scale_floor: float = 1e-6) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return np.nan, np.nan
    loc = float(np.median(clean))
    mad = float(np.median(np.abs(clean - loc)))
    scale = max(1.4826 * mad, scale_floor)
    return loc, scale


def mean_std_location_scale(values: np.ndarray, scale_floor: float = 1e-6) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return np.nan, np.nan
    loc = float(np.mean(clean))
    scale = max(float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0, scale_floor)
    return loc, scale


def logsumexp(a: np.ndarray, axis: int = 0) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def empirical_cgf(z: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    if z.size == 0:
        return np.full_like(thetas, np.nan, dtype=float)
    vals = np.outer(thetas, z)
    return logsumexp(vals, axis=1) - np.log(float(z.size))


def legendre_rate_curve(
    returns: Iterable[float],
    *,
    window_name: str,
    return_col: str,
    x_values: np.ndarray,
    thetas: np.ndarray,
    standardization: str = "robust",
    scale_floor: float = 1e-6,
    z_clip: float = 12.0,
    tolerance: float = 1e-9,
) -> RateSurface:
    raw = np.asarray(list(returns), dtype=float)
    raw = raw[np.isfinite(raw)]
    if standardization == "mean_std":
        loc, scale = mean_std_location_scale(raw, scale_floor=scale_floor)
    else:
        loc, scale = robust_location_scale(raw, scale_floor=scale_floor)
    if raw.size == 0 or not np.isfinite(loc) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("cannot compute rate curve without finite returns and scale")

    z = (raw - loc) / (scale + EPS)
    clipped = np.clip(z, -z_clip, z_clip)
    clipped_count = int(np.count_nonzero(np.abs(z) > z_clip))
    lam = empirical_cgf(clipped, thetas)
    objective = np.outer(x_values, thetas) - lam[None, :]
    arg = np.argmax(objective, axis=1)
    rate = objective[np.arange(len(x_values)), arg]
    theta_star = thetas[arg]

    tiny_neg = (rate < 0) & (rate >= -tolerance)
    if np.any(tiny_neg):
        rate = rate.copy()
        rate[tiny_neg] = 0.0

    second_diff = np.diff(rate, n=2)
    min_second = float(np.nanmin(second_diff)) if second_diff.size else np.nan
    theta_boundary_count = int(np.count_nonzero((arg == 0) | (arg == len(thetas) - 1)))

    # Lambda'(theta) is the tilted sample mean. If x is outside this range, the
    # finite theta grid is probably truncating the supremum.
    vals = np.outer(thetas, clipped)
    row_max = np.max(vals, axis=1, keepdims=True)
    weights = np.exp(vals - row_max)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    deriv = weights @ clipped
    support_min = float(np.nanmin(deriv))
    support_max = float(np.nanmax(deriv))
    support_warning_count = int(np.count_nonzero((x_values < support_min - 1e-6) | (x_values > support_max + 1e-6)))

    return RateSurface(
        window_name=window_name,
        return_col=return_col,
        sample_count=int(raw.size),
        location=float(loc),
        scale=float(scale),
        clipped_count=clipped_count,
        clipped_pct=float(clipped_count / max(raw.size, 1)),
        x_grid=np.asarray(x_values, dtype=float),
        theta_grid=np.asarray(thetas, dtype=float),
        rate_values=np.asarray(rate, dtype=float),
        theta_star=np.asarray(theta_star, dtype=float),
        theta_boundary_count=theta_boundary_count,
        support_warning_count=support_warning_count,
        convexity_min_second_diff=min_second,
        min_rate_value=float(np.nanmin(rate)),
    )


def value_at(surface: RateSurface, x: float) -> float:
    idx = np.where(np.isclose(surface.x_grid, x))[0]
    if idx.size == 0:
        return np.nan
    return float(surface.rate_values[int(idx[0])])


def theta_at(surface: RateSurface, x: float) -> float:
    idx = np.where(np.isclose(surface.x_grid, x))[0]
    if idx.size == 0:
        return np.nan
    return float(surface.theta_star[int(idx[0])])


def left_area(surface: RateSurface) -> float:
    xs = np.asarray([-4.0, -3.5, -3.0, -2.5, -2.0, -1.5], dtype=float)
    vals = np.asarray([value_at(surface, x) for x in xs], dtype=float)
    if np.any(~np.isfinite(vals)):
        return np.nan
    return float(np.trapezoid(vals, xs))
