from __future__ import annotations

import os
import warnings

import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable.*",
    category=UserWarning,
)


def connect(dsn: str | None = None):
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("Install postgres support with: pip install 'crypto-risk-index[postgres]'") from exc
    raw = dsn or os.getenv("PM_DB_DSN") or os.getenv("DATABASE_URL")
    if not raw:
        raise ValueError("missing dsn, PM_DB_DSN, or DATABASE_URL")
    cleaned = str(raw).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1]
    return psycopg2.connect(cleaned)


def load_derivatives_venue_metrics(
    *,
    dsn: str | None = None,
    asset: str = "BTC",
    table: str = "derivatives_venue_metrics",
    lookback_minutes: int = 1800,
) -> pd.DataFrame:
    sql = f"""
        select *
        from public.{table}
        where asset = %(asset)s
          and timestamp >= now() - (%(lookback_minutes)s || ' minutes')::interval
        order by timestamp asc, venue asc
    """
    with connect(dsn) as conn:
        return pd.read_sql(sql, conn, params={"asset": asset.upper(), "lookback_minutes": int(lookback_minutes)})
