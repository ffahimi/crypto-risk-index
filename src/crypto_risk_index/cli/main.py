from __future__ import annotations

import argparse
import os

import pandas as pd

from crypto_risk_index.io import load_derivatives_venue_metrics
from crypto_risk_index.leverage import LFIConfig, compute_leverage_fragility


def fmt(value, nd: int = 3) -> str:
    try:
        x = float(value)
        if pd.notna(x):
            return f"{x:.{nd}f}"
    except Exception:
        pass
    return "NA"


def print_leverage_debug(rows: pd.DataFrame, *, tail_rows: int) -> None:
    if rows.empty:
        print("[crypto-risk-index] no leverage rows")
        return
    cols = [
        "timestamp",
        "canonical_perp_price",
        "source_count",
        "funding_composite_8h",
        "basis_composite_bps",
        "open_interest_composite_usd",
        "open_interest_pct_change_60m",
        "long_crowding_percentile",
        "short_crowding_percentile",
        "long_unwind_pressure_percentile",
        "short_squeeze_pressure_percentile",
        "long_fragility_score_0_100",
        "short_fragility_score_0_100",
        "leverage_fragility_direction",
        "long_state",
        "short_state",
    ]
    shown = rows.tail(tail_rows)
    for r in shown.to_dict("records"):
        print(
            "[LFI] "
            f"ts={r.get('timestamp')} px={fmt(r.get('canonical_perp_price'), 2)} src={r.get('source_count')} "
            f"funding8h={fmt(r.get('funding_composite_8h'), 6)} basis={fmt(r.get('basis_composite_bps'), 2)}bp "
            f"oi=${fmt(r.get('open_interest_composite_usd'), 0)} dOI1h={fmt(r.get('open_interest_pct_change_60m'), 5)} | "
            f"crowd L/S={fmt(r.get('long_crowding_percentile'), 1)}/{fmt(r.get('short_crowding_percentile'), 1)} "
            f"trigger L/S={fmt(r.get('long_unwind_pressure_percentile'), 1)}/{fmt(r.get('short_squeeze_pressure_percentile'), 1)} | "
            f"frag L/S={fmt(r.get('long_fragility_score_0_100'), 1)}/{fmt(r.get('short_fragility_score_0_100'), 1)} "
            f"dir={r.get('leverage_fragility_direction')} state={r.get('long_state')}/{r.get('short_state')}"
        )
    print("\n[latest table]")
    print(shown[[c for c in cols if c in shown]].to_string(index=False))


def run_leverage(args: argparse.Namespace) -> None:
    if args.csv:
        raw = pd.read_csv(args.csv)
    else:
        raw = load_derivatives_venue_metrics(
            dsn=args.dsn or os.getenv("PM_DB_DSN"),
            asset=args.asset,
            lookback_minutes=args.lookback_minutes,
            table=args.table,
        )
    print(f"[load] leverage raw rows={len(raw):,} asset={args.asset} source={'csv' if args.csv else args.table}")
    cfg = LFIConfig(
        asset=args.asset,
        normalization_lookback=args.normalization_lookback,
        normalization_min_history=args.normalization_min_history,
    )
    out = compute_leverage_fragility(raw, cfg)
    print(f"[compute] leverage output rows={len(out):,} debug_only=True inserts=0")
    print_leverage_debug(out, tail_rows=args.tail_rows)
    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"[saved] {args.out_csv}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="crypto-risk-index")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lev = sub.add_parser("leverage", help="Compute leverage fragility from derivatives venue metrics.")
    lev.add_argument("--dsn", default="")
    lev.add_argument("--asset", default="BTC")
    lev.add_argument("--table", default="derivatives_venue_metrics")
    lev.add_argument("--lookback-minutes", type=int, default=1800)
    lev.add_argument("--normalization-lookback", type=int, default=500)
    lev.add_argument("--normalization-min-history", type=int, default=50)
    lev.add_argument("--tail-rows", type=int, default=8)
    lev.add_argument("--csv", default="", help="Optional CSV input instead of Postgres.")
    lev.add_argument("--out-csv", default="", help="Optional path to save output CSV.")
    lev.add_argument("--debug", action="store_true", help="Print results only. This package does not insert rows.")
    lev.set_defaults(func=run_leverage)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
