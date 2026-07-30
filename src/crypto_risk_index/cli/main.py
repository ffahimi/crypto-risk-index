from __future__ import annotations

import argparse
import json
import os
import time
import uuid

import pandas as pd

from crypto_risk_index.io import connect, load_derivatives_venue_metrics
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


def summarize_leverage_health(raw: pd.DataFrame, out: pd.DataFrame) -> dict:
    now = pd.Timestamp.utcnow()
    latest_ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").max() if "timestamp" in out else pd.NaT
    latest_age_s = float((now - latest_ts).total_seconds()) if pd.notna(latest_ts) else float("nan")
    latest = out.sort_values("timestamp").tail(1).to_dict("records")
    latest_row = latest[0] if latest else {}
    health_cols = [
        "canonical_perp_price",
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
    ]
    pct = {}
    for col in health_cols:
        pct[col] = 100.0 * float(out[col].notna().mean()) if col in out and len(out) else float("nan")
    return {
        "cycle_ts": now.isoformat(),
        "raw_rows": int(len(raw)),
        "output_rows": int(len(out)),
        "latest_ts": latest_ts.isoformat() if pd.notna(latest_ts) else None,
        "latest_age_s": latest_age_s,
        "latest_source_count": latest_row.get("source_count"),
        "latest_long_fragility": latest_row.get("long_fragility_score_0_100"),
        "latest_short_fragility": latest_row.get("short_fragility_score_0_100"),
        "latest_direction": latest_row.get("leverage_fragility_direction"),
        "latest_long_state": latest_row.get("long_state"),
        "latest_short_state": latest_row.get("short_state"),
        "nonnull_pct": pct,
    }


def print_leverage_health(summary: dict) -> None:
    pct = summary["nonnull_pct"]
    print(
        "[health LFI] "
        f"raw={summary['raw_rows']:,} out={summary['output_rows']:,} "
        f"latest={summary['latest_ts']} age={fmt(summary['latest_age_s'], 1)}s "
        f"src={summary['latest_source_count']} "
        f"fragL/S={fmt(summary['latest_long_fragility'], 1)}/{fmt(summary['latest_short_fragility'], 1)} "
        f"dir={summary['latest_direction']} state={summary['latest_long_state']}/{summary['latest_short_state']} | "
        f"nonnull px/funding/basis/oi/dOI/frag="
        f"{fmt(pct.get('canonical_perp_price'), 1)}/"
        f"{fmt(pct.get('funding_composite_8h'), 1)}/"
        f"{fmt(pct.get('basis_composite_bps'), 1)}/"
        f"{fmt(pct.get('open_interest_composite_usd'), 1)}/"
        f"{fmt(pct.get('open_interest_pct_change_60m'), 1)}/"
        f"{fmt(pct.get('long_fragility_score_0_100'), 1)}%"
    )


def create_temp_leverage_health_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            create temporary table if not exists crypto_risk_index_lfi_health_debug (
                run_id text,
                cycle_n integer,
                cycle_ts timestamptz,
                raw_rows integer,
                output_rows integer,
                latest_ts timestamptz,
                latest_age_s double precision,
                latest_source_count double precision,
                latest_long_fragility double precision,
                latest_short_fragility double precision,
                latest_direction text,
                latest_long_state text,
                latest_short_state text,
                nonnull_pct jsonb
            ) on commit preserve rows
            """
        )
    conn.commit()


def insert_temp_leverage_health(conn, *, run_id: str, cycle_n: int, summary: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into crypto_risk_index_lfi_health_debug (
                run_id, cycle_n, cycle_ts, raw_rows, output_rows, latest_ts, latest_age_s,
                latest_source_count, latest_long_fragility, latest_short_fragility,
                latest_direction, latest_long_state, latest_short_state, nonnull_pct
            ) values (
                %(run_id)s, %(cycle_n)s, %(cycle_ts)s, %(raw_rows)s, %(output_rows)s,
                %(latest_ts)s, %(latest_age_s)s, %(latest_source_count)s,
                %(latest_long_fragility)s, %(latest_short_fragility)s,
                %(latest_direction)s, %(latest_long_state)s, %(latest_short_state)s,
                %(nonnull_pct)s::jsonb
            )
            """,
            {
                **summary,
                "run_id": run_id,
                "cycle_n": cycle_n,
                "nonnull_pct": json.dumps(summary["nonnull_pct"]),
            },
        )
    conn.commit()


def run_leverage(args: argparse.Namespace) -> None:
    cfg = LFIConfig(asset=args.asset, normalization_lookback=args.normalization_lookback, normalization_min_history=args.normalization_min_history)
    run_id = str(uuid.uuid4())[:8]
    temp_conn = connect(args.dsn or os.getenv("PM_DB_DSN")) if args.temp_log else None
    if temp_conn:
        create_temp_leverage_health_table(temp_conn)
        print(f"[temp-log] table=crypto_risk_index_lfi_health_debug run_id={run_id} permanent=False")
    try:
        for cycle_n in range(1, args.repeat + 1):
            if args.csv:
                raw = pd.read_csv(args.csv)
            else:
                raw = load_derivatives_venue_metrics(dsn=args.dsn or os.getenv("PM_DB_DSN"), asset=args.asset, lookback_minutes=args.lookback_minutes, table=args.table)
            print(f"[load] cycle={cycle_n}/{args.repeat} leverage raw rows={len(raw):,} asset={args.asset} source={'csv' if args.csv else args.table}")
            out = compute_leverage_fragility(raw, cfg)
            print(f"[compute] leverage output rows={len(out):,} debug_only=True inserts=0")
            summary = summarize_leverage_health(raw, out)
            print_leverage_health(summary)
            print_leverage_debug(out, tail_rows=args.tail_rows)
            if temp_conn:
                insert_temp_leverage_health(temp_conn, run_id=run_id, cycle_n=cycle_n, summary=summary)
                print(f"[temp-log] inserted cycle={cycle_n} run_id={run_id}")
            if args.out_csv:
                stem = args.out_csv
                if args.repeat > 1:
                    stem = stem.replace(".csv", f"_cycle{cycle_n}.csv")
                out.to_csv(stem, index=False)
                print(f"[saved] {stem}")
            if cycle_n < args.repeat:
                time.sleep(args.sleep_s)
    finally:
        if temp_conn:
            temp_conn.close()


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
    lev.add_argument("--repeat", type=int, default=1, help="Number of debug cycles to run.")
    lev.add_argument("--sleep-s", type=float, default=60.0, help="Seconds between repeated debug cycles.")
    lev.add_argument("--temp-log", action="store_true", help="Write cycle health rows to a session-local temporary table.")
    lev.set_defaults(func=run_leverage)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
