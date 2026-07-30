#!/usr/bin/env bash
set -euo pipefail

# Reads derivatives_venue_metrics, computes LFI, prints latest rows, inserts nothing.
crypto-risk-index leverage \
  --dsn "${PM_DB_DSN:?set PM_DB_DSN}" \
  --asset "${ASSET:-BTC}" \
  --lookback-minutes "${LOOKBACK_MINUTES:-1800}" \
  --normalization-min-history "${NORMALIZATION_MIN_HISTORY:-50}" \
  --debug \
  --tail-rows "${TAIL_ROWS:-8}"
