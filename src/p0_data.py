"""P0 — data collection and instrument check. Nothing is tested.

Stage 1 of the plan (SPEC §10): fixing the universe and collecting data. Selection rule from
SPEC §2: an instrument stays in the working universe if it has ≥ 3 years
of daily data AND average turnover ≥ $10M/day. This is a descriptive check,
not a hypothesis — it does not spend trials (m=8, SPEC §3).

Additionally a bias warning is printed for BTC (trades 7 days a
week, daily bars do not match the exchange calendar) and for the HY-IG
spread (FRED, may be unavailable — then H3/H6 are partly BLOCKED, not
FAIL, see SPEC §9).
"""
from __future__ import annotations

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

START = "2015-01-01"   # with a margin before COVID (2020) for the walk-forward 2020-2023 train
END = "2026-09-13"

MIN_DAYS = 756          # ~3 years of trading days, SPEC §2
MIN_DOLLAR_VOL = 10e6    # $10M/day, SPEC §2

# ---------------------------------------------------------------- universe
UNIVERSE: dict[str, list[str]] = {
    "core_index": [
        "SPY", "^GSPC", "QQQ", "^NDX", "IWM", "EEM",
        "EWY", "FLKR", "FXI", "MCHI", "EWJ", "EWG",
    ],
    "sector_etf": [
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC",
        "SMH",
    ],
    "cross_market": [
        "^VIX", "^VVIX", "^VIX3M",
        "HYG", "LQD",
        "TLT", "IEF",
        "UUP",          # DXY is not a yfinance ticker directly; UUP as a dollar proxy
        "GLD", "USO",
        "BTC-USD",
    ],
    "leveraged_etf": [
        "TQQQ", "SQQQ", "SPXL", "SPXS", "SOXL", "SOXS",
    ],
}

FRED_SERIES = {
    "BAMLH0A0HYM2": "hy_oas",   # ICE BofA US High Yield OAS
    "BAMLC0A0CM": "ig_oas",     # ICE BofA US Corp Master OAS (IG)
}


def fetch_yfinance(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True,
                       progress=False, threads=True, timeout=120,
                       group_by="ticker")
    close, vol = {}, {}
    for t in tickers:
        try:
            sub = raw[t] if len(tickers) > 1 else raw
        except KeyError:
            continue
        if "Close" in sub:
            close[t] = sub["Close"]
        if "Volume" in sub:
            vol[t] = sub["Volume"]
    return pd.DataFrame(close), pd.DataFrame(vol)


def fetch_fred() -> pd.DataFrame:
    try:
        from pandas_datareader import data as pdr
        df = pdr.DataReader(list(FRED_SERIES), "fred", START, END)
        df = df.rename(columns=FRED_SERIES)
        return df
    except Exception as e:                                     # noqa: BLE001
        print(f"  FRED unavailable: {type(e).__name__}: {e}")
        return pd.DataFrame()


def main() -> int:
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 78)
    print("P0  DATA COLLECTION AND INSTRUMENT CHECK  (nothing is tested)")
    print("=" * 78)

    all_tickers = sorted({t for grp in UNIVERSE.values() for t in grp})
    print(f"  tickers to load: {len(all_tickers)}")
    close, vol = fetch_yfinance(all_tickers)
    close.to_csv(os.path.join(DATA, "p0_close.csv"))
    vol.to_csv(os.path.join(DATA, "p0_volume.csv"))

    have = [c for c in close.columns if close[c].dropna().shape[0] > 0]
    missing = sorted(set(all_tickers) - set(have))
    print(f"  series received: {len(have)}   empty: {len(missing)}"
          + (f"   {missing}" if missing else ""))

    print(f"\n  FRED (HY-IG spread)")
    fred = fetch_fred()
    if not fred.empty:
        fred.to_csv(os.path.join(DATA, "p0_fred.csv"))
        cov_start = fred.index.min().date()
        req_start = pd.Timestamp(START).date()
        print(f"    received {len(fred)} rows, {list(fred.columns)}, "
              f"{cov_start}..{fred.index.max().date()}")
        if cov_start > req_start:
            print(f"    WARNING: requested from {req_start}, actually from {cov_start}.")
            print(f"    The ICE BofA (BAML*) series were removed from FRED in 2022 over")
            print(f"    a licensing dispute and later reissued under the same IDs —")
            print(f"    history before the reissue is unavailable via the public CSV.")
            print(f"    Consequence: the walk-forward train 2020-2023 (SPEC §5.5) for H3")
            print(f"    does not fully cover 2020-2023 for HY-IG — the train period")
            print(f"    for H3 is effectively shortened to {cov_start}..2023-12-31,")
            print(f"    this is recorded as a data limitation, not hidden.")
    else:
        print(f"    FRED BLOCKED — H3/H6 partly cannot be tested "
              f"directly (SPEC §9: BLOCKED != FAIL)")

    # ---------------------------------------------------------- selection (SPEC §2)
    print(f"\n  INSTRUMENT SELECTION  (>= {MIN_DAYS} d., >= ${MIN_DOLLAR_VOL/1e6:.0f}M/day)")
    rows, dropped = [], []
    for grp, tickers in UNIVERSE.items():
        for t in tickers:
            if t not in close.columns or close[t].dropna().shape[0] == 0:
                dropped.append((grp, t, "no data"))
                continue
            s = close[t].dropna()
            n = len(s)
            if t in vol.columns:
                v = vol[t].reindex(s.index)
                dollar_vol = float((s * v).tail(60).mean())
            else:
                dollar_vol = float("nan")   # indices (^VIX etc.) have no volume — not filtered by volume
            is_index = t.startswith("^")
            ok_days = n >= MIN_DAYS
            ok_liq = is_index or (np.isfinite(dollar_vol) and dollar_vol >= MIN_DOLLAR_VOL)
            status = "pass" if (ok_days and ok_liq) else "FAIL"
            rows.append({
                "group": grp, "ticker": t, "n_days": n,
                "first": str(s.index[0].date()), "last": str(s.index[-1].date()),
                "dollar_vol_60d_musd": None if not np.isfinite(dollar_vol) else round(dollar_vol / 1e6, 1),
                "status": status,
            })
            if status == "FAIL":
                dropped.append((grp, t, f"n={n} liq_ok={ok_liq}"))

    tbl = pd.DataFrame(rows).sort_values(["group", "ticker"])
    tbl.to_csv(os.path.join(RESULTS, "p0_universe_check.csv"), index=False)
    print(tbl.to_string(index=False))

    passed = tbl[tbl.status == "pass"]
    print(f"\n  TOTAL: {len(passed)}/{len(tbl)} passed selection")
    if dropped:
        print(f"  dropped/problematic:")
        for grp, t, why in dropped:
            print(f"    {grp:14s} {t:10s} {why}")

    print(f"\n  NOTES ON BIASES:")
    print(f"    - BTC-USD trades 7d/wk, the daily bar closes at UTC midnight,")
    print(f"      it does not match the SPY/QQQ exchange calendar. Align to exchange")
    print(f"      trading dates — otherwise a false lead/lag in H7-like tests.")
    print(f"    - UUP is an ETF proxy for DXY (the dollar index itself is not a yfinance ticker),")
    print(f"      the correlation with DXY is high but not 1.0 — this is stated, not hidden.")
    print(f"    - ^VIX3M may be missing from some yfinance sources; if")
    print(f"      empty — H2 is partly BLOCKED, recorded explicitly, not hidden.")

    json.dump({
        "universe": UNIVERSE, "start": START, "end": END,
        "min_days": MIN_DAYS, "min_dollar_vol": MIN_DOLLAR_VOL,
        "n_passed": int(len(passed)), "n_total": int(len(tbl)),
        "fred_available": bool(not fred.empty),
    }, open(os.path.join(RESULTS, "p0_summary.json"), "w"), indent=1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
