"""P2 — H4: breadth (share of sectors above their 50-d MA) as a signal.

SPEC §3 H4: "The share of sectors above their 50-day MA falls below 30% before
SPX shows a regime change. Metric: lead time."

The SPEC itself (the framing at the start of the document) classifies breadth as
"confirmation, not a lead" — i.e. the pre-registered expectation is
mixed: the letter of H4 asks for lead time > 0, but the spec's own literature
framing predicts lead time <= 0 (or ~0). Both variants are fixed
BEFORE looking at the result, so that whatever comes out — it is not fitting.

The "true" episodes are the same as in H1 (VIX ground truth, results/p2_h1_summary.json)
the START of the episode is used rather than the end date. Sector ETFs have their own
trading calendar, so dates are matched by date string, not by
position number from H1.
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

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH"]
MA_WIN = 50
BREADTH_THRESH = 0.30
LOOKBACK = 20      # days BEFORE onset in which to look for a breadth<30% crossing
N_PERM = 5000


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 78)
    print("P2  H4 — BREADTH (share of sectors above MA50) AS A SIGNAL")
    print("=" * 78)

    close_all = pd.read_csv(os.path.join(DATA, "p0_close.csv"), index_col=0, parse_dates=True)
    # p0_close.csv is indexed by the union of dates of ALL 40 tickers, including
    # BTC-USD (trades 7 days a week) — on weekends all sectors are NaN, and
    # a naive `NaN > MA` would give False -> a false "below MA" every weekend.
    close = close_all[SECTORS].dropna(how="all")     # only real sector trading days
    ma = close.rolling(MA_WIN).mean()
    above = (close > ma)
    n_avail = close.notna().sum(axis=1)               # XLC (from 2018), XLRE (from 2015-10) — not always 11
    breadth = (above.sum(axis=1) / n_avail).dropna()
    breadth = breadth[n_avail.reindex(breadth.index) >= 5]   # no computation without a reliable denominator
    print(f"  sectors: {len(SECTORS)} (not all available from the start — XLC from 2018, XLRE from 2015-10)")
    print(f"  breadth period: {breadth.index[0].date()}..{breadth.index[-1].date()}   n={len(breadth)}")
    print(f"  share of time breadth < {BREADTH_THRESH:.0%}: {(breadth < BREADTH_THRESH).mean():.1%}")

    below = (breadth < BREADTH_THRESH).to_numpy()
    idx = breadth.index

    h1_summary = json.load(open(os.path.join(RESULTS, "p2_h1_summary.json")))
    episodes = h1_summary["episodes"]
    print(f"\n  episodes from H1 (VIX ground truth): {len(episodes)}")

    rows = []
    for e in episodes:
        onset = pd.Timestamp(e["onset_date"])
        pos = idx.searchsorted(onset)
        if pos >= len(idx) or idx[pos] != onset:
            pos = min(pos, len(idx) - 1)
        window = below[max(0, pos - LOOKBACK):pos + 1]
        cross = np.where(window)[0]
        if len(cross):
            lead = int(len(window) - 1 - cross[-1])  # days BEFORE onset when breadth last entered <30%
        else:
            lead = None
        rows.append({"onset_date": e["onset_date"], "onset_pos": int(pos),
                     "lead_days": lead, "breadth_at_onset": float(breadth.iloc[pos])})
        print(f"    {e['onset_date']}: breadth={rows[-1]['breadth_at_onset']:.0%}"
              f"   lead={'none in the ' + str(LOOKBACK) + 'd window' if lead is None else f'{lead}d before onset'}")

    leads = [r["lead_days"] for r in rows if r["lead_days"] is not None]
    n_found = len(leads)
    print(f"\n  breadth<{BREADTH_THRESH:.0%} found in the {LOOKBACK}-day window BEFORE onset:"
          f" {n_found}/{len(rows)} episodes")
    if leads:
        print(f"  median lead time: {np.median(leads):.1f} d."
              f"   (>0 = breadth led; 0 = coincided with onset)")

    # ------------------------------------------------ significance (permutation)
    onset_positions = [r["onset_pos"] for r in rows]

    def observed_stat(flag_arr: np.ndarray) -> float:
        hits = []
        for pos in onset_positions:
            w = flag_arr[max(0, pos - LOOKBACK):pos + 1]
            hits.append(1.0 if np.any(w) else 0.0)
        return float(np.mean(hits)) if hits else float("nan")

    obs = observed_stat(below)
    rng = np.random.default_rng(0)
    n = len(below)
    count = 0
    for _ in range(N_PERM):
        shifted = np.roll(below, int(rng.integers(1, n)))
        if observed_stat(shifted) >= obs - 1e-9:
            count += 1
    p = (count + 1) / (N_PERM + 1)
    print(f"\n  SIGNIFICANCE (perm, H0: breadth<30% unrelated to episode dates, N={N_PERM}):")
    print(f"    share of episodes with a signal in the {LOOKBACK}d window: {obs:.1%}   perm p={p:.4f}"
          f"  <- H4 trial")

    from scipy import stats as st
    z = float(st.norm.ppf(1 - p)) if p < 1.0 else 0.0
    trial = [{"n": 4, "name": "H4 breadth<30% lead", "b": obs, "t": z, "p": p,
              "mde": float("nan")}]

    print(f"\n  INTERPRETATION: the SPEC framing predicted breadth as CONFIRMATION, not")
    print(f"  a lead. {'There is a statistical link' if p < 0.05 else 'No significant link found'},"
          f" median lead={np.median(leads) if leads else float('nan'):.1f}d"
          f" {'— looks like a confirming rather than a leading signal.' if leads and np.median(leads) <= 1 else ''}")

    json.dump({"episodes": rows, "n_found": n_found, "n_total": len(rows),
               "median_lead": float(np.median(leads)) if leads else None,
               "perm_p": p, "trial": trial,
               "params": {"MA_WIN": MA_WIN, "BREADTH_THRESH": BREADTH_THRESH, "LOOKBACK": LOOKBACK}},
              open(os.path.join(RESULTS, "p2_h4_summary.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(RESULTS, 'p2_h4_summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
