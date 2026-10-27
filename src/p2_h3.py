"""P2 — H3: HY-IG spread widening leads SPX declines.

SPEC §3 H3: "HY-IG spread widening significantly precedes
SPX declines ... at a 1-10 day horizon. Test: lagged cross-correlation."

STATUS: BLOCKED, not FAIL (reclassified after review). FRED provides
BAMLH0A0HYM2/BAMLC0A0CM only from 2023-09-12 (ICE BofA removed the series from FRED in
2022 over a licensing dispute and reissued them later). Of the 11 H1 episodes (VIX
ground truth) ONLY 2025-04-03 falls into this window (1 of 11) — COVID
(VIX 82.7, the largest episode) and all 6 episodes of 2022 are OUTSIDE the sample. The regression
below technically runs and is printed as a diagnostic (not as the deciding
H3 test): with n=750, almost all of it the calm 2023-2026 period, the test
structurally can neither confirm nor refute "credit leads equity" —
it never saw a real widening of the credit spread. FAIL here would
be wrong: FAIL means "the hypothesis was tested and not confirmed", not
"the instrument never saw the phenomenon it is supposed to detect".

Primary specification (for reference, NOT part of the m=8 family because of BLOCKED):
  y = SPY return (cumulative) t+1..t+h,  h = 10 (the upper bound)
  X = ΔSpread_t (daily change in the HY-IG spread), control rm_t (SPY return
      on day t — removes the trivial momentum/reversal confound)
  expected sign: β(ΔSpread) < 0 (widening -> decline).
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
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

HORIZONS = list(range(1, 11))
PRIMARY_H = 10
HAC_LAGS = 15


def hac_fit(y, X):
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 78)
    print("P2  H3 — THE HY-IG SPREAD LEADS SPX")
    print("=" * 78)

    fred_path = os.path.join(DATA, "p0_fred.csv")
    if not os.path.exists(fred_path):
        print("  No FRED data (p0_fred.csv missing) — H3 entirely BLOCKED.")
        json.dump({"status": "BLOCKED", "reason": "no FRED data"},
                  open(os.path.join(RESULTS, "p2_h3_summary.json"), "w"), indent=1)
        return 0

    fred = pd.read_csv(fred_path, index_col=0, parse_dates=True)
    spread = (fred["hy_oas"] - fred["ig_oas"]).dropna()
    close = pd.read_csv(os.path.join(DATA, "p0_close.csv"), index_col=0, parse_dates=True)
    spy = close["SPY"].dropna()
    ret = spy.pct_change()

    common = spread.index.intersection(ret.index)
    print(f"  COVERAGE: {common.min().date()}..{common.max().date()}   n={len(common)}")
    print(f"  (COVID-2020 and most of the 2022 bear market are OUTSIDE the sample — see docstring)")

    df = pd.DataFrame({"spread": spread.reindex(common), "rm": ret.reindex(common)}).dropna()
    df["d_spread"] = df["spread"].diff()
    for h in HORIZONS:
        df[f"fwd_ret{h}"] = spy.reindex(common).pct_change(h).shift(-h).reindex(df.index)
    df = df.dropna(subset=["d_spread"])
    print(f"  observations after diff: {len(df)}")

    print(f"\n  --- MDE (SE x |t|=2, reference) ---")
    sub0 = df.dropna(subset=[f"fwd_ret{PRIMARY_H}"])
    m0 = hac_fit(sub0[f"fwd_ret{PRIMARY_H}"], sub0[["d_spread", "rm"]])
    se0 = float(m0.bse["d_spread"])
    print(f"    h={PRIMARY_H}: n={int(m0.nobs)}  SE(d_spread)={se0:.5f}  MDE={2 * se0:.5f}")

    # how many of the 11 VIX episodes (H1) actually fall into the FRED window
    h1_path = os.path.join(RESULTS, "p2_h1_summary.json")
    n_covered, n_total_episodes, covered_dates = 0, None, []
    if os.path.exists(h1_path):
        eps = json.load(open(h1_path))["episodes"]
        n_total_episodes = len(eps)
        for e in eps:
            d = pd.Timestamp(e["onset_date"])
            if common.min() <= d <= common.max():
                n_covered += 1
                covered_dates.append(e["onset_date"])

    print(f"\n  --- RESULTS, DIAGNOSTIC (HAC, maxlags={HAC_LAGS}) — not the H3 trial ---")
    print(f"  {'h':>3s} {'n':>5s} {'coef':>10s} {'t':>7s} {'p':>9s}")
    diag = []
    for h in HORIZONS:
        sub = df.dropna(subset=[f"fwd_ret{h}"])
        m = hac_fit(sub[f"fwd_ret{h}"], sub[["d_spread", "rm"]])
        b, t, p = (float(m.params["d_spread"]), float(m.tvalues["d_spread"]),
                   float(m.pvalues["d_spread"]))
        print(f"  {h:3d} {int(m.nobs):5d} {b:+10.4f} {t:+7.2f} {p:9.4f}  diagnostic")
        diag.append({"h": h, "n": int(m.nobs), "b": b, "t": t, "p": p})

    print(f"\n  STATUS: BLOCKED (reclassified after review, not FAIL).")
    if n_total_episodes:
        print(f"  of {n_total_episodes} H1 episodes (VIX ground truth), the FRED window"
              f" {common.min().date()}..{common.max().date()} contains {n_covered}"
              f" ({', '.join(covered_dates) if covered_dates else '-'})")
    print(f"  COVID-2020 (the largest episode, VIX 82.7) and all of 2022 are outside the sample.")
    print(f"  The test structurally could not see the phenomenon it is supposed to detect —")
    print(f"  this makes the diagnostic above uninformative for FDR, not a 'weak signal'.")

    json.dump({"status": "BLOCKED", "reason": "FRED history starts 2023-09-12; "
              f"only {n_covered}/{n_total_episodes if n_total_episodes else '?'} VIX episodes covered, "
              "COVID-2020 and all of 2022 excluded",
              "diagnostic_not_trial": diag,
              "n_episodes_covered": n_covered, "n_episodes_total": n_total_episodes,
              "coverage": [str(common.min().date()), str(common.max().date())],
              "params": {"HAC_LAGS": HAC_LAGS, "PRIMARY_H": PRIMARY_H}},
              open(os.path.join(RESULTS, "p2_h3_summary.json"), "w"), indent=1)
    df.to_csv(os.path.join(RESULTS, "p2_h3_panel.csv"))
    print(f"\n  wrote {os.path.join(RESULTS, 'p2_h3_summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
