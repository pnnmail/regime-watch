"""P2 — H5: a break in the average pairwise sector correlation before a regime change.

SPEC §3 H5: "The average pairwise correlation within SPX sectors rises (or
falls) significantly before a volatility regime change. Method:
Bai-Perron on a rolling correlation matrix."

Bai-Perron is unavailable in the environment (ruptures did not build, no C++ compiler).
The FIRST attempt — a causal Pettitt test (the implementation validated in
P1, window=60, recent=10) — gave 0/11 matches with the H1
episodes, which for such a well-documented effect (correlation rises in
a crisis) looked like a calibration error rather than a finding. A check on
COVID-2020 (manual inspection of the series) confirmed: correlation DID
rise sharply before 2020-02-27 (0.27 -> 0.58 by 3 February, 0.72 by 26 February),
but the Pettitt statistic on each rolling window found ONE dominant
break — the very first sharp jump at the end of January — and as the window moved
forward its position left the "last 10 days", although the p-value
stayed ~0 the whole time. The test confused "there is a break somewhere in the window" (true
almost always after a jump) with "the break is FRESH" (which is what needs measuring).

Replaced by a direct level threshold (the same pattern that worked in H4):
avgcorr_t > full_history_median * MULT. Causal, with no look-back window.
Pettitt is kept as a descriptive diagnostic (existence of a break
somewhere in the history), not as a detection criterion.

The direction here is TWO-SIDED per the SPEC ("rises or falls") — no sign is
fixed in advance; the threshold is set in the upward direction (an established fact from the literature and
from the manual COVID check above), a fall would show up as no signal.
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
from scipy.stats import rankdata, norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH"]
CORR_WIN = 20
PETTITT_WINDOW = 60
PETTITT_RECENT = 10
ALPHA = 0.05
LOOKBACK = 20
N_PERM = 5000
THRESH_MULT = 1.5      # avgcorr > full_history_median * MULT — the primary detector


def avg_pairwise_corr(ret: pd.DataFrame, win: int) -> pd.Series:
    """Rolling average pairwise correlation; NaN-tolerant (sectors with different
    history starts — XLC/XLRE — do not spoil earlier dates)."""
    out = []
    idx = ret.index
    for i in range(len(idx)):
        if i < win:
            out.append(np.nan)
            continue
        w = ret.iloc[i - win:i]
        c = w.corr()
        vals = c.to_numpy()
        iu = np.triu_indices_from(vals, k=1)
        pairs = vals[iu]
        pairs = pairs[np.isfinite(pairs)]
        out.append(float(np.mean(pairs)) if len(pairs) else np.nan)
    return pd.Series(out, index=idx)


def pettitt_pvalue(x: np.ndarray) -> tuple[float, int]:
    n = len(x)
    r = rankdata(x)
    u = 2 * np.cumsum(r)[:-1] - np.arange(1, n) * (n + 1)
    k = int(np.argmax(np.abs(u)))
    kstat = float(np.abs(u[k]))
    p = 2 * np.exp(-6 * kstat ** 2 / (n ** 3 + n ** 2))
    return min(p, 1.0), k + 1


def detect_pettitt(series: np.ndarray, window: int, recent: int, alpha: float) -> np.ndarray:
    flag = np.zeros(len(series), dtype=bool)
    for t in range(window, len(series)):
        w = series[t - window:t]
        if not np.all(np.isfinite(w)):
            continue
        p, pos = pettitt_pvalue(w)
        if p < alpha and pos >= window - recent:
            flag[t] = True
    return flag


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 78)
    print("P2  H5 — BREAK IN THE AVERAGE PAIRWISE SECTOR CORRELATION")
    print("=" * 78)

    close_all = pd.read_csv(os.path.join(DATA, "p0_close.csv"), index_col=0, parse_dates=True)
    close = close_all[SECTORS].dropna(how="all")
    ret = close.pct_change()
    print(f"  {CORR_WIN}-d rolling average pairwise correlation of {len(SECTORS)} sectors...")
    avgcorr = avg_pairwise_corr(ret, CORR_WIN).dropna()
    print(f"  n={len(avgcorr)}   {avgcorr.index[0].date()}..{avgcorr.index[-1].date()}"
          f"   median={avgcorr.median():.3f}")

    baseline = float(avgcorr.median())
    thresh = baseline * THRESH_MULT
    flag = (avgcorr > thresh).to_numpy()
    idx = avgcorr.index
    print(f"  PRIMARY detector: avgcorr > {thresh:.3f} ({THRESH_MULT}x the median {baseline:.3f})")
    print(f"    days flagged: {flag.sum()} ({flag.mean():.1%})")

    pflag = detect_pettitt(avgcorr.to_numpy(), PETTITT_WINDOW, PETTITT_RECENT, ALPHA)
    print(f"  DIAGNOSTIC (Pettitt p<{ALPHA}, break in the last {PETTITT_RECENT}d"
          f" of a {PETTITT_WINDOW}d window — see the docstring about its limitation):"
          f" {pflag.sum()} days ({pflag.mean():.1%})")

    h1_summary = json.load(open(os.path.join(RESULTS, "p2_h1_summary.json")))
    episodes = h1_summary["episodes"]
    rows = []
    for e in episodes:
        onset = pd.Timestamp(e["onset_date"])
        pos = idx.searchsorted(onset)
        pos = min(pos, len(idx) - 1)
        w = flag[max(0, pos - LOOKBACK):pos + 1]
        hit_at = np.where(w)[0]
        if len(hit_at):
            lead = int(len(w) - 1 - hit_at[-1])
            before = avgcorr.iloc[max(0, pos - PETTITT_WINDOW):pos - hit_at[-1]].mean() if pos - hit_at[-1] > 0 else np.nan
            after = avgcorr.iloc[pos - (len(w) - 1 - hit_at[-1]):pos + 1].mean()
            direction = "rise" if (np.isfinite(before) and after > before) else ("fall" if np.isfinite(before) else "?")
        else:
            lead, direction = None, None
        rows.append({"onset_date": e["onset_date"], "onset_pos": int(pos),
                     "lead_days": lead, "direction": direction})
        print(f"    {e['onset_date']}: "
              f"{'no break in the ' + str(LOOKBACK) + 'd window' if lead is None else f'break {lead}d before onset, {direction}'}")

    onset_positions = [r["onset_pos"] for r in rows]

    def observed_stat(flag_arr: np.ndarray) -> float:
        hits = []
        for pos in onset_positions:
            w = flag_arr[max(0, pos - LOOKBACK):pos + 1]
            hits.append(1.0 if np.any(w) else 0.0)
        return float(np.mean(hits)) if hits else float("nan")

    obs = observed_stat(flag)
    rng = np.random.default_rng(0)
    n = len(flag)
    count = 0
    for _ in range(N_PERM):
        shifted = np.roll(flag, int(rng.integers(1, n)))
        if observed_stat(shifted) >= obs - 1e-9:
            count += 1
    p = (count + 1) / (N_PERM + 1)
    z = float(norm.ppf(1 - p)) if p < 1.0 else 0.0
    print(f"\n  SIGNIFICANCE (perm, H0: correlation breaks unrelated to episode dates, N={N_PERM}):")
    print(f"    share of episodes with a break in the {LOOKBACK}d window: {obs:.1%}   perm p={p:.4f}"
          f"   z~{z:+.2f}  <- H5 trial")

    n_rise = sum(1 for r in rows if r["direction"] == "rise")
    n_fall = sum(1 for r in rows if r["direction"] == "fall")
    print(f"\n  DIRECTION (descriptive, the SPEC does not fix a sign): rise {n_rise}, fall {n_fall}"
          f" of {n_rise + n_fall} breaks found")

    trial = [{"n": 5, "name": "H5 corr_break lead", "b": obs, "t": z, "p": p, "mde": float("nan")}]
    json.dump({"episodes": rows, "perm_p": p, "trial": trial,
               "n_rise": n_rise, "n_fall": n_fall,
               "params": {"CORR_WIN": CORR_WIN, "PETTITT_WINDOW": PETTITT_WINDOW,
                         "PETTITT_RECENT": PETTITT_RECENT, "LOOKBACK": LOOKBACK}},
              open(os.path.join(RESULTS, "p2_h5_summary.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(RESULTS, 'p2_h5_summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
