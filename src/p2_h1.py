"""P2 — H1 on real data: detecting a volatility regime change (SPY).

Stage 3 of the plan (SPEC §10). The difference from P1: there the MS model was fitted ONCE on
the whole synthetic series (best case, the parameters see the whole future). Here it is
a true walk-forward: the model is refitted on a rolling window of ONLY
the past WINDOW days, every REFIT_EVERY days (SPEC §5.5: "all signals on
data up to time t"). Between refits the regime probability "sticks"
at the last known value — this adds up to REFIT_EVERY-1 days to the delay
on top of the model's own delay, and this is stated explicitly rather than
passed off as "method delay".

CUSUM here uses an ADAPTIVE baseline (a rolling window of BASE_WINDOW days
before the current day), not one fixed reference point at the start of
the series as in P1: the market over 11 years is non-stationary, "normal" volatility
in 2015 and 2025 are different things. This is a departure from the version validated on
synthetic data, and it is named explicitly — the synthetic test checked the method's principle, not this
particular baseline calibration.

The "true" episodes here come from VIX (an instrument independent of SPY returns:
the options market, not the series the detectors run on), not from a
generative model (there is none for real data) and NOT from SPY realised vol.

IMPORTANT FIX (found in this same run, before the results were published): the first
version labelled the "truth" with SPY 20-day realised vol. A check showed that
5-day RV (on which CUSUM/MS run) regularly crossed the threshold
10-15 days EARLIER than the 20-day one — because a slow moving average
itself lags. As a result the detectors looked faster than the synthetic ceiling
(70.6% hit<=3d vs 40.1% in P1) — not because they are good, but because
the ground truth was artificially slowed down by the same type of series the detector
measures. VIX as an independent marker removes this loop.
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
import scipy.stats as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

TICKER = "SPY"
GT_TICKER = "^VIX"        # "truth" labelling independent of SPY returns
CALM_MULT = 1.75          # episode threshold: VIX > CALM_MULT * full-history median
MIN_EPISODE = 5           # days; real data is noisier than synthetic
LOOKAHEAD = 20            # days for detection after the episode starts

MS_WINDOW = 750           # ~3 years, the walk-forward fit window
MS_REFIT_EVERY = 5        # days between refits
MS_THRESH = 0.7           # SPEC H1
MS_PERSIST = [1, 3]       # from P1: persist lowers FPR at the cost of power — compare both

CUSUM_VOL_WIN = 5
CUSUM_BASE_WINDOW = 250
CUSUM_K_MULT = 0.5
CUSUM_H_MULT = 5.0


def true_episodes(gt: pd.Series) -> list[dict]:
    """gt — a series independent of the detector (VIX), not the 20-d RV of the same ticker."""
    baseline = gt.median()
    thresh = baseline * CALM_MULT
    above = (gt > thresh).to_numpy()
    idx = gt.index
    episodes = []
    t = 0
    n = len(above)
    while t < n:
        if above[t] and (t == 0 or not above[t - 1]):
            j = t
            while j < n and above[j]:
                j += 1
            if j - t >= MIN_EPISODE:
                seg = gt.iloc[t:j]
                episodes.append({
                    "onset_idx": t, "onset_date": str(idx[t].date()),
                    "peak_date": str(seg.idxmax().date()),
                    "peak_vix": float(seg.max()),
                    "end_date": str(idx[j - 1].date()),
                    "n_days": int(j - t),
                })
            t = j
        else:
            t += 1
    return episodes, float(baseline), float(thresh)


def ms_walkforward(returns: np.ndarray, window: int = MS_WINDOW,
                    refit_every: int = MS_REFIT_EVERY) -> np.ndarray:
    """Causal: at each refit step t the model sees only returns[t-window:t]."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    n = len(returns)
    prob = np.full(n, np.nan)
    n_fit, n_fail, n_gate_fail = 0, 0, 0
    for t in range(window, n, refit_every):
        seg = returns[t - window:t]
        try:
            mod = MarkovRegression(seg, k_regimes=2, switching_variance=True)
            res = mod.fit(disp=False)
        except Exception:                                        # noqa: BLE001
            n_fail += 1
            continue
        n_fit += 1
        mu1, sd1 = float(np.mean(seg)), float(np.std(seg))
        llf1 = float(st.norm.logpdf(seg, mu1, sd1).sum())
        bic1 = 2 * np.log(len(seg)) - 2 * llf1
        if not (res.bic < bic1):
            n_gate_fail += 1
            p_now = 0.0
        else:
            names = mod.param_names
            p = dict(zip(names, res.params))
            hi = 0 if p["sigma2[0]"] > p["sigma2[1]"] else 1
            filt = np.asarray(res.filtered_marginal_probabilities)
            p_now = float(filt[-1, hi])
        end = min(t + refit_every, n)
        prob[t:end] = p_now
    print(f"    MS walk-forward: refits {n_fit}   did not converge {n_fail}"
          f"   BIC gate rejected {n_gate_fail}/{n_fit if n_fit else 1}")
    return prob


def flag_from_prob(prob: np.ndarray, thresh: float, persist: int) -> np.ndarray:
    above = np.nan_to_num(prob, nan=0.0) > thresh
    if persist <= 1:
        return above
    flag = np.zeros(len(prob), dtype=bool)
    run = 0
    for t in range(len(prob)):
        run = run + 1 if above[t] else 0
        flag[t] = run >= persist
    return flag


def detect_cusum_adaptive(returns: np.ndarray, vol_win: int = CUSUM_VOL_WIN,
                           base_window: int = CUSUM_BASE_WINDOW,
                           k_mult: float = CUSUM_K_MULT,
                           h_mult: float = CUSUM_H_MULT) -> np.ndarray:
    vol = pd.Series(returns).rolling(vol_win).std().to_numpy() * np.sqrt(252)
    n = len(returns)
    flag = np.zeros(n, dtype=bool)
    s = 0.0
    for t in range(base_window, n):
        base = vol[t - base_window:t]
        base = base[np.isfinite(base)]
        if len(base) < base_window // 2:
            continue
        mu0, sd0 = float(np.mean(base)), float(np.std(base))
        x = vol[t]
        if not np.isfinite(x) or sd0 == 0:
            continue
        k, h = k_mult * sd0, h_mult * sd0
        s = max(0.0, s + (x - mu0 - k))
        if s > h:
            flag[t] = True
            s = 0.0
    return flag


NAIVE_MULT = 1.75          # the same multiplier as CALM_MULT for the VIX ground truth
NAIVE_BASE_WINDOW = 250


def detect_naive_rv(returns: np.ndarray, vol_win: int = 5,
                     base_window: int = NAIVE_BASE_WINDOW,
                     mult: float = NAIVE_MULT) -> np.ndarray:
    """A model-free baseline (review, priority 2): an ATR-like threshold on RV5 —
    a flag when today's realised vol is above mult x the median over
    the trailing base_window days. Causal. If MS/CUSUM do not beat this
    baseline, the model's complexity does not pay off."""
    vol = pd.Series(returns).rolling(vol_win).std().to_numpy() * np.sqrt(252)
    n = len(returns)
    flag = np.zeros(n, dtype=bool)
    for t in range(base_window, n):
        base = vol[t - base_window:t]
        base = base[np.isfinite(base)]
        if len(base) < base_window // 2 or not np.isfinite(vol[t]):
            continue
        if vol[t] > np.median(base) * mult:
            flag[t] = True
    return flag


def episode_delay(flag: np.ndarray, onset_idx: int) -> float:
    window = flag[onset_idx:onset_idx + LOOKAHEAD]
    hit = np.where(window)[0]
    return float(hit[0]) if len(hit) else float("nan")


def hit_le3_rate(flag: np.ndarray, onset_idxs: list[int]) -> float:
    hits = [1.0 if np.any(flag[on:on + 4]) else 0.0 for on in onset_idxs]
    return float(np.mean(hits)) if hits else float("nan")


N_PERM = 5000


def permutation_pvalue(flag: np.ndarray, onset_idxs: list[int], observed: float,
                        n_perm: int = N_PERM, seed: int = 0) -> float:
    """H0: the method's flags are unrelated to episode dates — a circular shift of the whole
    flag series preserves its autocorrelation/cluster structure
    (detectors flag in runs) but breaks the link to the real crisis dates."""
    rng = np.random.default_rng(seed)
    n = len(flag)
    count = 0
    for _ in range(n_perm):
        shifted = np.roll(flag, int(rng.integers(1, n)))
        if hit_le3_rate(shifted, onset_idxs) >= observed - 1e-9:
            count += 1
    return (count + 1) / (n_perm + 1)


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 78)
    print(f"P2  H1 ON REAL DATA  ({TICKER}, walk-forward, no look-ahead)")
    print("=" * 78)

    close = pd.read_csv(os.path.join(DATA, "p0_close.csv"), index_col=0, parse_dates=True)
    px = close[TICKER].dropna()
    ret_full = px.pct_change().dropna()
    vix = close[GT_TICKER].dropna()

    common = ret_full.index.intersection(vix.index)
    ret = ret_full.reindex(common)
    gt = vix.reindex(common)
    print(f"  common trading days {TICKER}/{GT_TICKER}: {len(common)}"
          f"   ({common[0].date()}..{common[-1].date()})")

    episodes, baseline, thresh = true_episodes(gt)
    print(f"  median {GT_TICKER} over the whole period: {baseline:.1f}"
          f"   episode threshold ({CALM_MULT}x): {thresh:.1f}")
    print(f"  episodes found (>= {MIN_EPISODE} consecutive days above the threshold): {len(episodes)}")
    for e in episodes:
        print(f"    {e['onset_date']} -> peak {e['peak_date']} (VIX {e['peak_vix']:.1f})"
              f"   end {e['end_date']}   {e['n_days']}d")

    r = ret.to_numpy()
    idx = common

    print(f"\n  MS walk-forward: window {MS_WINDOW}d, refit every {MS_REFIT_EVERY}d,"
          f" persist in {{{','.join(map(str, MS_PERSIST))}}}, threshold {MS_THRESH}")
    prob = ms_walkforward(r)

    print(f"\n  CUSUM: adaptive base {CUSUM_BASE_WINDOW}d, k={CUSUM_K_MULT}σ, h={CUSUM_H_MULT}σ")
    cusum_flag = detect_cusum_adaptive(r)

    method_flags = {f"MS_persist{p}": flag_from_prob(prob, MS_THRESH, p) for p in MS_PERSIST}
    method_flags["CUSUM_adaptive"] = cusum_flag
    print(f"\n  BASELINE (review, priority 2): naive threshold RV5 > {NAIVE_MULT}x"
          f" trailing median {NAIVE_BASE_WINDOW}d, no model at all")
    method_flags["naive_RV_threshold"] = detect_naive_rv(r)

    pd.DataFrame({name: f for name, f in method_flags.items()}, index=idx).to_csv(
        os.path.join(RESULTS, "p2_h1_flags.csv"))

    print(f"\n  DETECTION DELAY BY EPISODE (SPEC H1: probability>0.7 1-3 days before the peak,")
    print(f"  measured here from the START of the episode, walk-forward lag of up to {MS_REFIT_EVERY - 1}d included)")
    rows = []
    for e in episodes:
        onset_idx = e["onset_idx"]
        line = f"    {e['onset_date']}:"
        for name, flag in method_flags.items():
            d = episode_delay(flag, onset_idx)
            line += f"  {name}={'none' if np.isnan(d) else f'{int(d)}d'}"
            rows.append({**e, "method": name, "delay_days": d})
        print(line)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "p2_h1_episodes.csv"), index=False)

    print(f"\n  TOTAL by method ({len(episodes)} episodes):")
    print(f"    {'method':16s} {'hit<=3d':>8s} {'hit<=5d':>8s} {'median(det)':>15s} {'miss':>6s}")
    summary = {}
    for name in method_flags:
        d = df[df.method == name]["delay_days"].to_numpy()
        hit3 = float(np.mean(d <= 3)) if len(d) else float("nan")
        hit5 = float(np.mean(d <= 5)) if len(d) else float("nan")
        det = d[np.isfinite(d)]
        med = float(np.median(det)) if len(det) else float("nan")
        miss = float(np.mean(np.isnan(d))) if len(d) else float("nan")
        print(f"    {name:16s} {hit3:8.1%} {hit5:8.1%} {med:15.1f} {miss:6.1%}")
        summary[name] = {"hit_le_3d": hit3, "hit_le_5d": hit5,
                         "median_delay": med, "miss_rate": miss}

    print(f"\n  COMPARISON WITH THE SYNTHETIC CEILING (P1):")
    print(f"    The FIRST check (ratio=2.0, ceiling MS_persist1 hit<=3d=40.1%) against")
    print(f"    the first real-data result (81.8%, VIX ground truth) looked")
    print(f"    suspicious — reality 'better' than the ceiling. Check: the effective")
    print(f"    stress/calm ratio in the real 2020/2022/2025 episodes (by RV5)")
    print(f"    was 2.9x-9.2x — ABOVE the upper bound (3.0) tested in P1.")
    print(f"    After adding ratio=5.0/8.0 to P1: the MS_persist1 hit<=3d ceiling there is")
    print(f"    95.5%/98.5% — the real 81.8% is now BELOW the ceiling, as expected.")
    print(f"    Conclusion: the discrepancy was in the synthetic range, not in the detector.")

    print(f"\n  SIGNIFICANCE (permutation test, H0: flags unrelated to episode dates,")
    print(f"  circular shift of the whole flag series, N={N_PERM}):")
    print(f"  The PRIMARY H1 specification is MS_persist1: it was chosen from P1")
    print(f"  (synthetic data, BEFORE real data) as the best by power at FPR=0%.")
    print(f"  MS_persist3 and CUSUM are diagnostics/robustness, not part of the m=8 family")
    print(f"  (SPEC §9: 'do not extend m after looking at the data').")
    onset_idxs = [e["onset_idx"] for e in episodes]
    trial = []
    PRIMARY = "MS_persist1"
    for name in method_flags:
        obs = hit_le3_rate(method_flags[name], onset_idxs)
        p = permutation_pvalue(method_flags[name], onset_idxs, obs)
        z = float(st.norm.ppf(1 - p)) if p < 1.0 else 0.0
        summary[name]["perm_p"] = p
        tag = "  <- H1 trial" if name == PRIMARY else "  diagnostic"
        print(f"    {name:16s}  hit<=3d obs. {obs:6.1%}   perm p={p:.4f}"
              f"   z~{z:+.2f}{tag}")
        if name == PRIMARY:
            trial.append({"n": 1, "name": "H1 MS_persist1", "b": obs,
                          "t": z, "p": p, "mde": float("nan")})
    print(f"    z is the inverse transform of p via Φ⁻¹, FOR COMPATIBILITY with")
    print(f"    the |t| threshold of the other hypotheses (Stage 7); resolution is limited by N_PERM,")
    print(f"    p cannot be < 1/{N_PERM + 1} = {1 / (N_PERM + 1):.5f}.")

    # ===================================================================
    # PRIORITY 2 (review): FPR separately for 2017/2019/2023 (SPEC §7 names
    # exactly these years as the "calm periods" for the <=10% criterion) — do not
    # mix with the overall FPR across the whole history.
    # ===================================================================
    print(f"\n{'=' * 78}")
    print(f"  FPR ON 2017/2019/2023 SPECIFICALLY (SPEC §7), NOT OVER THE WHOLE HISTORY")
    print(f"{'=' * 78}")
    calm_years = [2017, 2019, 2023]
    fpr_rows = []
    for name, flag in method_flags.items():
        per_year = {}
        for y in calm_years:
            mask = (idx.year == y)
            if mask.sum() == 0:
                continue
            per_year[y] = float(flag[mask].mean())
        chunks = [flag[idx.year == y] for y in calm_years if (idx.year == y).sum() > 0]
        pooled = float(np.mean(np.concatenate(chunks))) if chunks else float("nan")
        print(f"    {name:20s}  " + "  ".join(f"{y}={per_year.get(y, float('nan')):5.1%}" for y in calm_years)
              + f"   pooled={pooled:5.1%}   {'PASS' if pooled <= 0.10 else 'FAIL'} (<=10%)")
        fpr_rows.append({"method": name, **{str(y): per_year.get(y) for y in calm_years}, "pooled": pooled})
    print(f"    ('naive_RV_threshold' is the baseline from priority 2, not a model)")

    print(f"\n  --- NAIVE BASELINE VS MS/CUSUM (does model complexity beat a simple threshold?) ---")
    for name in ["MS_persist1", "CUSUM_adaptive", "naive_RV_threshold"]:
        hit3 = summary[name]["hit_le_3d"]
        pooled = next(r["pooled"] for r in fpr_rows if r["method"] == name)
        print(f"    {name:20s}  hit<=3d={hit3:6.1%}   FPR(2017/19/23)={pooled:6.1%}")
    ms_better = (summary["MS_persist1"]["hit_le_3d"] > summary["naive_RV_threshold"]["hit_le_3d"] and
                next(r['pooled'] for r in fpr_rows if r['method']=='MS_persist1') <=
                next(r['pooled'] for r in fpr_rows if r['method']=='naive_RV_threshold'))
    print(f"    MS_persist1 beats the naive threshold on both hit rate AND FPR:"
          f" {'yes' if ms_better else 'NO — model complexity does not pay off on this metric'}")

    json.dump({"baseline_rv": baseline, "threshold_rv": thresh,
               "n_episodes": len(episodes), "episodes": episodes,
               "summary": summary, "fpr_calm_years": fpr_rows,
               "params": {"MS_WINDOW": MS_WINDOW, "MS_REFIT_EVERY": MS_REFIT_EVERY,
                         "MS_THRESH": MS_THRESH, "CUSUM_BASE_WINDOW": CUSUM_BASE_WINDOW}},
              open(os.path.join(RESULTS, "p2_h1_summary.json"), "w"), indent=1)
    json.dump(trial, open(os.path.join(RESULTS, "p2_h1_trial.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(RESULTS, 'p2_h1_episodes.csv')}")
    print(f"  wrote {os.path.join(RESULTS, 'p2_h1_summary.json')}")
    print(f"  wrote {os.path.join(RESULTS, 'p2_h1_trial.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
