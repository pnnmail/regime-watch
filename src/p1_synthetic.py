"""P1 — synthetic data and power estimation. Nothing is tested on real data.

Stage 2 of the plan (SPEC §10): before the regime change detection methods
(§5.1) see a single real ticker, they are run on series with a KNOWN
transition point. Power/delay/FPR are printed first — not the other way round. This is the same
technique as the MDE in an earlier study: the numbers that determine whether
a method is fit for purpose are published before the method sees reality (see SPEC §9: "do not
extend m after looking at the data" — here the analogue for detector thresholds).

Three methods from SPEC §5.1, all with an honest caveat about causality:
  - MS  — statsmodels MarkovRegression (2 states, switching_variance).
          Filtered probabilities are causal (they use only the past); but
          the model PARAMETERS are estimated once on the whole series (full-sample
          MLE) — this is a best-case estimate (as if the model knew the true
          dynamics), not full live refitting. The limitation is named,
          not hidden.
  - CUSUM — on rolling realised volatility, fully causal
          (only the past 5 days at each step).
  - Pettitt — on a rolling window of realised volatility, fully
          causal (only the last W days at each step).

hmmlearn and ruptures are unavailable in this environment (no C++ compiler) — see
the SPEC "Implementation note".
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
from scipy.stats import rankdata

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# ---------------------------------------------------------------- generator
CALM_MU_ANN, CALM_SIG_ANN = 0.08, 0.12          # SPEC §3 H1: "low volatility"
STRESS_MU_ANN = -0.15                            # realism; direction is not tested (SPEC §8)
MEAN_DUR_CALM, MEAN_DUR_STRESS = 150, 20         # days, typical regime lengths
MIN_EPISODE = 3                                  # a shorter episode does not count as a "real" transition
LOOKAHEAD = 20                                   # days for detection after the true transition
RATIOS = [1.5, 2.0, 3.0, 5.0, 8.0]               # stress_sigma / calm_sigma — the power axis
                                                  # 5.0/8.0 were added after P2 H1: real
                                                  # crisis episodes (COVID, Apr-2025) give
                                                  # an effective ratio of 2.9x-9.2x, outside the
                                                  # 1.5-3.0 range — a comparison point of the same order is needed
N_REPS = 25
N_DAYS = 1500
N_DAYS_CALM_ONLY = 1000
N_REPS_FPR = 25
MS_THRESH = 0.7                                  # SPEC H1: "probability > 0.7"
SEED0 = 20260101


def simulate(n_days: int, sigma_ratio: float | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """sigma_ratio=None -> a purely calm series (for FPR), with no second regime."""
    rng = np.random.default_rng(seed)
    calm_sig = CALM_SIG_ANN / np.sqrt(252)
    calm_mu = CALM_MU_ANN / 252
    if sigma_ratio is None:
        r = calm_mu + calm_sig * rng.standard_normal(n_days)
        return r, np.zeros(n_days, dtype=int)
    stress_sig = calm_sig * sigma_ratio
    stress_mu = STRESS_MU_ANN / 252
    p_stay_calm = 1 - 1 / MEAN_DUR_CALM
    p_stay_stress = 1 - 1 / MEAN_DUR_STRESS
    state = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        p_stay = p_stay_calm if state[t - 1] == 0 else p_stay_stress
        state[t] = state[t - 1] if rng.random() < p_stay else 1 - state[t - 1]
    mu = np.where(state == 0, calm_mu, stress_mu)
    sig = np.where(state == 0, calm_sig, stress_sig)
    r = mu + sig * rng.standard_normal(n_days)
    return r, state


def true_onsets(state: np.ndarray) -> list[int]:
    """Indices of days where a regime-1 episode lasting >= MIN_EPISODE days begins."""
    onsets = []
    t = 1
    n = len(state)
    while t < n:
        if state[t] == 1 and state[t - 1] == 0:
            j = t
            while j < n and state[j] == 1:
                j += 1
            if j - t >= MIN_EPISODE:
                onsets.append(t)
            t = j
        else:
            t += 1
    return onsets


# ---------------------------------------------------------------- methods
def ms_prob(returns: np.ndarray) -> tuple[np.ndarray, bool] | None:
    """Filtered probability of the high-volatility regime, causal (past only).

    BIC gate: 72% FPR on the first run (persist did not help, see the
    commit/log history) meant not "slow detection" but that a 2-regime
    model on HOMOGENEOUS data still finds an arbitrary split —
    non-identifiability, not signal. The gate: the 2-regime model is accepted
    only if its BIC beats the BIC of a plain Gaussian without switching; otherwise
    no regime is detected at all (the probability array is zero), rather than "late".
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    import scipy.stats as st
    try:
        mod = MarkovRegression(returns, k_regimes=2, switching_variance=True)
        res = mod.fit(disp=False)
    except Exception:                                            # noqa: BLE001
        return None
    mu1, sd1 = float(np.mean(returns)), float(np.std(returns))
    llf1 = float(st.norm.logpdf(returns, mu1, sd1).sum())
    bic1 = 2 * np.log(len(returns)) - 2 * llf1
    gate_passed = bool(res.bic < bic1)
    names = mod.param_names
    p = dict(zip(names, res.params))
    hi = 0 if p["sigma2[0]"] > p["sigma2[1]"] else 1
    filt = np.asarray(res.filtered_marginal_probabilities)
    prob = filt[:, hi] if gate_passed else np.zeros(len(returns))
    return prob, gate_passed


def flag_from_prob(prob: np.ndarray | None, thresh: float = MS_THRESH,
                    persist: int = 1) -> np.ndarray | None:
    """persist>1: requires `persist` consecutive days above the threshold — suppresses spurious
    two-regime splits of homogeneous data (see the MS_filtered FPR below)."""
    if prob is None:
        return None
    above = prob > thresh
    if persist <= 1:
        return above
    flag = np.zeros(len(prob), dtype=bool)
    run = 0
    for t in range(len(prob)):
        run = run + 1 if above[t] else 0
        flag[t] = run >= persist
    return flag


def detect_cusum(returns: np.ndarray, burn_in: int = 60, k_mult: float = 0.5,
                  h_mult: float = 5.0) -> np.ndarray:
    vol = pd.Series(returns).rolling(5).std().to_numpy() * np.sqrt(252)
    mu0 = np.nanmean(vol[:burn_in])
    sd0 = np.nanstd(vol[:burn_in])
    k = k_mult * sd0
    h = h_mult * sd0
    flag = np.zeros(len(returns), dtype=bool)
    s = 0.0
    for t in range(len(returns)):
        x = vol[t]
        if not np.isfinite(x):
            continue
        s = max(0.0, s + (x - mu0 - k))
        if s > h:
            flag[t] = True
            s = 0.0                    # reset — otherwise one episode keeps the signal on until the end of the series
    return flag


def pettitt_pvalue(x: np.ndarray) -> tuple[float, int]:
    """K, p-value and the change position (Pettitt 1979)."""
    n = len(x)
    r = rankdata(x)
    u = 2 * np.cumsum(r)[:-1] - np.arange(1, n) * (n + 1)
    k = int(np.argmax(np.abs(u)))
    kstat = float(np.abs(u[k]))
    p = 2 * np.exp(-6 * kstat ** 2 / (n ** 3 + n ** 2))
    return min(p, 1.0), k + 1          # k+1: position within the window (1..n-1)


def detect_pettitt(returns: np.ndarray, window: int = 60, recent: int = 10,
                    alpha: float = 0.05) -> np.ndarray:
    vol = pd.Series(returns).rolling(5).std().to_numpy() * np.sqrt(252)
    flag = np.zeros(len(returns), dtype=bool)
    for t in range(window, len(returns)):
        w = vol[t - window:t]
        if not np.all(np.isfinite(w)):
            continue
        p, pos = pettitt_pvalue(w)
        if p < alpha and pos >= window - recent:
            flag[t] = True
    return flag


CAUSAL_METHODS = {"CUSUM": detect_cusum, "Pettitt": detect_pettitt}
MS_PERSIST = [1, 2, 3]     # consecutive days above the threshold; 1 = literally as in SPEC H1
MS_NAMES = [f"MS_persist{p}" for p in MS_PERSIST]
ALL_METHOD_NAMES = MS_NAMES + list(CAUSAL_METHODS)


def flags_for_replication(returns: np.ndarray) -> tuple[dict[str, np.ndarray | None], bool | None]:
    """One MS fit per replica, three persist variants from it + CUSUM/Pettitt separately.

    Returns (flags, gate_passed) — gate_passed=None if MS did not converge.
    """
    r = ms_prob(returns)
    if r is None:
        prob, gate_passed = None, None
    else:
        prob, gate_passed = r
    out = {name: (None if prob is None else flag_from_prob(prob, MS_THRESH, p))
           for name, p in zip(MS_NAMES, MS_PERSIST)}
    for name, fn in CAUSAL_METHODS.items():
        out[name] = fn(returns)
    return out, gate_passed


def episode_delays(flag: np.ndarray, onsets: list[int]) -> list[float]:
    out = []
    for on in onsets:
        window = flag[on:on + LOOKAHEAD]
        hit = np.where(window)[0]
        out.append(float(hit[0]) if len(hit) else float("nan"))
    return out


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 78)
    print("P1  SYNTHETIC DATA AND POWER  (real data untouched)")
    print("=" * 78)
    print(f"  generator: 2-state Markov-switching, calm σ_ann={CALM_SIG_ANN:.0%}"
          f", mean dur calm/stress={MEAN_DUR_CALM}/{MEAN_DUR_STRESS} d.")
    print(f"  power axis: stress σ / calm σ ∈ {RATIOS}   N_REPS={N_REPS}"
          f"   N_DAYS={N_DAYS}   LOOKAHEAD={LOOKAHEAD}d   MIN_EPISODE={MIN_EPISODE}d")

    # ------------------------------------------------------- FPR on calm series
    print(f"\n  --- FALSE POSITIVE RATE (purely calm series, {N_REPS_FPR}"
          f" replicas x {N_DAYS_CALM_ONLY} d.) ---")
    fpr = {m: [] for m in ALL_METHOD_NAMES}
    gate_pass = []
    for i in range(N_REPS_FPR):
        r, _ = simulate(N_DAYS_CALM_ONLY, None, seed=SEED0 + i)
        flags, gp = flags_for_replication(r)
        if gp is not None:
            gate_pass.append(gp)
        for name, flag in flags.items():
            if flag is None:
                continue
            fpr[name].append(float(np.mean(flag)))
    fpr_summary = {m: (float(np.mean(v)) if v else None) for m, v in fpr.items()}
    for m, v in fpr_summary.items():
        n_fail = N_REPS_FPR - len(fpr[m])
        print(f"    {m:14s}  share of days with a false flag: "
              f"{'n/a' if v is None else f'{v:6.2%}'}"
              + (f"   (did not converge: {n_fail}/{N_REPS_FPR})" if n_fail else ""))
    print(f"    SPEC §7 success criterion: <= 10% in calm periods")
    if gate_pass:
        print(f"    MS BIC gate: the 2-regime model was accepted in "
              f"{np.mean(gate_pass):.0%} of {len(gate_pass)} replicas"
              f" (on HOMOGENEOUS data — ideally ~0%)")

    # ------------------------------------------------------- power by ratio
    power_rows = []
    for ratio in RATIOS:
        print(f"\n  --- RATIO stress/calm σ = {ratio} ---")
        delays = {m: [] for m in ALL_METHOD_NAMES}
        n_onsets_total = 0
        n_converge_fail = {m: 0 for m in ALL_METHOD_NAMES}
        gate_pass = []
        for i in range(N_REPS):
            r, state = simulate(N_DAYS, ratio, seed=SEED0 + 1000 + i)
            onsets = true_onsets(state)
            n_onsets_total += len(onsets)
            flags, gp = flags_for_replication(r)
            if gp is not None:
                gate_pass.append(gp)
            for name, flag in flags.items():
                if flag is None:
                    n_converge_fail[name] += 1
                    continue
                delays[name].extend(episode_delays(flag, onsets))
        print(f"    true episodes (>= {MIN_EPISODE} d.) total: {n_onsets_total}"
              + (f"   MS BIC gate accepted in {np.mean(gate_pass):.0%} of replicas" if gate_pass else ""))
        for name in ALL_METHOD_NAMES:
            d = np.array(delays[name], dtype=float)
            hit3 = float(np.mean(d <= 3)) if len(d) else float("nan")      # share <=3d, misses counted as misses
            hit5 = float(np.mean(d <= 5)) if len(d) else float("nan")
            det = d[np.isfinite(d)]
            med = float(np.median(det)) if len(det) else float("nan")
            miss_rate = float(np.mean(np.isnan(d))) if len(d) else float("nan")
            cf = n_converge_fail[name]
            print(f"    {name:14s}  hit<=3d: {hit3:6.1%}   hit<=5d: {hit5:6.1%}"
                  f"   median delay (detected): {med:5.1f}d   miss: {miss_rate:6.1%}"
                  + (f"   did not converge: {cf}/{N_REPS}" if cf else ""))
            power_rows.append({
                "ratio": ratio, "method": name, "n_episodes": int(len(d)),
                "hit_le_3d": hit3, "hit_le_5d": hit5,
                "median_delay_detected": med, "miss_rate": miss_rate,
                "converge_fail": int(cf),
            })

    pd.DataFrame(power_rows).to_csv(os.path.join(RESULTS, "p1_power.csv"), index=False)
    json.dump({"fpr": fpr_summary, "power": power_rows,
               "params": {"CALM_MU_ANN": CALM_MU_ANN, "CALM_SIG_ANN": CALM_SIG_ANN,
                          "STRESS_MU_ANN": STRESS_MU_ANN, "MEAN_DUR_CALM": MEAN_DUR_CALM,
                          "MEAN_DUR_STRESS": MEAN_DUR_STRESS, "RATIOS": RATIOS,
                          "N_REPS": N_REPS, "N_DAYS": N_DAYS, "LOOKAHEAD": LOOKAHEAD,
                          "MS_THRESH": MS_THRESH}},
              open(os.path.join(RESULTS, "p1_power.json"), "w"), indent=1)

    print(f"\n  CONCLUSION (before real data):")
    print(f"    The H1 threshold (SPEC §3) — 'probability > 0.7 1-3 days before the peak' — is")
    print(f"    a question of HOW FAST the regime becomes visible, not whether it exists.")
    print(f"    The numbers above are a ceiling under a known generative model. On real")
    print(f"    data (mixed regimes, heavy tails, gaps) worse is expected —")
    print(f"    if the result on real data turns out BETTER than this ceiling,")
    print(f"    that signals a bug in the code rather than luck.")
    print(f"\n  wrote {os.path.join(RESULTS, 'p1_power.csv')}")
    print(f"  wrote {os.path.join(RESULTS, 'p1_power.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
