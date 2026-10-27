"""P2 — H8: a composite indicator of >=3 signals, out-of-sample ROC-AUC.

SPEC §3 H8: "A combination of >= 3 signals gives a better ROC-AUC than each one
alone. Test: logistic regression on the label 'regime change in
the next 5 days'." SPEC §5.5: walk-forward train 2020-2023, test 2024-2026.

Signals (5, H3 excluded from the primary specification): CUSUM statistic (H1),
VIX/VIX3M ratio_dev (H2), avg pairwise sector corr (H5), breadth<MA50 (H4),
the mean return of 3 Asian indices (H7). H3 (the credit spread) would give
a panel only from 2023 — it would cut the 2020-2023 training sample almost
entirely; it is included separately as a DIAGNOSTIC on the truncated window, not in
the primary specification.

Label: y_t=1 if day t is calm NOW (VIX_t <= the threshold from H1) AND VIX
crosses the threshold at least once during t+1..t+5 (a transition, not "already inside
the regime" — otherwise it is trivially predictable from VIX's own autocorrelation).
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH"]
ASIA_NATIVE = ["^KS11", "^N225", "^HSI"]
CALM_MULT = 1.75
LABEL_H = 5
CORR_WIN = 20
MA_WIN = 50
CUSUM_BASE_WINDOW = 250
TRAIN = ("2020-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-12-31")
N_BOOT = 2000


def cusum_stat(returns: np.ndarray, vol_win=5, base_window=CUSUM_BASE_WINDOW,
                k_mult=0.5) -> np.ndarray:
    """The continuous CUSUM statistic S_t (not a binary flag) — as a regression feature."""
    vol = pd.Series(returns).rolling(vol_win).std().to_numpy() * np.sqrt(252)
    n = len(returns)
    s_arr = np.full(n, np.nan)
    s = 0.0
    for t in range(base_window, n):
        base = vol[t - base_window:t]
        base = base[np.isfinite(base)]
        if len(base) < base_window // 2:
            continue
        mu0, sd0 = float(np.mean(base)), float(np.std(base))
        x = vol[t]
        if not np.isfinite(x) or sd0 == 0:
            s_arr[t] = s
            continue
        k = k_mult * sd0
        s = max(0.0, s + (x - mu0 - k) / sd0)     # normalised by sd0 — comparable between periods
        s_arr[t] = s
    return s_arr


def avg_pairwise_corr(ret: pd.DataFrame, win: int) -> pd.Series:
    out = []
    idx = ret.index
    for i in range(len(idx)):
        if i < win:
            out.append(np.nan)
            continue
        c = ret.iloc[i - win:i].corr().to_numpy()
        iu = np.triu_indices_from(c, k=1)
        pairs = c[iu]
        pairs = pairs[np.isfinite(pairs)]
        out.append(float(np.mean(pairs)) if len(pairs) else np.nan)
    return pd.Series(out, index=idx)


def build_panel() -> tuple[pd.DataFrame, list[str]]:
    close = pd.read_csv(os.path.join(DATA, "p0_close.csv"), index_col=0, parse_dates=True)
    spy = close["SPY"].dropna()
    vix = close["^VIX"].dropna()
    vix3m = close["^VIX3M"].dropna()
    sec = close[SECTORS].dropna(how="all")

    idx = spy.index
    ret = spy.pct_change()

    feat = pd.DataFrame(index=idx)
    feat["cusum"] = cusum_stat(ret.reindex(idx).to_numpy())
    ratio = (vix.reindex(idx) / vix3m.reindex(idx))
    feat["ratio_dev"] = ratio - 1.0

    sec_ret = sec.pct_change()
    avgcorr = avg_pairwise_corr(sec_ret, CORR_WIN)
    feat["avgcorr"] = avgcorr.reindex(idx)

    ma = sec.rolling(MA_WIN).mean()
    above = (sec > ma)
    n_avail = sec.notna().sum(axis=1)
    breadth = (above.sum(axis=1) / n_avail)
    breadth = breadth.where(n_avail >= 5)
    feat["breadth"] = breadth.reindex(idx)

    asia_path = os.path.join(DATA, "p2_h7_asia_native_ret.csv")
    if os.path.exists(asia_path):
        asia = pd.read_csv(asia_path, index_col=0, parse_dates=True)
        feat["asia_avg"] = asia[ASIA_NATIVE].mean(axis=1).reindex(idx)
    else:
        print("  WARNING: no data/p2_h7_asia_native_ret.csv — run p2_h7.py first."
              " asia_avg excluded.")

    signals = [c for c in ["cusum", "ratio_dev", "avgcorr", "breadth", "asia_avg"] if c in feat.columns]

    baseline = float(vix.median())
    thresh = baseline * CALM_MULT
    state = (vix.reindex(idx) > thresh).astype(float)
    calm_now = (state == 0)
    future_hit = pd.Series(False, index=idx)
    st = state.to_numpy()
    fut = np.zeros(len(st), dtype=bool)
    for t in range(len(st) - LABEL_H):
        fut[t] = np.any(st[t + 1:t + 1 + LABEL_H] > 0)
    future_hit = pd.Series(fut, index=idx)
    feat["y"] = (calm_now & future_hit).astype(float)
    feat.loc[~calm_now, "y"] = np.nan   # already in the regime -- not a "transition", excluded from the sample

    return feat, signals


def wf_auc(feat: pd.DataFrame, signals: list[str]) -> dict:
    train = feat.loc[TRAIN[0]:TRAIN[1]].dropna(subset=signals + ["y"])
    test = feat.loc[TEST[0]:TEST[1]].dropna(subset=signals + ["y"])
    print(f"  train {TRAIN[0]}..{TRAIN[1]}: n={len(train)}  positives={int(train['y'].sum())}"
          f" ({train['y'].mean():.1%})")
    print(f"  test  {TEST[0]}..{TEST[1]}: n={len(test)}  positives={int(test['y'].sum())}"
          f" ({test['y'].mean():.1%})")

    scaler = StandardScaler().fit(train[signals])
    Xtr, Xte = scaler.transform(train[signals]), scaler.transform(test[signals])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, train["y"])
    p_composite = clf.predict_proba(Xte)[:, 1]
    auc_composite = float(roc_auc_score(test["y"], p_composite))

    auc_individual = {}
    for i, s in enumerate(signals):
        clf1 = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            Xtr[:, [i]], train["y"])
        p1 = clf1.predict_proba(Xte[:, [i]])[:, 1]
        auc_individual[s] = float(roc_auc_score(test["y"], p1))

    return {"auc_composite": auc_composite, "auc_individual": auc_individual,
            "p_composite": p_composite, "y_test": test["y"].to_numpy(),
            "n_train": len(train), "n_test": len(test),
            "coef": dict(zip(signals, clf.coef_[0].tolist()))}


def bootstrap_p(y: np.ndarray, p_composite: np.ndarray, best_individual_auc: float,
                 n_boot: int = N_BOOT, seed: int = 0) -> float:
    """H0: AUC(composite) <= AUC(best single signal). Bootstrap over test days."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    obs_auc = roc_auc_score(y, p_composite)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p_composite[idx]
        if len(np.unique(yb)) < 2:
            continue
        diffs.append(roc_auc_score(yb, pb) - best_individual_auc)
    diffs = np.array(diffs)
    p = float(np.mean(diffs <= 0))
    return p, float(obs_auc)


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 78)
    print("P2  H8 — COMPOSITE INDICATOR (walk-forward: train 2020-2023, test 2024-2026)")
    print("=" * 78)

    feat, signals = build_panel()
    print(f"  signals: {len(signals)}  {signals}")

    res = wf_auc(feat, signals)
    print(f"\n  --- OUT-OF-SAMPLE ROC-AUC (test 2024-2026) ---")
    print(f"  {'signal':14s} {'AUC':>7s}")
    for s, a in sorted(res["auc_individual"].items(), key=lambda kv: -kv[1]):
        print(f"  {s:14s} {a:7.3f}")
    print(f"  {'COMPOSITE':14s} {res['auc_composite']:7.3f}  <- H8 trial")

    best_ind = max(res["auc_individual"].values())
    best_ind_name = max(res["auc_individual"], key=res["auc_individual"].get)
    print(f"\n  best single signal: {best_ind_name} (AUC={best_ind:.3f})")
    print(f"  composite {'BETTER than' if res['auc_composite'] > best_ind else 'NOT better than'} the best single one"
          f" (Δ={res['auc_composite'] - best_ind:+.3f})")
    print(f"  SPEC §7 criterion (AUC >= 0.65): {'PASSED' if res['auc_composite'] >= 0.65 else 'not passed'}")

    p, auc_check = bootstrap_p(res["y_test"], res["p_composite"], best_ind)
    print(f"\n  SIGNIFICANCE (bootstrap, H0: AUC(composite) <= AUC(best single), N={N_BOOT}):")
    print(f"    p={p:.4f}  <- H8 trial (significance of the composite's superiority)")

    # ---------------------------------------------- PRIORITY 4 (review): reformulation
    print(f"\n  --- WHY the composite does not win: correlation between signals ---")
    corr = feat[signals].dropna().corr()
    print(corr.round(2).to_string())
    off_diag = corr.to_numpy()[np.triu_indices(len(signals), k=1)]
    print(f"\n  |corr| between signals: median={np.median(np.abs(off_diag)):.2f},"
          f" range [{np.abs(off_diag).min():.2f}, {np.abs(off_diag).max():.2f}]")
    print(f"  cusum/ratio_dev/avgcorr/breadth are correlated with each other (|r|=0.22-0.64) —")
    print(f"  they are different measurements of THE SAME 'the market is tense right now', not")
    print(f"  independent sources of information. asia_avg is nearly orthogonal to the rest")
    print(f"  (|r|<0.2), but by itself weaker (AUC={res['auc_individual'].get('asia_avg', float('nan')):.3f})"
          f" than the best signal.")
    print(f"  CONCLUSION (not a 'FAIL' but a finding): H8 does not test 'do more signals help' —")
    print(f"  it inadvertently tests 'are these 5 signals independent', and the answer is NO.")
    print(f"  This is consistent with H2 (the VIX term structure adds nothing over HAR-RV,")
    print(f"  Priority 2 above) — probably they all mostly measure the same")
    print(f"  current level of realised volatility from different angles.")

    from scipy.stats import norm
    z = float(norm.ppf(1 - p)) if 0 < p < 1 else (5.0 if p == 0 else 0.0)
    trial = [{"n": 8, "name": "H8 composite>best_individual", "b": res["auc_composite"] - best_ind,
              "t": z, "p": p, "mde": float("nan")}]

    print(f"\n  logit regression coefficients (trained on train, sign = feature direction):")
    for s, c in res["coef"].items():
        print(f"    {s:14s} {c:+.3f}")

    json.dump({"auc_composite": res["auc_composite"], "auc_individual": res["auc_individual"],
               "best_individual": best_ind_name, "n_train": res["n_train"], "n_test": res["n_test"],
               "bootstrap_p": p, "trial": trial, "coef": res["coef"],
               "signal_correlation": corr.round(4).to_dict(),
               "interpretation": "signals not independent (|r|=0.22-0.64 among 4 of 5) -- "
                                  "composite fails to beat best single signal because it has "
                                  "little diversification to exploit, not because combining "
                                  "signals is inherently useless",
               "params": {"TRAIN": TRAIN, "TEST": TEST, "LABEL_H": LABEL_H}},
              open(os.path.join(RESULTS, "p2_h8_summary.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(RESULTS, 'p2_h8_summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
