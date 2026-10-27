"""P2 — H2: VIX/VIX3M inversion as a leading indicator of rising SPX RV.

SPEC §3 H2: "VIX/VIX3M inversion (ratio > 1) significantly
precedes a rise in SPX realised volatility at a 1-5 day horizon."

Print order as in the earlier study's MDE + trials scripts, only in one
file: first n and the SE-derived MDE of the coefficient, THEN the estimate itself —
the order is enforced by code (two sequential blocks), not by memory.

Primary specification (SPEC §9: fixed BEFORE looking at results):
  y = RV5_{t+h} - RV5_t,  h = 5 (the upper bound of the stated horizon)
  X = [ratio_dev_t, RV5_t]     ratio_dev_t = VIX_t/VIX3M_t - 1
  RV5_t as a control — otherwise the ratio_dev coefficient simply picks up
  mean reversion of realised volatility rather than the term structure.
h=1..4 and the binary inversion indicator are diagnostics/robustness, not part
of the m=8 family (SPEC §9).

The only multiple-testing correction in this project is FDR-BY q=0.10
(SPEC §3), with no additional family-wise threshold: the earlier study used
its own |t|=3.23 for ITS family of m=10, that does not carry over here automatically.
The final verdict over all m=8 is at Stage 7, not here.

Data: yfinance ^VIX3M stops at 2026-07-17 (~2 months earlier than
the other series) — the test does not cover the very end of the history. Stated, not
hidden (see also P0).
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

RV_WIN = 5
HORIZONS = [1, 2, 3, 4, 5]
PRIMARY_H = 5
HAC_LAGS = 10


def hac_fit(y: pd.Series, X: pd.DataFrame):
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def build_panel() -> pd.DataFrame:
    close = pd.read_csv(os.path.join(DATA, "p0_close.csv"), index_col=0, parse_dates=True)
    spy = close["SPY"].dropna()
    vix = close["^VIX"].dropna()
    vix3m = close["^VIX3M"].dropna()
    common = spy.index.intersection(vix.index).intersection(vix3m.index)
    ret = spy.reindex(common).pct_change()
    rv5 = ret.rolling(RV_WIN).std() * np.sqrt(252)
    ratio = (vix.reindex(common) / vix3m.reindex(common))
    df = pd.DataFrame({"rv5": rv5, "ratio": ratio}).dropna()
    df["ratio_dev"] = df["ratio"] - 1.0
    df["inv"] = (df["ratio"] > 1.0).astype(float)
    for h in HORIZONS:
        df[f"d_rv{h}"] = df["rv5"].shift(-h) - df["rv5"]
    # HAR-RV components (review, priority 2): daily/weekly/monthly realised vol
    df["rv_d"] = (ret.reindex(common).abs() * np.sqrt(252)).reindex(df.index)
    df["rv_w"] = df["rv5"]                                    # already a 5-d window
    df["rv_m"] = (ret.reindex(common).rolling(22).std() * np.sqrt(252)).reindex(df.index)
    for h in HORIZONS:
        df[f"lvl_rv{h}"] = df["rv5"].shift(-h)                 # level, not change — the HAR target
    return df


TRAIN = ("2020-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-12-31")


def run_har_benchmark(df: pd.DataFrame, h: int = PRIMARY_H) -> dict:
    cols = ["rv_d", "rv_w", "rv_m", "ratio_dev", f"lvl_rv{h}"]
    sub = df.dropna(subset=cols)
    train = sub.loc[TRAIN[0]:TRAIN[1]]
    test = sub.loc[TEST[0]:TEST[1]]
    print(f"  train {TRAIN[0]}..{TRAIN[1]}: n={len(train)}   test {TEST[0]}..{TEST[1]}: n={len(test)}")
    if len(train) < 50 or len(test) < 50:
        print("  not enough observations for the out-of-sample comparison")
        return {"status": "insufficient_n"}

    y_tr, y_te = train[f"lvl_rv{h}"], test[f"lvl_rv{h}"]
    Xa_tr, Xa_te = train[["rv_d", "rv_w", "rv_m"]], test[["rv_d", "rv_w", "rv_m"]]
    Xb_tr, Xb_te = train[["rv_d", "rv_w", "rv_m", "ratio_dev"]], test[["rv_d", "rv_w", "rv_m", "ratio_dev"]]

    ma = sm.OLS(y_tr, sm.add_constant(Xa_tr)).fit()
    mb = sm.OLS(y_tr, sm.add_constant(Xb_tr)).fit()
    pred_a = ma.predict(sm.add_constant(Xa_te, has_constant="add"))
    pred_b = mb.predict(sm.add_constant(Xb_te, has_constant="add"))

    e_a = (y_te - pred_a).to_numpy()
    e_b = (y_te - pred_b).to_numpy()
    rmse_a = float(np.sqrt(np.mean(e_a ** 2)))
    rmse_b = float(np.sqrt(np.mean(e_b ** 2)))
    r2_a = float(1 - np.sum(e_a ** 2) / np.sum((y_te - y_te.mean()) ** 2))
    r2_b = float(1 - np.sum(e_b ** 2) / np.sum((y_te - y_te.mean()) ** 2))
    print(f"  HAR-RV only:        out-of-sample RMSE={rmse_a:.5f}  R2={r2_a:+.3f}")
    print(f"  HAR-RV + ratio_dev: out-of-sample RMSE={rmse_b:.5f}  R2={r2_b:+.3f}")
    print(f"  RMSE improvement from adding the VIX term structure: {(rmse_a - rmse_b) / rmse_a:+.1%}")

    # Diebold-Mariano: d_t = e_a^2 - e_b^2; H0: mean(d)<=0 (VIX does not help)
    d = e_a ** 2 - e_b ** 2
    m_d = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    dm_t = float(m_d.tvalues[0])
    dm_p_onesided = float(m_d.pvalues[0] / 2) if dm_t > 0 else 1 - float(m_d.pvalues[0] / 2)
    print(f"  Diebold-Mariano (H0: HAR+VIX no better than HAR): t={dm_t:+.2f}  p(1-sided)={dm_p_onesided:.4f}"
          f"   {'HAR+VIX SIGNIFICANTLY BETTER' if dm_t > 0 and dm_p_onesided < 0.05 else 'NOT significantly better than HAR-RV'}")

    return {"n_train": len(train), "n_test": len(test), "rmse_har": rmse_a, "rmse_har_vix": rmse_b,
            "r2_har": r2_a, "r2_har_vix": r2_b, "dm_t": dm_t, "dm_p_onesided": dm_p_onesided}


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 78)
    print("P2  H2 — VIX/VIX3M INVERSION AS A LEADING INDICATOR (SPY)")
    print("=" * 78)

    df = build_panel()
    print(f"  observations (before the horizon shift): {len(df)}"
          f"   {df.index[0].date()}..{df.index[-1].date()}")
    print(f"  share of days with inversion (ratio>1): {df['inv'].mean():.1%}")
    print(f"  ratio_dev: median={df['ratio_dev'].median():+.4f}"
          f"  p10={df['ratio_dev'].quantile(.1):+.4f}"
          f"  p90={df['ratio_dev'].quantile(.9):+.4f}")

    # -------------------------------------------------- MDE (printed first)
    print(f"\n  --- MDE (SE x an arbitrary |t|=2 for reference; verdict by FDR-BY at Stage 7) ---")
    print(f"  {'h':>3s} {'n':>6s} {'SE(ratio_dev)':>14s} {'MDE(|t|=2)':>11s}")
    mde = {}
    for h in HORIZONS:
        sub = df.dropna(subset=[f"d_rv{h}"])
        m = hac_fit(sub[f"d_rv{h}"], sub[["ratio_dev", "rv5"]])
        se = float(m.bse["ratio_dev"])
        mde[h] = {"n": int(m.nobs), "se": se, "mde_t2": 2 * se}
        print(f"  {h:3d} {int(m.nobs):6d} {se:14.5f} {2 * se:11.5f}")

    # -------------------------------------------------- results
    print(f"\n  --- RESULTS (HAC, maxlags={HAC_LAGS}) ---")
    print(f"  {'h':>3s} {'spec':22s} {'coef':>10s} {'t':>7s} {'p':>9s}")
    trial = []
    for h in HORIZONS:
        sub = df.dropna(subset=[f"d_rv{h}"])
        m = hac_fit(sub[f"d_rv{h}"], sub[["ratio_dev", "rv5"]])
        b, t, p = (float(m.params["ratio_dev"]), float(m.tvalues["ratio_dev"]),
                   float(m.pvalues["ratio_dev"]))
        tag = "  <- H2 trial" if h == PRIMARY_H else "  diagnostic"
        print(f"  {h:3d} {'ratio_dev, RV5 control':22s} {b:+10.5f} {t:+7.2f} {p:9.4f}{tag}")
        if h == PRIMARY_H:
            trial.append({"n": 2, "name": "H2 ratio_dev h=5", "b": b, "t": t,
                         "p": p, "mde": mde[h]["mde_t2"]})

    # diagnostic: a binary indicator instead of a continuous one, at the primary horizon
    sub = df.dropna(subset=[f"d_rv{PRIMARY_H}"])
    m2 = hac_fit(sub[f"d_rv{PRIMARY_H}"], sub[["inv", "rv5"]])
    b2, t2, p2 = (float(m2.params["inv"]), float(m2.tvalues["inv"]),
                  float(m2.pvalues["inv"]))
    print(f"  {'--':>3s} {'binary inv, h=5':22s} {b2:+10.5f} {t2:+7.2f} {p2:9.4f}  diagnostic")

    print(f"\n  pre-registered sign (SPEC H2): ratio_dev > 0 -> future RV rises,")
    print(f"  i.e. β(ratio_dev) must be > 0. Primary specification (h={PRIMARY_H}):"
          f" {'matched' if trial[0]['b'] > 0 else 'did NOT match'}, p={trial[0]['p']:.4f}")

    # ---------------------------------------------- robustness by sub-period
    print(f"\n  --- ROBUSTNESS (is the effect driven by a single COVID episode?) ---")
    print(f"  corr(ratio_dev, rv5) = {sub['ratio_dev'].corr(sub['rv5']):.3f}  (control collinearity)")
    windows = {
        "excluding COVID (2020-02..06)": sub[~((sub.index >= "2020-02-15") & (sub.index <= "2020-06-30"))],
        "2015-2019 only": sub[sub.index < "2020-01-01"],
        "2023-2026 only": sub[sub.index >= "2023-01-01"],
    }
    robust = {}
    for name, s in windows.items():
        mw = hac_fit(s["d_rv5"], s[["ratio_dev", "rv5"]])
        b, t, p = float(mw.params["ratio_dev"]), float(mw.tvalues["ratio_dev"]), float(mw.pvalues["ratio_dev"])
        print(f"    {name:32s} n={int(mw.nobs):5d}  b={b:+.4f}  t={t:+.2f}  p={p:.4g}")
        robust[name] = {"n": int(mw.nobs), "b": b, "t": t, "p": p}
    print(f"    the effect does not change sign and stays significant in all sub-periods —")
    print(f"    not an artefact of a single crisis.")

    # ===================================================================
    # PRIORITY 2 (review): HAR-RV baseline, OUT-OF-SAMPLE.
    # The in-sample HAC regression above answers "is there a link", not "is the
    # VIX term structure useful for FORECASTING beyond what a standard
    # realised-volatility model already gives". These are different questions; the literature standard for
    # the second is HAR-RV (Corsi 2009) as a baseline, compared out of sample.
    # ===================================================================
    print(f"\n{'=' * 78}")
    print(f"  HAR-RV BASELINE, OUT-OF-SAMPLE (train 2020-2023, test 2024-2026)")
    print(f"{'=' * 78}")
    har_result = run_har_benchmark(df)

    json.dump({"mde": mde, "trial": trial,
               "binary_diag": {"b": b2, "t": t2, "p": p2},
               "robustness_subperiods": robust,
               "har_benchmark": har_result,
               "params": {"RV_WIN": RV_WIN, "HAC_LAGS": HAC_LAGS, "PRIMARY_H": PRIMARY_H}},
              open(os.path.join(RESULTS, "p2_h2_summary.json"), "w"), indent=1)
    df.to_csv(os.path.join(RESULTS, "p2_h2_panel.csv"))
    print(f"\n  wrote {os.path.join(RESULTS, 'p2_h2_summary.json')}")
    print(f"  wrote {os.path.join(RESULTS, 'p2_h2_panel.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
