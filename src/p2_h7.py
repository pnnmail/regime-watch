"""P2 — H7: Asian markets during the day predict the SPX open.

SPEC §3 H7: "The intraday move of Asian markets (EWY, EWJ, FXI)
significantly predicts the SPX open on the same day."

A DEPARTURE FROM THE LETTER OF THE SPEC, STATED EXPLICITLY: EWY/EWJ/FXI are US ETFs,
traded in US hours (9:30-16:00 ET), NOT in Asian local hours. Their
daily return is the return OVER THE US session (it already includes
yesterday's Asian reaction via arbitrage), not "the move of the Asian market
during the day" as the wording literally requires. The test "Asia close ->
US open" (the classic overnight effect mentioned in the SPEC itself)
technically requires NATIVE indices (^KS11, ^N225, ^HSI) trading in
local hours, which by calendar date close BEFORE the SPX open
(Korea/Tokyo UTC+9 — their day session is night in UTC, long before
the 13:30 UTC NYSE open). The native indices are used as the primary
specification; EWY/EWJ/FXI as a diagnostic, for literal compliance with
the ticker list in SPEC §2.1/§3.

Date alignment: Asia's return for CALENDAR day D (their local
session) is matched with the SPX gap open(D) relative to close(D-1) — this
is the lead-lag, because the Asian session ends in UTC before the
NYSE open of the same calendar day.
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

START, END = "2015-01-01", "2026-09-13"
ASIA_NATIVE = ["^KS11", "^N225", "^HSI"]
ASIA_ETF = ["EWY", "EWJ", "FXI"]
HAC_LAGS = 3


def fetch_ohlc(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    d = yf.download(tickers, start=START, end=END, auto_adjust=True,
                     progress=False, group_by="ticker", threads=True, timeout=120)
    out = {}
    for t in tickers:
        out[t] = d[t][["Open", "Close"]].dropna()
    return out


def hac_fit(y, X):
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def main() -> int:
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 78)
    print("P2  H7 — ASIA (CLOSE) -> SPX OPEN")
    print("=" * 78)

    print(f"  loading SPX (Open is needed, not just Close) and native Asian indices...")
    spx = fetch_ohlc(["^GSPC"])["^GSPC"]
    gap = (spx["Open"] - spx["Close"].shift(1)) / spx["Close"].shift(1)
    gap.name = "gap_spx"
    gap.to_csv(os.path.join(DATA, "p2_h7_spx_gap.csv"))

    asia = fetch_ohlc(ASIA_NATIVE)
    asia_ret = pd.DataFrame({t: asia[t]["Close"].pct_change() for t in ASIA_NATIVE})
    asia_ret.to_csv(os.path.join(DATA, "p2_h7_asia_native_ret.csv"))

    close_etf = pd.read_csv(os.path.join(DATA, "p0_close.csv"), index_col=0, parse_dates=True)
    etf_ret = close_etf[ASIA_ETF].pct_change()

    panel = pd.concat([gap, asia_ret, etf_ret], axis=1).dropna(
        subset=["gap_spx"] + ASIA_NATIVE)
    panel["asia_native_avg"] = panel[ASIA_NATIVE].mean(axis=1)
    print(f"  observations (native indices + gap): {len(panel)}"
          f"   {panel.index[0].date()}..{panel.index[-1].date()}")

    for t in ASIA_NATIVE:
        c = panel["gap_spx"].corr(panel[t])
        print(f"    corr(gap_spx, {t:6s}) = {c:+.3f}   n={panel[[t, 'gap_spx']].dropna().shape[0]}")

    # -------------------------------------------------- MDE first
    print(f"\n  --- MDE (SE x |t|=2, reference; FDR-BY verdict at Stage 7) ---")
    sub = panel.dropna(subset=["asia_native_avg", "gap_spx"])
    m0 = hac_fit(sub["gap_spx"], sub[["asia_native_avg"]])
    se0 = float(m0.bse["asia_native_avg"])
    print(f"    primary (asia_native_avg): n={int(m0.nobs)}  SE={se0:.5f}  MDE={2 * se0:.5f}")

    # -------------------------------------------------- primary result
    print(f"\n  --- RESULT: naive specification (mean of 3 native indices, NO control) ---")
    b, t, p = (float(m0.params["asia_native_avg"]), float(m0.tvalues["asia_native_avg"]),
               float(m0.pvalues["asia_native_avg"]))
    print(f"    gap_spx ~ asia_native_avg:   β={b:+.4f}  t={t:+.2f}  p={p:.4g}"
          f"   (replaced by the ES control below as the official H7, see review priority 1)")
    trial = [{"n": 7, "name": "H7 asia_native_avg", "b": b, "t": t, "p": p, "mde": 2 * se0,
              "naive_full_sample": {"n": int(m0.nobs), "b": b, "t": t, "p": p}}]

    print(f"\n  --- DIAGNOSTIC: indices separately ---")
    diag = {}
    for t_ in ASIA_NATIVE:
        s = panel.dropna(subset=[t_, "gap_spx"])
        mm = hac_fit(s["gap_spx"], s[[t_]])
        bb, tt, pp = float(mm.params[t_]), float(mm.tvalues[t_]), float(mm.pvalues[t_])
        print(f"    {t_:8s}  n={int(mm.nobs):5d}  β={bb:+.4f}  t={tt:+.2f}  p={pp:.4g}")
        diag[t_] = {"n": int(mm.nobs), "b": bb, "t": tt, "p": pp}

    print(f"\n  --- DIAGNOSTIC: literally per the SPEC (EWY/EWJ/FXI, US session, in context")
    print(f"      they already include yesterday's Asian reaction via arbitrage) ---")
    for t_ in ASIA_ETF:
        s = panel.dropna(subset=[t_, "gap_spx"])
        if len(s) < 30:
            print(f"    {t_:6s}: n={len(s)} < 30, skipped")
            continue
        mm = hac_fit(s["gap_spx"], s[[t_]])
        bb, tt, pp = float(mm.params[t_]), float(mm.tvalues[t_]), float(mm.pvalues[t_])
        print(f"    {t_:6s}  n={int(mm.nobs):5d}  β={bb:+.4f}  t={tt:+.2f}  p={pp:.4g}")
        diag[t_] = {"n": int(mm.nobs), "b": bb, "t": tt, "p": pp}

    print(f"\n  pre-registered sign (SPEC H7): β(asia_native_avg) > 0"
          f" (Asia up -> SPX opens higher):"
          f" {'matched' if b > 0 else 'did NOT match'}, p={p:.4g}")

    # ===================================================================
    # PRIORITY 1 (review): control for ES futures. Between the Asian close
    # (UTC ~06:30) and the NYSE open (UTC ~13:30-14:30) lies the whole
    # European session and ~7-8h of continuous ES=F trading. If β(asia)
    # does not survive a control for what already happened in ES during that time,
    # Asia is not a source of information but an indicator correlated with the overall overnight
    # move, which ES reflects more precisely.
    # ===================================================================
    print(f"\n{'=' * 78}")
    print(f"  CONTROL FOR ES=F (E-mini S&P 500 futures)")
    print(f"{'=' * 78}")

    es_result = run_es_control(panel, sub)
    trial[0]["es_control"] = es_result

    # The OFFICIAL H7 specification after review: ES-controlled, window
    # 2024-04..2026-09 (shorter than the full history — the control needs 1h bars,
    # yfinance provides them only for ~730 days). The naive full-sample estimate
    # above is confirmed as INFLATED (β drops from 0.139 to 0.030 on THE SAME
    # sub-period when ES is included) — using it as the H7 trial would
    # be dishonest given the confound found.
    prec = es_result.get("precise_recent_window", {})
    if prec.get("status") not in ("insufficient_n", "error") and "p_asia_with_control" in prec:
        trial[0]["b"] = prec["b_asia_with_control"]
        trial[0]["t"] = prec["t_asia_with_control"]
        trial[0]["p"] = prec["p_asia_with_control"]
        trial[0]["n_obs"] = prec["n"]                          # NOT "n" -- that key is reserved for the hypothesis number (7) in p3_fdr.py
        trial[0]["name"] = f"H7 asia_native_avg|ES-control ({prec['start']}..{prec['end']})"
        print(f"\n  OFFICIAL H7 (after review, priority 1) — ES-controlled specification,")
        print(f"  window {prec['start']}..{prec['end']} (n={prec['n']}, shorter than the full history):")
        print(f"    β={prec['b_asia_with_control']:+.4f}  t={prec['t_asia_with_control']:+.2f}"
              f"  p={prec['p_asia_with_control']:.4g}  <- H7 trial (replaces the naive one)")
    else:
        print(f"\n  ES control not performed ({prec.get('status')}) — H7 stays with the naive estimate,")
        print(f"  EXPLICITLY MARKED as not checked for the ES-futures confound.")

    json.dump({"trial": trial, "diagnostics": diag,
               "corr": {t_: float(panel["gap_spx"].corr(panel[t_])) for t_ in ASIA_NATIVE},
               "params": {"HAC_LAGS": HAC_LAGS}, "es_control": es_result},
              open(os.path.join(RESULTS, "p2_h7_summary.json"), "w"), indent=1)
    panel.to_csv(os.path.join(RESULTS, "p2_h7_panel.csv"))
    print(f"\n  wrote {os.path.join(RESULTS, 'p2_h7_summary.json')}")
    print(f"  wrote {os.path.join(RESULTS, 'p2_h7_panel.csv')}")
    return 0


def run_es_control(panel: pd.DataFrame, primary_sub: pd.DataFrame) -> dict:
    """Returns {} in parts: a crude full-sample control + a precise one on
    the available window of ES=F hourly bars (yfinance limits 1h history to
    ~730 days — the window is named explicitly, not passed off as the full sample)."""
    import yfinance as yf
    result: dict = {}

    # ---------------- crude, over the whole history: ES Close(t) vs Close(t-1)
    # WARNING: this is NOT the overnight window — Close(t) includes the RTH session
    # of day t, i.e. it is partly CONTAMINATED by what happens AFTER
    # the SPX open. This makes the control STRICTER than needed (over-control), not
    # weaker — if asia survives even this, the lead-lag is more convincing.
    print(f"\n  --- (a) CRUDE control, full history: ES Close(t) vs Close(t-1) ---")
    print(f"      (contaminated by day t's RTH session — over-control, not reduced strictness)")
    es_d = yf.download("ES=F", start=START, end=END, auto_adjust=True,
                       progress=False, timeout=90)
    if isinstance(es_d.columns, pd.MultiIndex):
        es_d.columns = es_d.columns.get_level_values(0)
    gap_es_full = (es_d["Close"] - es_d["Close"].shift(1)) / es_d["Close"].shift(1)
    s = panel.join(gap_es_full.rename("es_full"), how="inner").dropna(
        subset=["gap_spx", "asia_native_avg", "es_full"])
    m = hac_fit(s["gap_spx"], s[["asia_native_avg", "es_full"]])
    b, t, p = (float(m.params["asia_native_avg"]), float(m.tvalues["asia_native_avg"]),
               float(m.pvalues["asia_native_avg"]))
    print(f"      n={int(m.nobs)}  β(asia|es_full)={b:+.4f}  t={t:+.2f}  p={p:.4g}"
          f"   {'SURVIVED' if p < 0.05 else 'VANISHED'}")
    result["crude_full_sample"] = {"n": int(m.nobs), "b": b, "t": t, "p": p}

    # ---------------- precise: ES 09:00 ET vs ES ~16:00 ET previous day, 1h bars
    print(f"\n  --- (b) PRECISE control: ES @ ~09:00 ET vs ES @ ~16:00 ET previous day ---")
    print(f"      yfinance 1h bars are available only ~730 days back — a shorter window")
    try:
        es1h = yf.download("ES=F", interval="1h", period="730d", progress=False,
                           auto_adjust=True, timeout=90)
        if isinstance(es1h.columns, pd.MultiIndex):
            es1h.columns = es1h.columns.get_level_values(0)
        es1h.index = es1h.index.tz_convert("America/New_York")
        pre_open = es1h[es1h.index.hour == 9]["Open"].copy()
        pre_open.index = pre_open.index.normalize().tz_localize(None)
        prior_close = es1h[es1h.index.hour == 15]["Close"].copy()
        prior_close.index = prior_close.index.normalize().tz_localize(None)
        po = pre_open.groupby(pre_open.index).last()
        pc = prior_close.groupby(prior_close.index).last()
        common_dates = po.index
        es_overnight = pd.Series(index=common_dates, dtype=float)
        pc_dates = pc.index.to_list()
        for d in common_dates:
            prior = [x for x in pc_dates if x < d]
            if not prior:
                continue
            pd_ = max(prior)
            if (d - pd_).days > 5:
                continue
            es_overnight[d] = (po[d] - pc[pd_]) / pc[pd_]
        es_overnight = es_overnight.dropna()
        print(f"      coverage: {es_overnight.index.min().date()}..{es_overnight.index.max().date()}"
              f"   n_raw={len(es_overnight)}")

        s2 = panel.join(es_overnight.rename("es_precise"), how="inner").dropna(
            subset=["gap_spx", "asia_native_avg", "es_precise"])
        if len(s2) < 50:
            print(f"      n={len(s2)} < 50 — not enough for a conclusion")
            result["precise_recent_window"] = {"n": len(s2), "status": "insufficient_n"}
        else:
            m2 = hac_fit(s2["gap_spx"], s2[["asia_native_avg", "es_precise"]])
            b2, t2, p2 = (float(m2.params["asia_native_avg"]), float(m2.tvalues["asia_native_avg"]),
                         float(m2.pvalues["asia_native_avg"]))
            be, te, pe = (float(m2.params["es_precise"]), float(m2.tvalues["es_precise"]),
                         float(m2.pvalues["es_precise"]))
            print(f"      n={int(m2.nobs)}  {s2.index.min().date()}..{s2.index.max().date()}")
            print(f"      β(asia|es_precise) = {b2:+.4f}  t={t2:+.2f}  p={p2:.4g}"
                  f"   {'SURVIVED' if p2 < 0.05 else 'VANISHED'}")
            print(f"      β(es_precise)      = {be:+.4f}  t={te:+.2f}  p={pe:.4g}  (for comparison)")
            # without the control, same sub-period — to see the effect of the control itself
            m3 = hac_fit(s2["gap_spx"], s2[["asia_native_avg"]])
            b3, t3, p3 = (float(m3.params["asia_native_avg"]), float(m3.tvalues["asia_native_avg"]),
                         float(m3.pvalues["asia_native_avg"]))
            print(f"      (for comparison, WITHOUT control, same sub-period): β={b3:+.4f} t={t3:+.2f} p={p3:.4g}")
            result["precise_recent_window"] = {
                "n": int(m2.nobs), "start": str(s2.index.min().date()), "end": str(s2.index.max().date()),
                "b_asia_with_control": b2, "t_asia_with_control": t2, "p_asia_with_control": p2,
                "b_es": be, "t_es": te, "p_es": pe,
                "b_asia_without_control_same_subsample": b3, "t_asia_without_control_same_subsample": t3,
                "p_asia_without_control_same_subsample": p3,
            }
    except Exception as e:                                          # noqa: BLE001
        print(f"      not performed: {type(e).__name__}: {e}")
        result["precise_recent_window"] = {"status": "error", "error": str(e)}

    return result


if __name__ == "__main__":
    sys.exit(main())
