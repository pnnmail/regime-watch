# Specification

"Detecting regime change and early signs of movement in global markets and ETFs"

Version: 1.0
Scope: global indices, ETFs, cross-market signals
Focus: detection, not prediction

## Framing: what people actually look for and what of it works

Before writing the spec, three things that are often mixed up must be separated:

1. **Predicting direction.** "Where will it go tomorrow". This is what failed in
   an earlier study of the Korean market (a separate project). In efficient markets it is
   not linearly predictable, and that is not a data bug but a property of the object.

2. **Detecting a regime change.** "A calm market has become restless". This is not
   the same thing. Here the signal does not "predict the future" but "states the
   present" — with a delay. The value lies in keeping the delay short.

3. **Early detection of the start of a move.** "The move has already started but has not yet
   become obvious". This is the most practical. Not "predict", but "notice
   earlier than the crowd".

### What works and what does not (literature and practice)

| Signal family | Status |
|---|---|
| Volatility (GARCH, HMM, realized vol) | Works for regime detection, not for direction |
| Cross-market lead-lag (credit leads equities, VIX term structure) | Works partly, the edge has decayed |
| Microstructure (OFI, VPIN) | Works at HFT horizons, unavailable without data |
| ETF flows (creation/redemption) | Works over a few days, but the data is paid |
| Breadth (% above MA, advance-decline) | Works as confirmation, not as a lead |
| Options skew / put-call | Weak, many false signals |
| Benford / distributional anomalies | Works for detecting manipulation, not movement |
| Synthetic data + ML (LSTM, Transformer) | Overfits, usually does not hold out of sample |

**Main conclusion:** look not for direction prediction but for detecting a
change in the volatility regime — because volatility clusters and
direction does not. A volatility regime change often precedes a move (without
determining its direction).

## 1. Goal

Build a reproducible pipeline that:

1. Monitors a given universe of instruments;
2. Detects a statistically significant regime change (volatility,
   correlations, flows) with minimal delay;
3. Gives no trading rules, but a "regime changed" flag with a reliability
   estimate.

Explicitly **NOT a goal**: predicting the direction of the move. Direction is a separate
task, and the previous study showed that on daily
data it cannot be solved linearly.

## 2. Universe (what to monitor)

### 2.1. Base indices (mandatory)

- SPY / SPX (US)
- QQQ / NDX (tech)
- IWM (small caps)
- EEM (emerging markets)
- EWY, FLKR (Korea) — the object of the previous study
- FXI, MCHI (China)
- EWJ (Japan)
- EWG (Germany)

### 2.2. Sector ETFs (for cross-sectional signals)

- XLK, XLF, XLE, XLV, XLI, XLP, XLU, XLB, XLRE, XLC
- SMH (semiconductors) — as a high-beta indicator

### 2.3. Cross-market indicators (mandatory)

- VIX, VVIX, VIX term structure (VIX/VIX3M)
- HYG, LQD (credit) and the HY-IG spread
- TLT, IEF (rates), DXY (dollar)
- GLD, USO (commodities)
- BTC (24/7 risk appetite)

### 2.4. Leveraged ETFs (to monitor the "volatility loop")

- TQQQ/SQQQ, SPXL/SPXS, SOXL/SOXS
- Any 2x/3x single-stock ETFs (by analogy with the Korean case)

**Selection rule:** an instrument is included if it has ≥ 3 years
of daily data and trades ≥ $10M/day. This cuts junk tickers
(a lesson from the previous study).

## 3. Hypotheses (pre-registration, m = 8)

All hypotheses are fixed before looking at the data. FDR-BY correction, q = 0.10.

**H1. Volatility regime change.**
A two-state HMM (low/high volatility) on realised
volatility detects the transition into the high-volatility regime with probability
> 0.7 1–3 days before the peak. Metric: detection delay.

**H2. VIX term structure as a leading indicator.**
VIX/VIX3M inversion (ratio > 1) significantly precedes a rise in
SPX realised volatility at a 1–5 day horizon.

**H3. The credit spread leads equities.**
HY-IG spread widening significantly precedes SPX declines (and
vice versa — narrowing leads rises) at a 1–10 day horizon. Test: lagged
cross-correlation.

**H4. Breadth as a confirming signal.**
The share of sectors above their 50-day MA falls below 30% before SPX shows
a regime change. Metric: lead time.

**H5. Correlation break.**
The average pairwise correlation within SPX sectors rises (or falls)
significantly before a volatility regime change. Method: Bai-Perron on
a rolling correlation matrix.

**H6. ETF flows as a signal.**
Abnormal creation/redemption in leveraged ETFs is significantly
related to next-day volatility. The hypothesis is **BLOCKED** without data
(creation/redemption data is paid).

**H7. Cross-market lead-lag.**
The intraday move of Asian markets (EWY, EWJ, FXI) significantly
predicts the SPX open on the same day. The classic overnight effect, but
tested on current data.

**H8. Composite indicator.**
A combination of ≥ 3 signals gives a better ROC-AUC than each one alone. Test:
logistic regression on the label "regime change in the next 5 days".

## 4. Data

### 4.1. Mandatory (free)

| Dataset | Frequency | Source |
|---|---|---|
| OHLCV for all ETFs/indices | Daily | Yahoo Finance, Stooq, Alpha Vantage |
| VIX, VIX3M, VVIX | Daily | CBOE (free) |
| HY-IG spread | Daily | FRED (BAMLH0A0HYM2, BAMLC0A0CM) |
| DXY, TLT, GLD, USO | Daily | Yahoo Finance |
| BTC | Daily | Binance, CoinGecko |
| Sector ETFs | Daily | Yahoo Finance |

### 4.2. Extensions (paid / partly available)

| Dataset | Frequency | Source | Cost |
|---|---|---|---|
| Intraday bars | 5-min | Polygon, Databento | $50–200/mo |
| ETF creation/redemption | Daily | Bloomberg, Refinitiv | Expensive |
| Options data | Daily | CBOE DataShop | Moderate |
| Limit order book | Tick | Exchange data | Expensive |

### 4.3. Synthetic data (mandatory)

Generating time series with known regime transitions (Markov-switching
GARCH, SETAR). Used to estimate the power of the methods before applying them to
real data. This addresses the small sample of crises.

## 5. Methods

### 5.1. Regime change detection

- HMM with 2–3 states (trained on a rolling window)
- Markov-switching GARCH
- CUSUM on residuals
- Pettitt test
- Bai-Perron for multiple breaks

### 5.2. Cross-market signals

- Lagged cross-correlation (H3, H7)
- Granger causality (checking for settlement cycles — a lesson from H5 of the previous
  study)
- VAR with lags 1–10
- Transfer entropy causality

### 5.3. Correlation structure

- Rolling correlation matrix
- Bai-Perron on the average correlation
- Eigenvalue analysis (the first component as an "everything moves one way" indicator)

### 5.4. Composite indicator (H8)

- Logistic regression (simple, interpretable)
- Gradient boosting with SHAP for interpretation
- LSTM autoencoder for unsupervised anomaly detection

### 5.5. Controls

- Pre-registration, m = 8, FDR-BY q = 0.10
- Synthetic data for the MDE before real data
- Walk-forward validation: train on 2020–2023, test on 2024–2026
- Look-ahead control: all signals use data up to time t
- Checking settlement cycles (residual autocorrelation)

## 6. Expected results

1. Regime change points for historical crises (2020 COVID, 2022 bear,
   2024–2026 corrections): detection date, delay, precision/recall.
2. Method comparison: ROC-AUC, precision-recall, delay.
3. Lead time for each signal: how many days/hours the signal leads
   the move.
4. Composite indicator: out-of-sample quality, if ≥ 3 signals are significant.
5. A false-positive map: in which regimes the signals lie.

Output format: not "buy/sell" but a table of flags:

```
Date | Instrument | Signal | Reliability | Lead time (days)
```

This is not a trading rule but an information layer for decisions by a human
or another system.

## 7. Success criteria

- At least 2 of 8 hypotheses confirmed at p < 0.05 after FDR-BY.
- Detection delay ≤ 3 days by at least one method on 2+ historical
  episodes.
- False positives ≤ 10% in calm periods (2017, 2019, 2023).
- Out-of-sample ROC-AUC ≥ 0.65 for the composite indicator (if it is significant).
- Reproducibility: code, data, seed, library versions.

## 8. Limitations

- Direction is not predicted. A volatility regime change does not say where
  the market will go. This is fundamental.
- A small sample of crises. 3–7 major episodes in 20 years. Synthetic data helps
  estimate power but does not create real data.
- Regime dependence. A method that worked in 2020 may not work in 2026.
  This is not a bug but a property of the market.
- Edge decay. Public signals weaken over time because
  everyone uses them.
- Microstructure is unavailable without paid data. Without it the accuracy ceiling is
  daily frequency.

## 9. What NOT to do

- Do not look for direction prediction. It failed twice. A third attempt is
  either overfitting or data mining.
- Do not formulate trading rules without out-of-sample testing.
- Do not extend m after looking at the data.
- Do not ignore settlement cycles in aggregated data.
- Do not close BLOCKED as FAIL. BLOCKED means no data, FAIL means the hypothesis
  is wrong.
- Do not trust a single method. Composite or nothing.
- Do not confuse "there is a signal" with "there is a rule". The first is descriptive statistics,
  the second is a trading decision. These are different things.

## 10. Work plan

| Stage | Content | Days |
|---|---|---|
| 1 | Fix hypotheses, collect data | 3 |
| 2 | Synthetic data: power estimation | 3 |
| 3 | H1–H2 (volatility, VIX) | 4 |
| 4 | H3, H7 (cross-market, credit) | 4 |
| 5 | H4–H5 (breadth, correlations) | 4 |
| 6 | H6, H8 (flows, composite) | 4 |
| 7 | FDR, out-of-sample, write-up | 3 |
| **Total** | | **25** |

## 11. Final framing

**What this pipeline gives:** a "regime is changing" flag with a reliability estimate and lead
time. Not "buy now" but "pay attention, something has shifted".

**What it does not give:** direction, an entry point, a trading rule.

**Main value:** the ability to check on historical data whether we had
signals before the move, and if so, with what delay. This is a scientific problem
solvable on public data.

**Main risk:** overfitting to historical crises. Against it —
synthetic data, walk-forward, FDR and pre-registration.

---

## Implementation note (environment)

Python 3.14, no C++ compiler in the environment. `hmmlearn` and `ruptures` do not
build from source on this machine — instead:

- **HMM (§5.1)** → `statsmodels.tsa.regime_switching.MarkovRegression` /
  `MarkovAutoregression` — the same model class (Markov-switching), available
  without compilation.
- **Bai-Perron (§5.2, H5)** → no ready implementation in the environment; written
  by hand (search over break points by sum of RSS, a Bai-Perron analogue for ≤3
  breaks) or replaced by CUSUM/Pettitt as the main method if resources are
  limited — this is stated explicitly in the findings, not hidden.
- **CUSUM / Pettitt** → implemented by hand (numpy/scipy), no libraries
  needed.
- **GARCH / Markov-switching GARCH** → the `arch` package (it built, there is a wheel for
  cp314) for plain GARCH; `arch` has no MS-GARCH — Markov-switching variance
  is done via `statsmodels` MarkovRegression with switching_variance=True on
  returns (the standard first-order substitute for MS-GARCH).
- **FRED** → `pandas_datareader.data.DataReader(..., 'fred', ...)` instead of
  `fredapi` (no API key needed).
