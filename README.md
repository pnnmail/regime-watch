# regime-watch

Can we **detect a change in market regime early** — not predict direction, but notice that a calm
market has turned restless, within a few days? A pre-registered study of 8 hypotheses on 40 global
indices, ETFs and cross-market indicators (2015–2026), with synthetic power analysis first and an
FDR-BY correction at the end.

The full specification, written before any data was looked at, is in [SPEC.md](SPEC.md).

## Results at a glance

FDR-BY, q = 0.10, family m = 8 fixed in advance. Success criterion (SPEC §7): at least 2 of 8 pass.
**3 of 6 tested hypotheses passed → criterion met.** H3 and H6 are BLOCKED (no data), not FAILED.

| # | Hypothesis | Test statistic | p | FDR-BY | Verdict |
|---|---|---|---|---|---|
| H2 | VIX/VIX3M inversion precedes a rise in SPX realised vol (h = 5 d) | t = 6.44 | 1e-10 | ✅ | **PASS** — robust without COVID (t = 7.35), 2015–19 (t = 7.98), 2023–26 (t = 2.82) |
| H7 | Asian close predicts the SPX open, **controlling for overnight ES futures** | t = 4.30 | 2e-5 | ✅ | **PASS**, but small: β fell from 0.139 to 0.030 once ES was controlled |
| H1 | Markov-switching model detects volatility regime change within 3 days | perm p | 0.0098 | ✅ | **PASS** — 9 of 11 VIX episodes caught within 3 days |
| H5 | Sector correlation break before a regime change | z = 1.81 | 0.035 | ❌ | FAIL after correction |
| H4 | Breadth (< 30% of sectors above 50-DMA) leads regime change | z = 1.80 | 0.036 | ❌ | FAIL — breadth *coincides* with onset (lead 0 d), it confirms rather than leads |
| H8 | A composite of 5 signals beats the best single signal (out of sample) | — | 0.83 | ❌ | FAIL — composite AUC 0.898 vs VIX term structure alone 0.915 |
| H3 | HY-IG credit spread leads SPX | — | — | — | **BLOCKED** — FRED only has the ICE BofA series from 2023-09; covers 1 of 11 episodes |
| H6 | Leveraged-ETF creation/redemption flows | — | — | — | **BLOCKED** — data is paid |

### The honest caveats that matter more than the pass count

- **A model-free baseline beats the model.** A naive rule "5-day realised vol > 1.75× its trailing
  median" caught 10/11 episodes within 3 days with a 7.3% false-positive rate in calm years
  (2017/2019/2023). The Markov-switching model (H1) caught 9/11 with **13.3% FPR — failing the
  ≤ 10% criterion**. Model complexity did not pay off.
- **The VIX term structure is significant in-sample but not useful out of sample.** Against a standard
  HAR-RV volatility forecast (train 2020–23, test 2024–26), adding VIX/VIX3M improved RMSE by 0.9%;
  Diebold–Mariano t = 0.11, p = 0.46. Not significantly better.
- **The "composite" failed for an informative reason:** the signals are not independent
  (|r| = 0.22–0.64). They are different views of the same thing — the current volatility level.
- **The Asia → US effect was mostly an overnight-futures effect.** Without the ES control the naive
  estimate was t = 8.99; with it, the coefficient shrinks by ~80%.

### Bugs caught by the protocol before results were published

- **Ground truth defined by the detector's own kind of series.** The first version labelled crisis
  episodes with SPY 20-day realised vol; the detectors (5-day RV) then looked *faster than the
  theoretical ceiling* (70.6% vs 40.1%). Switching ground truth to VIX — an independent instrument —
  removed the loop.
- **"Better than the ceiling" as a red flag.** Real-data detection still beat the synthetic ceiling;
  the cause was that real crises have a stress/calm vol ratio of 2.9–9.2×, beyond the 1.5–3.0× tested.
  After extending the synthetic grid to 5× and 8× the real result fell below the ceiling, as it should.
- **Pettitt test answering the wrong question.** It found 0/11 correlation breaks around crises —
  because on a rolling window it keeps locating the *first* jump, which drifts out of the "recent"
  zone. Replaced by a causal level threshold.
- **Weekend NaNs.** BTC trades 7 days a week; in a merged calendar every weekend counted sectors as
  "below MA". Fixed by restricting to sector trading days.

## Pipeline

| script | stage |
|---|---|
| `src/p0_data.py` | universe and data (Yahoo Finance, FRED), selection rule ≥ 3 years and ≥ $10M/day — 40/40 passed |
| `src/p1_synthetic.py` | Markov-switching synthetic series with known transitions: power, delay and false-positive rate of each detector **before** real data |
| `src/p2_h1.py … p2_h8.py` | one script per hypothesis; MDE printed before the estimate; walk-forward, no look-ahead |
| `src/p3_fdr.py` | final FDR-BY over the fixed family m = 8 |

Summaries of every stage are in `results/*.json`.

## Run

```bash
pip install pandas numpy scipy statsmodels scikit-learn yfinance pandas-datareader
python src/p0_data.py
python src/p1_synthetic.py
python src/p2_h1.py   # … through p2_h8.py
python src/p3_fdr.py
```
