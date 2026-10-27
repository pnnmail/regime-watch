"""P3 — the final multiple-testing correction (Stage 7 of the plan). SPEC §3, §5.5, §7.

The only correction in this project is FDR-BY (Benjamini-Yekutieli) at
q=0.10, the family m=8 was fixed BEFORE looking at the results (SPEC §3). H6 is
BLOCKED — it consumes no p-value but stays in the m=8 denominator, because
the family was fixed in advance (SPEC §9: BLOCKED != FAIL, do not recompute
m after the fact).

The success criterion for the whole project (SPEC §7): at least 2 of 8 hypotheses confirmed
at p<0.05 AFTER FDR-BY.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
M = 8
Q = 0.10

SOURCES = {
    1: ("p2_h1_trial.json", None),
    2: ("p2_h2_summary.json", "trial"),
    3: ("p2_h3_summary.json", "trial"),
    4: ("p2_h4_summary.json", "trial"),
    5: ("p2_h5_summary.json", "trial"),
    7: ("p2_h7_summary.json", "trial"),
    8: ("p2_h8_summary.json", "trial"),
}


def by_threshold(rank: int, m: int = M, q: float = Q) -> float:
    H = sum(1.0 / i for i in range(1, m + 1))
    return q * rank / (m * H)


def load_trial(hnum: int) -> list[dict]:
    fname, key = SOURCES[hnum]
    path = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        print(f"  H{hnum}: {fname} not found — not run")
        return []
    data = json.load(open(path))
    return data if key is None else data.get(key, [])


def main() -> int:
    print("=" * 78)
    print(f"P3  FINAL MULTIPLE-TESTING CORRECTION   FDR-BY q={Q}   m={M}")
    print("=" * 78)

    h6_path = os.path.join(RESULTS, "p2_h6_summary.json")
    h6_status = json.load(open(h6_path))["status"] if os.path.exists(h6_path) else "not run"
    print(f"  H6: {h6_status} — not counted in PASS/FAIL, but m={M} is not recomputed"
          f" (SPEC §9, the family was fixed in advance)")

    out = []
    for hnum in sorted(SOURCES):
        for r in load_trial(hnum):
            out.append(r)

    n_missing = M - len(out) - 1   # minus H6, which by definition has no p-value
    extra = f", {n_missing} more without a p-value for another reason" if n_missing > 0 else ""
    print(f"\n  trials with a p-value: {len(out)} of {M} (H6 BLOCKED{extra})")

    ps = sorted(out, key=lambda r: r["p"])
    print(f"\n  {'#':>2} {'trial':28s} {'b/coef':>9s} {'t/z':>7s} {'p':>9s}"
          f" {'BY crit.':>9s} {'FDR':>6s}  result")
    n_pass = 0
    for i, r in enumerate(ps, 1):
        crit = by_threshold(i)
        fdr_pass = r["p"] <= crit
        n_pass += int(fdr_pass)
        verdict = "PASS" if fdr_pass else "FAIL"
        print(f"  {r['n']:2} {r['name'][:28]:28s} {r['b']:+9.4f} {r['t']:+7.2f}"
              f" {r['p']:9.4f} {crit:9.4f} {'yes' if fdr_pass else 'no':>6s}  {verdict}")

    print(f"\n  PASSED FDR-BY: {n_pass} of {len(out)} tested"
          f" (SPEC §7 success criterion: >= 2 of 8)")
    print(f"  {'ACHIEVED' if n_pass >= 2 else 'NOT achieved'}")

    json.dump({"m": M, "q": Q, "n_tested": len(out), "n_pass_fdr": n_pass,
               "h6_status": h6_status,
               "results": [{**r, "by_crit": by_threshold(i + 1),
                           "fdr_pass": r["p"] <= by_threshold(i + 1)}
                          for i, r in enumerate(ps)]},
              open(os.path.join(RESULTS, "p3_fdr_final.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(RESULTS, 'p3_fdr_final.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
