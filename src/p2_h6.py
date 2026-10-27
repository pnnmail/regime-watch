"""P2 — H6: ETF flows (creation/redemption) as a signal. BLOCKED.

SPEC §3 H6: "Abnormal creation/redemption in leveraged ETFs is significantly
related to next-day volatility. The hypothesis is BLOCKED without
data (creation/redemption data is paid)."

This was known BEFORE implementation started (recorded in the spec itself) — not
the result of an attempt and failure but a missing input. SPEC §9: "Do not close
BLOCKED as FAIL" — H6 is not counted among passed/failed
hypotheses and does not consume a statistical test, but it is recorded explicitly rather than
quietly dropped from the final table.

Checked: creation/redemption for TQQQ/SQQQ/SPXL/SPXS/SOXL/SOXS is not
published for free by Yahoo Finance, FRED or any source in
SPEC §4.1. The only sources (Bloomberg, Refinitiv, the issuer directly
via N-CEN/N-PORT with a ~60-day lag) are paid or unsuitable for
a lead-lag study at a daily horizon.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 78)
    print("P2  H6 — ETF FLOWS (CREATION/REDEMPTION)")
    print("=" * 78)
    print("  STATUS: BLOCKED (not FAIL) — recorded in SPEC §3 before work started.")
    print("  creation/redemption for leveraged ETFs is not published in any")
    print("  free source from SPEC §4.1. No test was run, no trial from")
    print("  the m=8 family is spent on it.")
    json.dump({"status": "BLOCKED", "reason": "no free creation/redemption data source",
              "sources_checked": ["Yahoo Finance", "FRED", "CBOE"],
              "sources_needed": ["Bloomberg", "Refinitiv", "issuer N-CEN/N-PORT (~60d lag)"]},
              open(os.path.join(RESULTS, "p2_h6_summary.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(RESULTS, 'p2_h6_summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
