"""Contract checks for the figures quoted outside the repo (resume, README).

Deliberately NOT a pytest file: `pytest -q` must keep reporting exactly the 24
UAT cases. These are pipeline invariants, checked with:

    python -m analysis.check_contract
"""

from __future__ import annotations

import pandas as pd

from engine.rule_engine import load_rules
from tests.cases import CASES

EXPECTED = {
    "rows": 10_000,
    "fraud_rows": 197,
    "fraud_cases": 127,
    "rules": 8,
    "uat_cases": 24,
}


def main() -> int:
    df = pd.read_csv("data/transactions.csv")
    rules = load_rules("rules/rules.json")
    tuned = load_rules("rules/rules_tuned.json")

    actual = {
        "rows": len(df),
        "fraud_rows": int(df.is_fraud.sum()),
        "fraud_cases": int(df.loc[df.is_fraud == 1, "fraud_case_id"].nunique()),
        "rules": len(rules),
        "uat_cases": len(CASES),
    }

    failures = 0
    for key, want in EXPECTED.items():
        got = actual[key]
        ok = got == want
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {key:12s} expected {want:>6,}  got {got:>6,}")

    if len(tuned) != EXPECTED["rules"]:
        failures += 1
        print(f"FAIL  tuned rules  expected {EXPECTED['rules']}  got {len(tuned)}")

    types = {r["type"] for r in rules}
    if len(types) != 8:
        failures += 1
        print(f"FAIL  distinct rule implementations expected 8, got {len(types)}")

    print(f"\n{'all contracts hold' if not failures else f'{failures} contract breach(es)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
