"""Threshold sweep and tuned configuration.

Runs the engine once per candidate threshold change, records what each change
costs (fraud cases lost) and what it buys (false positives removed), then writes
the chosen combination to rules/rules_tuned.json.

    python -m analysis.tune

The point is not to find the "best" number -- the data is synthetic, so the
optimum is an artefact of how the noise was seeded. The point is that the
trade-off is measured rather than asserted.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from analysis.evaluate import summarise
from engine.rule_engine import load_rules, run_engine

DATA = "data/transactions.csv"

# Each variant is (label, {rule_id: {param: new_value}}).
VARIANTS: list[tuple[str, dict]] = [
    ("baseline", {}),
    ("R01 -> 30,000", {"R01": {"amount_threshold": 30000}}),
    ("R01 -> 35,000", {"R01": {"amount_threshold": 35000}}),
    ("R03 -> 12,000", {"R03": {"amount_threshold": 12000}}),
    ("R03 -> 15,000", {"R03": {"amount_threshold": 15000}}),
    ("R06 -> 25,000", {"R06": {"amount_threshold": 25000}}),
]

# The shipped rule set: the three changes with the best false-positives-saved
# per fraud-case-lost. R03 stops at 12,000 because 15,000 buys a little more
# precision for a disproportionate number of missed cases -- it is kept below as
# the "aggressive" alternative so the sweep shows that contrast explicitly.
CHOSEN = {"R01": {"amount_threshold": 35000},
          "R06": {"amount_threshold": 25000},
          "R03": {"amount_threshold": 12000}}

AGGRESSIVE = {"R01": {"amount_threshold": 35000},
              "R03": {"amount_threshold": 15000},
              "R06": {"amount_threshold": 25000}}


def apply(rules: list[dict], overrides: dict) -> list[dict]:
    out = copy.deepcopy(rules)
    for rule in out:
        if rule["rule_id"] in overrides:
            rule["params"].update(overrides[rule["rule_id"]])
    return out


def main() -> int:
    base_rules = load_rules("rules/rules.json")
    raw = pd.read_csv(DATA)

    variants = VARIANTS + [
        ("shipped: R01 35k, R06 25k, R03 12k", CHOSEN),
        ("aggressive: R01 35k, R03 15k, R06 25k", AGGRESSIVE),
    ]
    rows = []
    baseline_cases = None

    for label, overrides in variants:
        rules = apply(base_rules, overrides)
        df, flags = run_engine(raw, rules)
        s = summarise(df, flags)
        if baseline_cases is None:
            baseline_cases = s["fraud_cases_caught"]
        rows.append({
            "variant": label,
            "alerts": s["alerts"],
            "precision_pct": s["precision_pct"],
            "false_pos": s["false_pos"],
            "fp_rate_pct": s["fp_rate_pct"],
            "cases_caught": s["fraud_cases_caught"],
            "cases_lost": baseline_cases - s["fraud_cases_caught"],
            "txn_recall_pct": s["recall_txn_pct"],
        })

    sweep = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    sweep.to_csv("reports/tuning_sweep.csv", index=False)
    print(sweep.to_string(index=False))

    tuned = apply(base_rules, CHOSEN)
    with open("rules/rules_tuned.json", "w", encoding="utf-8") as fh:
        json.dump({"rules": tuned}, fh, indent=2)
    print("\nwrote rules/rules_tuned.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
