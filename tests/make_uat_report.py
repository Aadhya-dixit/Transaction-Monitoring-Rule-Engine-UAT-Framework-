"""Write the UAT pass/fail report to reports/uat_report.csv.

    python -m tests.make_uat_report
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.cases import CASES
from tests.test_rules import actual

Path("reports").mkdir(exist_ok=True)

rows = []
for c in CASES:
    observed = actual(c)
    rows.append({
        "test_id": c["test_id"],
        "rule": c["rule_id"],
        "scenario": c["scenario"],
        "description": c["desc"],
        "expected": c["expected"],
        "actual": observed,
        "result": "PASS" if observed == c["expected"] else "FAIL",
    })

report = pd.DataFrame(rows)
report.to_csv("reports/uat_report.csv", index=False)

print(report.result.value_counts().to_string())
print()
print(f"cases: {len(report)} | rules covered: {report.rule.nunique()} | "
      f"scenarios: {report.scenario.nunique()}")
if (report.result == "FAIL").any():
    print("\nFailures:")
    print(report[report.result == "FAIL"].to_string(index=False))
