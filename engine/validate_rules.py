"""Schema validation for rule configurations.

A typo in a rule type or a missing parameter currently produces zero alerts
rather than an error, which is the worst possible failure mode for a monitoring
system: silence looks like safety. This makes bad configuration loud.

    python -m engine.validate_rules [path]
"""

from __future__ import annotations

import re
import sys

from engine.rule_engine import RULES

EXPECTED_PARAMS = {
    "high_amount_new_beneficiary": {"amount_threshold"},
    "velocity": {"max_txns", "window_minutes"},
    "odd_hour_high_amount": {"start_hour", "end_hour", "amount_threshold"},
    "failed_then_success": {"min_failures", "window_minutes"},
    "structuring": {"lower", "upper"},
    "device_change_high_amount": {"amount_threshold"},
    "location_change": {"window_minutes"},
    "repeat_beneficiary": {"max_txns", "window_minutes"},
}

REQUIRED_FIELDS = {"rule_id", "name", "type", "params", "severity", "enabled"}
SEVERITIES = {"low", "medium", "high"}


def validate(rules: list[dict]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()

    for r in rules:
        rid = r.get("rule_id", "<missing>")
        missing = REQUIRED_FIELDS - set(r)
        if missing:
            problems.append(f"{rid}: missing fields {sorted(missing)}")
            continue

        if not re.fullmatch(r"R\d{2}", rid):
            problems.append(f"{rid}: rule_id must look like R01")
        if rid in seen:
            problems.append(f"{rid}: duplicate rule_id")
        seen.add(rid)

        if r["type"] not in RULES:
            problems.append(f"{rid}: unknown type {r['type']!r}; known: {sorted(RULES)}")
            continue

        expected = EXPECTED_PARAMS[r["type"]]
        got = set(r["params"])
        if expected - got:
            problems.append(f"{rid}: missing params {sorted(expected - got)}")
        if got - expected:
            problems.append(f"{rid}: unknown params {sorted(got - expected)}")
        for key, value in r["params"].items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"{rid}: param {key!r} must be numeric, got {type(value).__name__}")

        if r["severity"] not in SEVERITIES:
            problems.append(f"{rid}: severity must be one of {sorted(SEVERITIES)}")
        if not isinstance(r["enabled"], bool):
            problems.append(f"{rid}: enabled must be true/false")

        if r["type"] == "odd_hour_high_amount" and r["params"].get("start_hour", 0) >= r["params"].get("end_hour", 0):
            problems.append(f"{rid}: start_hour must be < end_hour")
        if r["type"] == "structuring" and r["params"].get("lower", 0) >= r["params"].get("upper", 0):
            problems.append(f"{rid}: lower must be < upper")

    if len(rules) != 8:
        problems.append(f"expected 8 rules, found {len(rules)}")
    missing_types = set(EXPECTED_PARAMS) - {r.get("type") for r in rules}
    if missing_types:
        problems.append(f"no rule covers {sorted(missing_types)}")

    return problems


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "rules/rules.json"
    from engine.rule_engine import load_rules

    rules = load_rules(path)
    problems = validate(rules)
    for p in problems:
        print(f"  ERROR {p}")
    if problems:
        print(f"{len(problems)} problem(s) in {path}")
        return 1
    print(f"{len(rules)} rules valid, 0 warnings ({path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
