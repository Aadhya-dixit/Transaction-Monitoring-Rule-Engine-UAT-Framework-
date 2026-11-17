"""Per-rule performance and overall alert metrics.

    python -m analysis.evaluate                       # baseline rules.json
    python -m analysis.evaluate rules/rules_tuned.json

Writes reports/rule_performance_<tag>.csv and reports/summary_<tag>.json.
The baseline run also writes reports/alerts.csv, the alert queue that a
downstream GenAI summariser would consume.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from engine.rule_engine import load_rules, run_engine

DATA = "data/transactions.csv"


def per_rule(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    fraud = df.is_fraud == 1
    legit = ~fraud
    rows = []
    for rid in flags.columns:
        hit = flags[rid]
        hits = int(hit.sum())
        tp = int((hit & fraud).sum())
        fp = int((hit & legit).sum())
        rows.append({
            "rule": rid,
            "hits": hits,
            "true_pos": tp,
            "false_pos": fp,
            "precision": round(tp / hits, 3) if hits else None,
            "fp_rate_pct": round(100 * fp / int(legit.sum()), 3),
        })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, flags: pd.DataFrame) -> dict:
    """Alert-level metrics.

    Two recall figures on purpose. `recall_txn_pct` counts every row labelled
    fraud, but a burst rule only fires on the triggering row, so it understates
    detection. `recall_case_pct` asks the question an investigator actually
    cares about: was the fraud *event* surfaced at all?
    """
    fraud = df.is_fraud == 1
    legit = ~fraud
    alerts = int(df.flagged.sum())
    caught_txn = int((df.flagged & fraud).sum())
    total_txn = int(fraud.sum())
    total_cases = int(df.loc[fraud, "fraud_case_id"].nunique())
    caught_cases = int(df.loc[fraud & df.flagged, "fraud_case_id"].nunique())
    false_pos = int((df.flagged & legit).sum())

    return {
        "alerts": alerts,
        "fraud_txns": total_txn,
        "fraud_caught_txn": caught_txn,
        "recall_txn_pct": round(100 * caught_txn / total_txn, 1),
        "fraud_cases": total_cases,
        "fraud_cases_caught": caught_cases,
        "recall_case_pct": round(100 * caught_cases / total_cases, 1),
        "false_pos": false_pos,
        "precision_pct": round(100 * caught_txn / alerts, 1),
        "fp_rate_pct": round(100 * false_pos / int(legit.sum()), 3),
        "rules_evaluated": int(flags.shape[1]),
    }


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "rules/rules.json"
    tag = "tuned" if "tuned" in path else "baseline"

    rules = load_rules(path)
    df, flags = run_engine(pd.read_csv(DATA), rules)

    perf = per_rule(df, flags)
    summary = summarise(df, flags)

    print(perf.to_string(index=False))
    print()
    print(f"Total alerts: {summary['alerts']} | "
          f"precision {summary['precision_pct']}% | "
          f"FP rate {summary['fp_rate_pct']}% of legitimate txns")
    print(f"Fraud cases caught: {summary['fraud_cases_caught']}/{summary['fraud_cases']} "
          f"({summary['recall_case_pct']}%) | "
          f"transactions: {summary['fraud_caught_txn']}/{summary['fraud_txns']} "
          f"({summary['recall_txn_pct']}%)")

    Path("reports").mkdir(exist_ok=True)
    perf.to_csv(f"reports/rule_performance_{tag}.csv", index=False)
    with open(f"reports/summary_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump({"path": path, **summary}, fh, indent=2)

    if tag == "baseline":
        names = {r["rule_id"]: r["name"] for r in rules}
        alerts = df[df.flagged].copy()
        alerts["rule_fired"] = alerts.fired_rules.apply(lambda ids: ",".join(ids))
        alerts["rule_description"] = alerts.fired_rules.apply(
            lambda ids: "; ".join(names[r] for r in ids)
        )
        alerts.rename(columns={"timestamp": "time"})[
            ["txn_id", "customer_id", "amount", "channel", "time",
             "rule_fired", "rule_description"]
        ].to_csv("reports/alerts.csv", index=False)
        print(f"wrote reports/alerts.csv ({len(alerts)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
