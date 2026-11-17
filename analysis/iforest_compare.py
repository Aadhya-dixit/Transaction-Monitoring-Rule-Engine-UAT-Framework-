"""Compare the rule engine against an unsupervised Isolation Forest.

    python -m analysis.iforest_compare

Writes reports/iforest_summary.json so the README generator can quote real
numbers without re-running scikit-learn.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

from engine.rule_engine import _window_count, load_rules, run_engine

DATA = "data/transactions.csv"
FEATURES = ["amount", "hour", "is_new_beneficiary", "failed", "txns_1h"]


def features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.timestamp.dt.hour
    df["failed"] = (df.status == "failed").astype(int)
    df["txns_1h"] = _window_count(df, ["customer_id"], 60)
    return df


def case_recall(df: pd.DataFrame, mask: pd.Series) -> tuple[int, int, float]:
    fraud = df.is_fraud == 1
    total = int(df.loc[fraud, "fraud_case_id"].nunique())
    caught = int(df.loc[fraud & mask, "fraud_case_id"].nunique())
    return caught, total, round(100 * caught / total, 1)


def main() -> int:
    df, _ = run_engine(pd.read_csv(DATA), load_rules())
    df = features(df)
    X = df[FEATURES]

    fraud = df.is_fraud == 1
    total_txn = int(fraud.sum())

    model = IsolationForest(contamination=0.02, random_state=42)
    df["ml_flag"] = model.fit_predict(X) == -1
    rules_flag, ml_flag = df.flagged, df.ml_flag

    rule_cases, total_cases, rule_case_pct = case_recall(df, rules_flag)
    ml_cases, _, ml_case_pct = case_recall(df, ml_flag)

    summary = {
        "contamination": 0.02,
        "features": FEATURES,
        "rows": int(len(df)),
        "rule_alerts": int(rules_flag.sum()),
        "ml_alerts": int(ml_flag.sum()),
        "both": int((rules_flag & ml_flag).sum()),
        "rules_only": int((rules_flag & ~ml_flag).sum()),
        "ml_only": int((~rules_flag & ml_flag).sum()),
        "fraud_txns": total_txn,
        "fraud_cases": total_cases,
        "rules_fraud_txn": int((fraud & rules_flag).sum()),
        "ml_fraud_txn": int((fraud & ml_flag).sum()),
        "both_fraud_txn": int((fraud & rules_flag & ml_flag).sum()),
        "ml_only_fraud_txn": int((fraud & ~rules_flag & ml_flag).sum()),
        "rules_only_fraud_txn": int((fraud & rules_flag & ~ml_flag).sum()),
        "rules_cases": rule_cases,
        "rules_case_recall_pct": rule_case_pct,
        "ml_cases": ml_cases,
        "ml_case_recall_pct": ml_case_pct,
    }

    # Sensitivity: contamination is normally unknown in production. Setting it
    # to the true fraud rate is an advantage the rules do not get.
    sensitivity = []
    for c in (0.02, 0.05, 0.10):
        flagged = IsolationForest(contamination=c, random_state=42).fit_predict(X) == -1
        caught, _, case_pct = case_recall(df, pd.Series(flagged, index=df.index))
        sensitivity.append({
            "contamination": c,
            "alerts": int(flagged.sum()),
            "fraud_txn": int((fraud & flagged).sum()),
            "case_recall_pct": case_pct,
            "cases": caught,
        })
    summary["sensitivity"] = sensitivity

    Path("reports").mkdir(exist_ok=True)
    with open("reports/iforest_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Rule alerts: {summary['rule_alerts']} | ML alerts: {summary['ml_alerts']}")
    print(f"Flagged by both: {summary['both']}")
    print(f"Rules only: {summary['rules_only']} | ML only: {summary['ml_only']}")
    print(f"Fraud txns caught by rules: {summary['rules_fraud_txn']} | "
          f"by ML: {summary['ml_fraud_txn']} | of {total_txn}")
    print(f"Fraud cases caught by rules: {rule_cases}/{total_cases} ({rule_case_pct}%) | "
          f"by ML: {ml_cases}/{total_cases} ({ml_case_pct}%)")
    print(f"Fraud ML caught but rules missed: {summary['ml_only_fraud_txn']} txns | "
          f"rules caught but ML missed: {summary['rules_only_fraud_txn']} txns")
    for s in sensitivity:
        print(f"  contamination {s['contamination']:.2f} -> {s['alerts']} alerts, "
              f"{s['cases']} cases ({s['case_recall_pct']}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
