"""Transaction monitoring rule engine.

Rules are declared in JSON (`rules/rules.json`) and dispatched here by their
`type` field. Every threshold and window lives in the JSON, never in this
module -- that is the point of the design, because rules change far more often
than code does.

Every rule function returns a boolean Series aligned to the input frame's index.
True means "this row triggered the rule".
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps and order by customer then time.

    Ordering matters: several rules compare a transaction against the customer's
    previous transaction, and the window counters assume chronological order
    within each group.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["customer_id", "timestamp"])


def _window_count(df: pd.DataFrame, keys, minutes: int) -> pd.Series:
    """Transactions by the same key inside the trailing window.

    The count includes the current transaction, so a result of 6 means "this is
    the 6th transaction in the window". Trailing means `timestamp - window`,
    inclusive of the boundary.
    """
    out = pd.Series(0, index=df.index, dtype=int)
    limit = np.timedelta64(int(minutes), "m")
    for _, g in df.groupby(keys, sort=False):
        t = g["timestamp"].values
        start = 0
        counts = []
        for i in range(len(t)):
            while t[i] - t[start] > limit:
                start += 1
            counts.append(i - start + 1)
        out.loc[g.index] = counts
    return out


# --------------------------------------------------------------------- rules
# Threshold comparisons are deliberately strict (`>`, not `>=`) so the boundary
# cases in the UAT suite have a defined, testable answer: "exactly the
# threshold" does not fire.


def high_amount_new_beneficiary(df: pd.DataFrame, p: dict) -> pd.Series:
    return (df.amount > p["amount_threshold"]) & (df.is_new_beneficiary == 1)


def velocity(df: pd.DataFrame, p: dict) -> pd.Series:
    return _window_count(df, ["customer_id"], p["window_minutes"]) > p["max_txns"]


def odd_hour_high_amount(df: pd.DataFrame, p: dict) -> pd.Series:
    hour = df.timestamp.dt.hour
    return hour.between(p["start_hour"], p["end_hour"] - 1) & (df.amount > p["amount_threshold"])


def failed_then_success(df: pd.DataFrame, p: dict) -> pd.Series:
    """A success preceded by >= min_failures failures inside the window."""
    out = pd.Series(False, index=df.index)
    window = pd.Timedelta(minutes=p["window_minutes"])
    for _, g in df.groupby("customer_id"):
        for i in range(len(g)):
            current = g.iloc[i]
            if current.status != "success":
                continue
            prior = g.iloc[:i]
            prior = prior[(prior.status == "failed") &
                          (current.timestamp - prior.timestamp <= window)]
            if len(prior) >= p["min_failures"]:
                out[g.index[i]] = True
    return out


def structuring(df: pd.DataFrame, p: dict) -> pd.Series:
    """Single transaction inside the band just below the reporting limit.

    Note: this is the simplified interpretation -- one transaction in the band.
    True structuring is *repeated* sub-threshold activity; see the README
    limitations section.
    """
    return (df.amount >= p["lower"]) & (df.amount < p["upper"])


def device_change_high_amount(df: pd.DataFrame, p: dict) -> pd.Series:
    prev_device = df.groupby("customer_id")["device_id"].shift()
    return prev_device.notna() & (df.device_id != prev_device) & (df.amount > p["amount_threshold"])


def location_change(df: pd.DataFrame, p: dict) -> pd.Series:
    grouped = df.groupby("customer_id")
    prev_loc = grouped["location"].shift()
    prev_time = grouped["timestamp"].shift()
    gap = df.timestamp - prev_time
    return (prev_loc.notna() & (df.location != prev_loc) &
            (gap <= pd.Timedelta(minutes=p["window_minutes"])))


def repeat_beneficiary(df: pd.DataFrame, p: dict) -> pd.Series:
    return _window_count(df, ["customer_id", "beneficiary_id"], p["window_minutes"]) > p["max_txns"]


RULES = {
    f.__name__: f
    for f in [
        high_amount_new_beneficiary,
        velocity,
        odd_hour_high_amount,
        failed_then_success,
        structuring,
        device_change_high_amount,
        location_change,
        repeat_beneficiary,
    ]
}


def load_rules(path: str = "rules/rules.json") -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["rules"]


def run_engine(df: pd.DataFrame, rules: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every enabled rule.

    Returns the sorted frame with `fired_rules` (list of rule ids) and `flagged`
    columns added, plus a boolean frame with one column per rule.
    """
    unknown = [r["rule_id"] for r in rules if r["type"] not in RULES]
    if unknown:
        raise KeyError(f"unknown rule type for {unknown}; known: {sorted(RULES)}")

    df = _prep(df)

    flags = {
        r["rule_id"]: RULES[r["type"]](df, r["params"]).fillna(False).astype(bool)
        for r in rules
        if r.get("enabled", True)
    }
    flag_df = pd.DataFrame(flags, index=df.index)

    columns = flag_df.columns.to_numpy()
    matrix = flag_df.to_numpy()
    df["fired_rules"] = [columns[row].tolist() for row in matrix]
    df["flagged"] = df["fired_rules"].str.len() > 0
    return df, flag_df
