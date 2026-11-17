"""Generate the synthetic transaction dataset.

Everything is seeded with rng(42), so the output is stable and the headline
figures are reproducible: 10,000 transactions, 197 fraud transactions, ~2%.

Fraud is injected in *cases*, not rows. A velocity burst of six transactions is
one case, not six, so `fraud_case_id` groups the rows that belong to one
injected fraud event. That lets recall be measured per case (the honest number)
as well as per transaction.

Run from the project root:

    python -m data.generate_data
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

N = 10_000
N_CUST = 500
SEED = 42
OUT = "data/transactions.csv"

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Chennai"]
CHANNELS = ["UPI", "IMPS", "Card"]
START = pd.Timestamp("2026-10-01")

rng = np.random.default_rng(SEED)
cust = {f"C{i:03d}": {"home": CITIES[i % 5], "device": f"D{i:03d}"} for i in range(N_CUST)}

_case_seq = itertools.count(1)


def case_id() -> str:
    """Next fraud case identifier (F001, F002, ...)."""
    return f"F{next(_case_seq):03d}"


def pick() -> str:
    return str(rng.choice(list(cust)))


def ts(h_lo: int = 9, h_hi: int = 21) -> pd.Timestamp:
    return START + pd.Timedelta(
        days=int(rng.integers(0, 30)),
        hours=int(rng.integers(h_lo, h_hi)),
        minutes=int(rng.integers(0, 60)),
    )


def newben() -> str:
    """A beneficiary the customer has never paid before."""
    return f"NEW{int(rng.integers(10**6))}"


def row(c, t, amt, ben=None, new=0, dev=None, loc=None, status="success",
        fraud=0, ftype="", case=""):
    return dict(
        customer_id=c,
        timestamp=t,
        amount=round(float(amt), 2),
        channel=str(rng.choice(CHANNELS)),
        beneficiary_id=ben or f"{c}_B{int(rng.integers(5))}",
        is_new_beneficiary=new,
        device_id=dev or cust[c]["device"],
        location=loc or cust[c]["home"],
        status=status,
        is_fraud=fraud,
        fraud_type=ftype,
        fraud_case_id=case,
    )


# ---------------------------------------------------------------- fraud cases
fraud = []

for _ in range(25):  # R01 high amount to new beneficiary
    c = pick()
    fraud.append(row(c, ts(), rng.integers(30000, 90000), newben(), 1,
                     fraud=1, ftype="R01", case=case_id()))

for _ in range(4):  # R02 velocity: 6 txns in 6 minutes == one case
    c = pick()
    t = ts()
    f = case_id()
    for k in range(6):
        fraud.append(row(c, t + pd.Timedelta(minutes=k), rng.integers(500, 3000),
                         fraud=1, ftype="R02", case=f))

for _ in range(25):  # R03 odd hour, high amount
    c = pick()
    fraud.append(row(c, ts(1, 4), rng.integers(12000, 40000),
                     fraud=1, ftype="R03", case=case_id()))

for _ in range(6):  # R04 3 failed then success
    c = pick()
    t = ts()
    f = case_id()
    for k in range(3):
        fraud.append(row(c, t + pd.Timedelta(minutes=k), 2000, status="failed",
                         fraud=1, ftype="R04", case=f))
    fraud.append(row(c, t + pd.Timedelta(minutes=3), 2000,
                     fraud=1, ftype="R04", case=f))

for _ in range(25):  # R05 structuring just under 50,000
    c = pick()
    fraud.append(row(c, ts(), rng.integers(46000, 50000),
                     fraud=1, ftype="R05", case=case_id()))

for _ in range(25):  # R06 new device + high amount
    c = pick()
    fraud.append(row(c, ts(), rng.integers(21000, 60000), dev=f"DX{int(rng.integers(10**5))}",
                     fraud=1, ftype="R06", case=case_id()))

for _ in range(12):  # R07 two cities within 30 minutes == one case
    c = pick()
    t = ts()
    f = case_id()
    other = [x for x in CITIES if x != cust[c]["home"]][0]
    fraud.append(row(c, t, 1500, fraud=1, ftype="R07", case=f))
    fraud.append(row(c, t + pd.Timedelta(minutes=10), 1500, loc=other,
                     fraud=1, ftype="R07", case=f))

for _ in range(5):  # R08 5 transfers to the same beneficiary within an hour == one case
    c = pick()
    t = ts()
    b = newben()
    f = case_id()
    for k in range(5):
        fraud.append(row(c, t + pd.Timedelta(minutes=8 * k), rng.integers(3000, 9000), b, 1,
                         fraud=1, ftype="R08", case=f))


# ------------------------------------------------- deliberate legitimate noise
# Unusual but genuine behaviour. This is what creates false positives, and it is
# what makes the precision and false-positive columns meaningful.
normal = []

for _ in range(60):  # genuine night payments
    c = pick()
    normal.append(row(c, ts(1, 4), rng.integers(10500, 20000)))

for _ in range(40):  # genuine large payments to a new beneficiary (rent, tuition)
    c = pick()
    normal.append(row(c, ts(), rng.integers(25500, 35000), newben(), 1))

for _ in range(30):  # genuine new phone + large payment
    c = pick()
    normal.append(row(c, ts(), rng.integers(20500, 30000), dev=f"DN{int(rng.integers(10**5))}"))

for _ in range(15):  # genuine large round payments just under 50,000
    c = pick()
    normal.append(row(c, ts(), rng.integers(45000, 50000)))

while len(normal) + len(fraud) < N:  # ordinary behaviour
    c = pick()
    amt = min(rng.lognormal(7.5, 1.0), 40000)
    st = "failed" if rng.random() < 0.03 else "success"
    normal.append(row(c, ts(8, 22), amt, status=st))


df = pd.DataFrame(fraud + normal).sort_values("timestamp").reset_index(drop=True)
df.insert(0, "txn_id", [f"T{i:05d}" for i in range(len(df))])
df.to_csv(OUT, index=False)

# ------------------------------------------------------------- contract checks
fraud_rows = int(df.is_fraud.sum())
fraud_cases = int(df.loc[df.is_fraud == 1, "fraud_case_id"].nunique())

assert len(df) == N, f"expected {N} rows, got {len(df)}"
assert fraud_rows == 197, f"expected 197 fraud rows, got {fraud_rows}"
assert fraud_cases == 127, f"expected 127 fraud cases, got {fraud_cases}"
fraud_types = sorted(df.loc[df.is_fraud == 1, "fraud_type"].unique())
assert len(fraud_types) == 8, f"expected 8 fraud types (R01..R08), got {fraud_types}"

print(f"{len(df)} rows | {fraud_rows} fraud | {df.is_fraud.mean() * 100:.2f}% | "
      f"{fraud_cases} fraud cases | 8 types")
