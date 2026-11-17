"""The 24 UAT cases: three per rule -- positive, negative, boundary.

Each case is a tiny hand-built dataset and an expectation about the *last*
transaction. Parameters are fixed here rather than read from rules.json, so
tuning a threshold cannot silently invalidate the acceptance suite.
"""

import pandas as pd

BASE = pd.Timestamp("2026-10-05 12:00")
NIGHT = pd.Timestamp("2026-10-05 02:30")


def at(m=0, h=0):
    return BASE + pd.Timedelta(hours=h, minutes=m)


def tx(t=None, amt=1000, ben="B1", new=0, dev="D1", loc="Mumbai", status="success"):
    return dict(customer_id="C001", timestamp=t or at(), amount=amt, channel="UPI",
                beneficiary_id=ben, is_new_beneficiary=new, device_id=dev,
                location=loc, status=status, is_fraud=0)


P = {
    "high_amount_new_beneficiary": {"amount_threshold": 25000},
    "velocity": {"max_txns": 5, "window_minutes": 10},
    "odd_hour_high_amount": {"start_hour": 1, "end_hour": 5, "amount_threshold": 10000},
    "failed_then_success": {"min_failures": 3, "window_minutes": 30},
    "structuring": {"lower": 45000, "upper": 50000},
    "device_change_high_amount": {"amount_threshold": 20000},
    "location_change": {"window_minutes": 30},
    "repeat_beneficiary": {"max_txns": 3, "window_minutes": 60},
}


def case(tid, rtype, scenario, desc, rows, expected):
    return dict(test_id=tid, rule_id=tid.split("-")[1], type=rtype, params=P[rtype],
                scenario=scenario, desc=desc, rows=rows, expected=expected)


CASES = [
    case("UAT-R01-01", "high_amount_new_beneficiary", "positive", "30,000 to new beneficiary",
         [tx(amt=30000, new=1)], True),
    case("UAT-R01-02", "high_amount_new_beneficiary", "negative", "30,000 to known beneficiary",
         [tx(amt=30000, new=0)], False),
    case("UAT-R01-03", "high_amount_new_beneficiary", "boundary", "exactly 25,000 to new beneficiary",
         [tx(amt=25000, new=1)], False),

    case("UAT-R02-01", "velocity", "positive", "6 txns in 5 minutes",
         [tx(at(i)) for i in range(6)], True),
    case("UAT-R02-02", "velocity", "negative", "3 txns in 3 minutes",
         [tx(at(i)) for i in range(3)], False),
    case("UAT-R02-03", "velocity", "boundary", "exactly 5 txns in 5 minutes",
         [tx(at(i)) for i in range(5)], False),

    case("UAT-R03-01", "odd_hour_high_amount", "positive", "15,000 at 02:30",
         [tx(NIGHT, amt=15000)], True),
    case("UAT-R03-02", "odd_hour_high_amount", "negative", "15,000 at 12:00",
         [tx(amt=15000)], False),
    case("UAT-R03-03", "odd_hour_high_amount", "boundary", "exactly 10,000 at 02:30",
         [tx(NIGHT, amt=10000)], False),

    case("UAT-R04-01", "failed_then_success", "positive", "3 failures then success",
         [tx(at(0), status="failed"), tx(at(1), status="failed"),
          tx(at(2), status="failed"), tx(at(3))], True),
    case("UAT-R04-02", "failed_then_success", "negative", "1 failure then success",
         [tx(at(0), status="failed"), tx(at(1))], False),
    case("UAT-R04-03", "failed_then_success", "boundary", "2 failures then success",
         [tx(at(0), status="failed"), tx(at(1), status="failed"), tx(at(2))], False),

    case("UAT-R05-01", "structuring", "positive", "49,000 amount",
         [tx(amt=49000)], True),
    case("UAT-R05-02", "structuring", "negative", "20,000 amount",
         [tx(amt=20000)], False),
    case("UAT-R05-03", "structuring", "boundary", "exactly 50,000",
         [tx(amt=50000)], False),

    case("UAT-R06-01", "device_change_high_amount", "positive", "new device, 30,000",
         [tx(at(0), amt=500), tx(at(5), amt=30000, dev="D9")], True),
    case("UAT-R06-02", "device_change_high_amount", "negative", "same device, 30,000",
         [tx(at(0), amt=500), tx(at(5), amt=30000)], False),
    case("UAT-R06-03", "device_change_high_amount", "boundary", "new device, exactly 20,000",
         [tx(at(0), amt=500), tx(at(5), amt=20000, dev="D9")], False),

    case("UAT-R07-01", "location_change", "positive", "Mumbai then Delhi after 10 min",
         [tx(at(0)), tx(at(10), loc="Delhi")], True),
    case("UAT-R07-02", "location_change", "negative", "Mumbai then Delhi after 2 hours",
         [tx(at(0)), tx(at(0, h=2), loc="Delhi")], False),
    case("UAT-R07-03", "location_change", "boundary", "Delhi exactly 30 min later",
         [tx(at(0)), tx(at(30), loc="Delhi")], True),

    case("UAT-R08-01", "repeat_beneficiary", "positive", "4 transfers to same beneficiary in 30 min",
         [tx(at(i * 10)) for i in range(4)], True),
    case("UAT-R08-02", "repeat_beneficiary", "negative", "2 transfers",
         [tx(at(i * 10)) for i in range(2)], False),
    case("UAT-R08-03", "repeat_beneficiary", "boundary", "exactly 3 transfers",
         [tx(at(i * 10)) for i in range(3)], False),
]

assert len(CASES) == 24, f"expected 24 UAT cases, got {len(CASES)}"
