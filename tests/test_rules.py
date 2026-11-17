"""UAT acceptance suite: 24 cases, parametrised one test per case."""

import pandas as pd
import pytest

from engine.rule_engine import RULES, _prep
from tests.cases import CASES


def actual(case):
    """True if the rule fires on the last (chronologically) row of the case."""
    frame = _prep(pd.DataFrame(case["rows"]))
    result = RULES[case["type"]](frame, case["params"])
    return bool(result.iloc[-1])


@pytest.mark.parametrize("case", CASES, ids=[c["test_id"] for c in CASES])
def test_rule(case):
    assert actual(case) == case["expected"], (
        f"{case['test_id']} ({case['scenario']}: {case['desc']}) "
        f"expected {case['expected']}, got {actual(case)}"
    )
