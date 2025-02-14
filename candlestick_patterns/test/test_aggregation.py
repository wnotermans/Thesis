import pandas as pd

from src.aggregation import aggregate


def test_no_aggregation():
    df = pd.DataFrame([0] * 5, ["open", "high", "low", "close", "volume"])
    assert aggregate.aggregate(df, interval_minutes=1).equals(df)
