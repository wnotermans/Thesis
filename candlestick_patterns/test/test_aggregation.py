import pandas as pd

from src.aggregation import aggregate


def test_no_aggregation():
    df = pd.DataFrame(
        {
            "open": 0,
            "high": 1,
            "low": 2,
            "close": 3,
            "volume": 4,
        },
        index=[pd.Timestamp("1/1/1970 00:00:00")],
    )
    assert aggregate.aggregate(df, interval_minutes=1).equals(df)


def test_aggregation():
    df = pd.DataFrame(
        {
            "open": [0, 1],
            "high": [1, 2],
            "low": [0, 1],
            "close": [1, 3],
            "volume": [1, 1],
        },
        index=pd.date_range("1/1/1970", periods=2, freq="min"),
    )
    expected = pd.DataFrame(
        {
            "open": 0,
            "high": 2,
            "low": 0,
            "close": 3,
            "volume": 2,
        },
        index=[pd.Timestamp("1/1/1970 00:02:00")],
    )
    assert aggregate.aggregate(df, interval_minutes=2).equals(expected)


def test_aggregation_5_minutes():
    df = pd.DataFrame(
        {
            "open": [0, 1, 4, 3, 7],
            "high": [1, 2, 5, 3, 7],
            "low": [0, 1, 3, 1, 2],
            "close": [1, 3, 4, 7, 5],
            "volume": [1, 2, 3, 2, 1],
        },
        index=pd.date_range("1/1/1970", periods=5, freq="min"),
    )
    expected = pd.DataFrame(
        {
            "open": 0,
            "high": 7,
            "low": 0,
            "close": 5,
            "volume": 9,
        },
        index=[pd.Timestamp("1/1/1970 00:05:00")],
    )
    assert aggregate.aggregate(df).equals(expected)
