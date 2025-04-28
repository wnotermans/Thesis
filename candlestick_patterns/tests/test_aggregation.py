import pandas as pd

from src.aggregation import aggregate

default_data = {
    "open": [100, 105, 110, 115, 120, 125],
    "high": [105, 110, 115, 120, 125, 130],
    "low": [99, 104, 109, 114, 119, 124],
    "close": [104, 109, 114, 119, 124, 129],
    "volume": [1000, 1100, 1200, 1300, 1400, 1500],
}
default_df = pd.DataFrame(
    default_data, index=pd.date_range("1/1/1970", periods=6, freq="min")
)


def test_aggregate_default_interval() -> None:
    result = aggregate.aggregate(default_df, 5)
    assert len(result) == 2
    assert result["open"].iloc[0] == 100
    assert result["high"].iloc[1] == 130
    assert result["low"].iloc[0] == 99
    assert result["close"].iloc[1] == 129
    assert result["volume"].iloc[0] == 6000


def test_aggregate_1_minute_interval() -> None:
    result = aggregate.aggregate(default_df, 1)
    assert len(result) == len(default_df)
    assert result["open"].iloc[0] == default_df["open"].iloc[0]
    assert result["high"].iloc[1] == default_df["high"].iloc[1]
    assert result["low"].iloc[2] == default_df["low"].iloc[2]
    assert result["close"].iloc[3] == default_df["close"].iloc[3]
    assert result["volume"].iloc[4] == default_df["volume"].iloc[4]


def test_aggregate_different_intervals() -> None:
    result = aggregate.aggregate(default_df, 2)
    assert len(result) == 3
    assert result["open"].iloc[0] == 100
    assert result["high"].iloc[1] == 120
    assert result["low"].iloc[2] == 119
    assert result["close"].iloc[0] == 109
    assert result["volume"].iloc[1] == 2500


def test_empty_dataframe() -> None:
    empty_df = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"], index=pd.to_datetime([])
    )
    assert aggregate.aggregate(empty_df, interval_minutes=1).empty
    assert aggregate.aggregate(empty_df, interval_minutes=2).empty
    assert aggregate.aggregate(empty_df).empty


def test_aggregate_single_row() -> None:
    single_row_data = {
        "open": [100],
        "high": [105],
        "low": [99],
        "close": [104],
        "volume": [1000],
    }
    single_row_df = pd.DataFrame(single_row_data, index=pd.to_datetime(["1/1/1970"]))
    result = aggregate.aggregate(single_row_df, 5)
    assert len(result) == 1
    assert result["open"].iloc[0] == 100
    assert result["high"].iloc[0] == 105
    assert result["low"].iloc[0] == 99
    assert result["close"].iloc[0] == 104
    assert result["volume"].iloc[0] == 1000
