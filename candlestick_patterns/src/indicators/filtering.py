import pandas as pd


def filter_indicators(df: pd.DataFrame, *, indicators: dict) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)
    for column, value in indicators.items():
        mask &= df[column] >= value
    return ~mask.reset_index(drop=True)
