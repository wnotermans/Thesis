import pandas as pd


def aggregate(df: pd.DataFrame, interval: int = 5) -> pd.DataFrame:
    """Aggregate data in df over `interval` minutes, defaults to 5.

    Inputs: dataframe with columns `open`,`high`,`low`,`close`,`volume`; `interval`.

    Outputs: new dataframe with columns aggregated over `interval` minutes.
    """
    return df.resample(f"{interval}min", label="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
