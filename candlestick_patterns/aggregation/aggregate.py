import pandas as pd


def aggregate(df: pd.DataFrame, interval: int = 5) -> pd.DataFrame:
    """Aggregate data in df over `interval` minutes, defaults to 5.

    Inputs
    ------
    Dataframe with columns `open`,`high`,`low`,`close`,`volume`; `interval`.

    Outputs
    -------
    New dataframe with columns aggregated over `interval` minutes.
    """
    if interval == 1:
        return df
    return df.resample(f"{interval}min", label="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
