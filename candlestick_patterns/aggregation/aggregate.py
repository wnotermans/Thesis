import pandas as pd


def aggregate(df: pd.DataFrame, interval_minutes: int = 5) -> pd.DataFrame:
    """
    Aggregate data in the DataFrame over the specified interval in minutes.

    The function aggregates the data in the given DataFrame (`df`) over a specified
    time interval (in minutes). The default interval is 5 minutes. The aggregation
    is applied to the `open`, `high`, `low`, `close`, and `volume` columns.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the columns `open`, `high`, `low`, `close`, and `volume`.
    interval_minutes : int, optional, default 5
        The time interval (in minutes) over which the data should be aggregated.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the data aggregated over the specified interval.
    """
    if interval_minutes == 1:
        return df
    return df.resample(f"{interval_minutes}min", label="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
