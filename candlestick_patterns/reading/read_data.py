import time

import pandas as pd

from aggregation import aggregate
from calibration import calibration
from trend import trend_calculation


def read_and_preprocess(
    filename: str, interval_minutes: int = 1
) -> pd.DataFrame | tuple:
    """
    Read the data from disk and perform some basic operations on it.

    Reads data in `.parquet` format and sets a datetimeindex, which is useful for time
    filtering or making plots with mplfinance. Next, if the parameter for aggregation is
    passed, aggregates the data. Finally, calculates moving averages and trend.

    Parameters
    ----------
    filename : str
        Filename of the data to read on disk.
    agg_interval : int, optional, default 1:
        Number of minutes over which the data will be aggregated. The default, 1,
        performs no aggregation.

    Returns
    -------
    pd.DataFrame
        A dataframe with datetime index, with an added "trend" column.
    tuple
        A tuple with the 10th/30th/70th percentiles of the real body, upper and lower
        shadows, respectively.
    """

    print("Reading and handling data", end="\r")
    t = time.time()
    df = pd.read_parquet(f"../data/{filename}.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")  # set datetime as index for mplfinance and filtering
    print(f"Reading and handling data done in {round(time.time()-t,2):<3.2f}s")

    df = aggregate.aggregate(df, agg_interval)

    print("Calibrating percentiles", end="\r")
    t = time.time()
    percentiles = calibration.calculate_percentiles(df.to_numpy())
    print(f"Calibrating percentiles done in {round(time.time()-t,2):<3.2f}s")

    print("Calculating moving average", end="\r")
    t = time.time()
    df["5_MA"] = df["close"].rolling(5).mean()
    print(f"Calculating moving average done in {round(time.time()-t,2):<3.2f}s")

    print("Calculating trend", end="\r")
    t = time.time()
    # 7 consecutive increases/decreases in moving average for trend to be defined
    df["trend"] = (
        df["5_MA"]
        .rolling(7)
        .apply(
            trend_calculation.monotonic,
            raw=True,
            engine="numba",
        )
    )
    del df["5_MA"]
    print(f"Calculating trend done in {round(time.time()-t,2):<3.2f}s", end="\n\n")
    return df, percentiles
