import time

import pandas as pd

from aggregation import aggregate
from calibration import calibration
from trend import trend_calculation


def run(filename: str, agg_interval=1) -> pd.DataFrame:
    """Function that reads data and sets a datetimeindex. Next, if the parameter for
    aggregation is passed, aggregates the data. Finally, calculates moving averages and
    trend.

    Inputs
    ------
    filename: filename of the data to read on disk.
    agg_interval, default 1: interval to which the data will be aggregated. The
    default, 1, performs no aggregation.

    Outputs
    -------
    A dataframe with datetime index, with an added "trend" column.
    """

    print("Reading and handling data", end="\r")
    t = time.time()
    df = pd.read_parquet(f"../data/{filename}.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")  # set datetime as index for mplfinance
    print(f"Reading and handling data done in {round(time.time()-t,2):<3.2f}s")

    df = aggregate.aggregate(df, agg_interval)

    print("Calibrating percentiles", end="\r")
    t = time.time()
    percentiles = calibration.percentiles(df.to_numpy())
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
            trend_calculation.trend,
            raw=True,
            engine="numba",
        )
    )
    del df["5_MA"]
    print(f"Calculating trend done in {round(time.time()-t,2):<3.2f}s", end="\n\n")
    return df
