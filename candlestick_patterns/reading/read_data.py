import time

import numpy as np
import pandas as pd

from aggregation import aggregate
from calibration import calibration
from trend import trend_calculation

pd.set_option("mode.copy_on_write", True)


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
    interval_minutes : int, optional, default 1:
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
    t = time.perf_counter()

    df = pd.read_parquet(f"../data/{filename}.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")  # set datetime as index for mplfinance and filtering
    unique_dates = sorted(set(df.index.date))
    time_idx = pd.concat(
        [
            pd.date_range(
                start=f"{date} 09:30:00", end=f"{date} 16:00:00", freq="min"
            ).to_series()
            for date in unique_dates
        ]
    )
    df = df.reindex(time_idx)
    df = df.interpolate(method="linear")
    df = df.round({"open": 3, "high": 3, "low": 3, "close": 3, "volume": 0})

    df = aggregate.aggregate(df, interval_minutes)

    df["gap"] = df.index.astype(np.int64) // 10**9
    df["gap"] = (df["gap"] - df["gap"].shift(1)) // 60 > interval_minutes

    reference_set = df[df.index < "2007-01-01"]
    main_set = df[df.index >= "2007-01-01"]

    percentiles = calibration.calculate_percentiles(reference_set.to_numpy())

    main_set["5_MA"] = main_set["close"].rolling(5).mean()
    # 7 consecutive increases/decreases in moving average for trend to be defined
    main_set["trend"] = (
        main_set["5_MA"]
        .rolling(7)
        .apply(
            trend_calculation.monotonic,
            raw=True,
            engine="numba",
        )
    )
    del main_set["5_MA"]

    print(
        f"Reading and handling data done in {round(time.perf_counter()-t,2):<3.2f}s",
        end="\n\n",
    )
    return main_set, percentiles
