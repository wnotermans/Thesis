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

    Read data in `.parquet` format and sets a DateTimeIndex, which is useful for time
    filtering or making plots with mplfinance. Filters data to US market time
    (09:30-16:00). Next, if the parameter for aggregation is passed, aggregates the
    data. Calculates gaps in the data and splits it into a reference and main set.
    Calculates percentiles of body and shadow length on the reference set.
    Finally, calculates moving averages and trend.

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
        The main dataset (starting 01/01/2007) with datetime index, with added "trend"
        and "gap" columns.
    tuple
        A tuple with the 10th/30th/70th percentiles of the real body (split between
        black and white if it fails the Kolmogorov-Smirnov test);
        10th/30th/70th/90th percentiles for upper and lower shadow length, respectively.
    """
    print(
        f"Reading and handling dataset {filename}"
        + f"with {interval_minutes} minute aggregation"
    )
    t = time.perf_counter()

    df = pd.read_parquet(f"data/{filename}.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    unique_dates = sorted(set(df.index.date))
    time_idx = pd.concat(
        [
            pd.date_range(
                start=f"{date} 09:30:00", end=f"{date} 16:00:00", freq="min"
            ).to_series()
            for date in unique_dates
        ]
    )

    print(calculate_missing(df, time_idx))

    df = (
        df.reindex(time_idx)  # filters to US market time and adds NaNs for missing data
        .interpolate(method="linear")
        .round({"open": 3, "high": 3, "low": 3, "close": 3, "volume": 0})
        .bfill()
        .ffill()
    )

    df = aggregate.aggregate(df, interval_minutes)

    df["gap"] = df.index.astype(np.int64) // 10**9
    df["gap"] = (df["gap"] - df["gap"].shift(1)) // 60 > interval_minutes

    reference_set, main_set = split_data(df, unique_dates)

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
        f"Reading and handling dataset {filename} done in {
            round(time.perf_counter() - t, 2):<3.2f
        }s",
        end="\n\n",
    )
    return main_set, percentiles


def calculate_missing(df: pd.DataFrame, time_idx: pd.DatetimeIndex) -> str:
    """
    Calculates the amount of missing data in the dataset.

    Data is considered missing if the gap to the previous data point is more than
    10 minutes. Periods where the market is closed are not considered as missing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the original data, before any filtering or aggregation.
    time_idx : pd.DatetimeIndex
        DatetimeIndex used for filtering the data.

    Returns
    -------
    str
        Percentage of missing data.
    """
    time_filtered_df = df.reindex(time_idx).dropna()
    time_filtered_df["unix_time"] = time_filtered_df.index.astype(np.int64) // 10**9
    minute_gaps = np.array(
        (time_filtered_df["unix_time"] - time_filtered_df["unix_time"].shift(1)) // 60
    )
    allowed_gaps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1050, 1051, 2490, 3930, 5370]
    missing_data_points = np.logical_not(np.isin(minute_gaps, allowed_gaps)).sum()
    return f"Missing data: {missing_data_points / len(df):.02%}"


def split_data(
    df: pd.DataFrame, unique_dates: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into a reference and main set.

    The reference set is taken to be the data before January 1st, 2007. If the data
    starts after this date or less than 5 years of data is available before
    January 1st, 2007, it will take the first 5 years of the data as a reference set.
    If the data covers less than 15 years, it is split in a 1:2 ratio.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data and a DateTimeIndex.
    unique_dates : list
        List of unique dates in the DateTimeIndex of the DataFrame.

    Returns
    -------
    pd.DataFrame
        Reference set and main in the same format as the input. See summary for how it
        is split up.
    """
    first_day = pd.Timestamp(unique_dates[0])
    last_day = pd.Timestamp(unique_dates[-1])
    print(f"Start date: {first_day}")
    print(f"End date: {last_day}")
    date_diff = last_day - first_day
    num_years = (date_diff.days + date_diff.seconds / 86400) / 365.25
    if num_years <= 15:
        split_date = first_day + date_diff / 3
    else:
        split_date = max(first_day + pd.DateOffset(years=5), pd.Timestamp(2007, 1, 1))
    print(f"Data split date: {split_date}")
    reference_set = df[df.index < split_date]
    main_set = df[df.index >= split_date]
    return reference_set, main_set
