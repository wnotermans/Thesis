import os

import numpy as np
import pandas as pd


def read_and_clean_data(filename: str, *, impact_level: tuple) -> pd.DataFrame:
    """
    Read the data from disk and clean it up into a DataFrame.

    Parameters
    ----------
    filename : str
        Filename of the target file.
    impact_level : tuple
        Which impact levels to exclude.

    Returns
    -------
    pd.DataFrame
        DataFrame with clean data and filtered impact levels.
    """
    raw = pd.read_csv(f"data/news/{filename}")
    raw = raw.drop(columns=["Currency", "Id", "Name"])
    raw["Start"] = pd.to_datetime(raw["Start"])
    sorted_df = custom_sort(raw)
    sorted_df = sorted_df.drop_duplicates(["Start"])
    return sorted_df[sorted_df["Impact"].isin(impact_level)]


def custom_sort(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Custom sorting of the DataFrame.

    Sorting a column ascending or by a key is easy in pandas, but not multiple columns
    at the same time with different sorting methods. The approach is to map a column to
    the desired sorting, sort all columns as/descending, then mapping back.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame.
    """
    custom_sort = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
    raw["Impact"] = raw["Impact"].map(custom_sort)
    sorted_df = raw.sort_values(["Start", "Impact"])
    sorted_df["Impact"] = sorted_df["Impact"].map(
        {v: k for k, v in custom_sort.items()}
    )
    return sorted_df


def get_news_df(
    *, impact_level: tuple, minutes_after: int
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Get a DataFrame of economic news events.

    Automatically reads csv-data from the folder `data/news`. This data should have
    the column "Id", "Start", "Name", "Currency", and "Impact". Columns "Id", "Name" and
    "Currency" are unused. Removes duplicate timestamps, keeping the highest level of
    impact in that case.

    Parameters
    ----------
    impact_level : tuple
        Which impact levels to include.
    minutes_after : int
        How many minutes of NaNs to include after the news events (for filtering).

    Returns
    -------
    pd.DataFrame
        A DataFrame with a DateTimeIndex and the corresponding economic news events.
    """
    news_df_list = [
        read_and_clean_data(file, impact_level=impact_level)
        for file in os.listdir("data/news")
        if file.endswith(".csv")
    ]
    news_df = pd.concat(news_df_list, ignore_index=True)
    news_df = news_df.set_index("Start")
    timestamps = news_df.index.to_numpy()
    filter_index = pd.to_datetime(
        np.repeat(timestamps, minutes_after)
    ) + pd.to_timedelta(np.tile(np.arange(minutes_after), len(timestamps)), unit="min")
    filter_index = pd.DatetimeIndex(filter_index).drop_duplicates()
    return news_df, filter_index
