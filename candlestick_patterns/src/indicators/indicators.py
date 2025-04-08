import time

import pandas as pd


def calculate_indicators(
    df: pd.DataFrame, *, indicator_kwargs: dict[dict]
) -> pd.DataFrame:
    """
    Calculate additional indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with OHLC and trend data.
    indicator_kwargs : dict[dict]
        Additional parameters for the indicators.

    Returns
    -------
    pd.DataFrame
        Original df together with additional indicators as new columns.
    """
    t = time.perf_counter()
    set_defaults(indicator_kwargs)
    df["ADX"] = average_directional_movement_index(
        df, indicator_kwargs=indicator_kwargs["ADX"]
    )
    print(
        f"Calculating additional indicators done in {time.perf_counter() - t:<3.2f}s",
        end="\n\n",
    )
    return df


def set_defaults(indicator_kwargs: dict[dict]) -> None:
    """
    Sets (in place) default kwargs for the indicators. Also creates keys for indicators,
    should they not exist already.

    Parameters
    ----------
    filter_dict : dict[dict]
        Already present kwargs. These are left unchanged.
    """
    indicator_kwargs.setdefault("ADX", {})
    indicator_kwargs["ADX"].setdefault("window", 15)


def average_directional_movement_index(
    df: pd.DataFrame, *, indicator_kwargs: dict
) -> pd.Series:
    """
    Calculates the average directional movement index indicator, also called ADX.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with OHLC and trend data.
    indicator_kwargs : dict
        kwarg `window`, that sets the window over which the rolling averages are
        calculated.

    Returns
    -------
    pd.Series
        The ADX indicator.
    """
    true_range = pd.DataFrame(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ]
    ).max()
    average_true_range = true_range.ewm(
        alpha=1 / indicator_kwargs["window"], adjust=False
    ).mean()
    up_move = df["high"] - df["high"].shift()
    down_move = df["low"].shift() - df["low"]
    plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0)
    plus_di = (
        100 * plus_dm.rolling(indicator_kwargs["window"]).mean() / average_true_range
    )
    minus_di = (
        100 * minus_dm.rolling(indicator_kwargs["window"]).mean() / average_true_range
    )
    adx = (plus_di - minus_di).abs() / (plus_di + minus_di)
    return 100 * adx.rolling(indicator_kwargs["window"]).mean()
