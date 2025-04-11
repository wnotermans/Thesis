import time

import pandas as pd

INDICATORS = ["ADX", "ATR", "DPO", "MA", "MACD", "momentum", "RSI", "TRIX", "VI"]


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
    df["ATR"] = average_true_range(df, indicator_kwargs=indicator_kwargs["ATR"])
    df["ADX"] = average_directional_movement_index(
        df, df["ATR"], indicator_kwargs=indicator_kwargs["ADX"]
    )
    df["DPO"] = detrended_price_oscillator(df, indicator_kwargs=indicator_kwargs["DPO"])
    df["MA"] = moving_average(df, indicator_kwargs=indicator_kwargs["MA"])
    df["MACD"], df["MACD_signal"] = moving_average_convergence_divergence(
        df, indicator_kwargs=indicator_kwargs["MACD"]
    )
    df["momentum"] = momentum(df, indicator_kwargs=indicator_kwargs["momentum"])
    df["RSI"] = relative_strength_index(df, indicator_kwargs=indicator_kwargs["RSI"])
    df["TRIX"] = triple_exponential(df, indicator_kwargs=indicator_kwargs["TRIX"])
    df["VI+"], df["VI-"], df["VI_diff"] = vortex(
        df, indicator_kwargs=indicator_kwargs["VI"]
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
    for indicator in INDICATORS:
        indicator_kwargs.setdefault(indicator, {})
    indicator_kwargs["ADX"].setdefault("window", 15)
    indicator_kwargs["ATR"].setdefault("window", 15)
    indicator_kwargs["DPO"].setdefault("window", 20)
    indicator_kwargs["MA"].setdefault("window", 7)
    indicator_kwargs["MACD"].setdefault("spans", (5, 11, 5))
    indicator_kwargs["momentum"].setdefault("span", 7)
    indicator_kwargs["RSI"].setdefault("window", 15)
    indicator_kwargs["TRIX"].setdefault("windows", (15, 15, 15))
    indicator_kwargs["VI"].setdefault("window", 21)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift()
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def average_true_range(df: pd.DataFrame, *, indicator_kwargs: dict) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / indicator_kwargs["window"], adjust=False).mean()


def average_directional_movement_index(
    df: pd.DataFrame, ATR: pd.Series, *, indicator_kwargs: dict
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
    window = indicator_kwargs["window"]
    up_move = df["high"] - df["high"].shift()
    down_move = df["low"].shift() - df["low"]
    plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0)
    plus_di = 100 * plus_dm.rolling(window).mean() / ATR
    minus_di = 100 * minus_dm.rolling(window).mean() / ATR
    adx = (plus_di - minus_di).abs() / (plus_di + minus_di)
    return 100 * adx.rolling(window).mean()


def detrended_price_oscillator(
    df: pd.DataFrame, *, indicator_kwargs: dict
) -> pd.Series:
    window = indicator_kwargs["window"]
    return df["close"].shift(window // 2 + 1) - df["close"].rolling(window).mean()


def moving_average(df: pd.DataFrame, *, indicator_kwargs: dict) -> pd.Series:
    return df["close"].rolling(indicator_kwargs["window"]).mean()


def moving_average_convergence_divergence(
    df: pd.DataFrame, *, indicator_kwargs: dict
) -> tuple[pd.Series]:
    span_1, span_2, span_3 = indicator_kwargs["spans"]
    EMA_short = df["close"].ewm(span=span_1, adjust=False).mean()
    EMA_long = df["close"].ewm(span=span_2, adjust=False).mean()
    MACD = EMA_short - EMA_long
    return MACD, MACD.ewm(span=span_3, adjust=False).mean()


def momentum(df: pd.DataFrame, *, indicator_kwargs: dict) -> pd.Series:
    return 100 * df["close"] / df["close"].shift(indicator_kwargs["span"])


def relative_strength_index(df: pd.DataFrame, *, indicator_kwargs: dict) -> pd.Series:
    window = indicator_kwargs["window"]
    diff = df["close"].diff()
    up_move = diff.where(diff > 0, 0)
    down_move = -diff.where(diff < 0, 0)
    up_SMA = up_move.rolling(window).mean()
    down_SMA = down_move.rolling(window).mean()
    return 100 - (100 / (1 + up_SMA / down_SMA))


def triple_exponential(df: pd.DataFrame, *, indicator_kwargs: dict) -> pd.Series:
    window_1, window_2, window_3 = indicator_kwargs["windows"]
    EMA_1 = df["close"].ewm(alpha=1 / window_1, adjust=False).mean()
    EMA_2 = EMA_1.ewm(alpha=1 / window_2, adjust=False).mean()
    EMA_3 = EMA_2.ewm(alpha=1 / window_3, adjust=False).mean()
    return 100 * EMA_3.diff()


def vortex(df: pd.DataFrame, *, indicator_kwargs: dict) -> pd.Series:
    window = indicator_kwargs["window"]
    tr_sum = true_range(df).rolling(window).sum()
    VM_plus = (df["high"] - df["low"].shift()).abs()
    VM_minus = (df["low"] - df["high"].shift()).abs()
    VI_plus = VM_plus.rolling(window).sum() / tr_sum
    VI_minus = VM_minus.rolling(window).sum() / tr_sum
    return VI_plus, VI_minus, VI_plus - VI_minus
