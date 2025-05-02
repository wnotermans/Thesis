import time

import pandas as pd

from shared import constants, shared_functions


def true_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the true range.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.

    Returns
    -------
    pd.Series
        The true range.
    """
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
    """
    Calculates the average true range (ATR).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    indicator_kwargs : dict
        ``window``: controls the window size.

    Returns
    -------
    pd.Series
        The ATR indicator.
    """
    tr = true_range(df)
    return tr.ewm(alpha=1 / indicator_kwargs["window"], adjust=False).mean()


def average_directional_movement_index(
    df: pd.DataFrame, *, indicator_kwargs: dict
) -> pd.Series:
    """
    Calculates the average directional movement index indicator (ADX). Reuses the ATR to
    speed up calculations.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with OHLC data and a column containing the ATR.
    indicator_kwargs : dict
        ``window``: sets the window over which the rolling averages are
        calculated.

    Returns
    -------
    pd.Series
        The ADX indicator.
    """
    window = indicator_kwargs["window"]
    up_move = df["high"] - df["high"].shift()
    down_move = df["low"].shift() - df["low"]
    atr = df["ATR"]
    plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0)
    plus_di = 100 * plus_dm.rolling(window).mean() / atr
    minus_di = 100 * minus_dm.rolling(window).mean() / atr
    adx = (plus_di - minus_di).abs() / (plus_di + minus_di)
    return 100 * adx.rolling(window).mean()


def Bollinger_bands(df: pd.DataFrame, *, indicator_kwargs: dict) -> tuple[pd.Series]:
    """
    Calculates Bollinger bands. Uses 2 std. dev. moves.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    indicator_kwargs : dict
        ``window``: controls the rolling window size.

    Returns
    -------
    tuple[pd.Series]
        Returns lower band, moving average and upper band.
    """
    window = indicator_kwargs["window"]
    ma = df["close"].rolling(window).mean()
    sd = df["close"].rolling(window).std()
    return ma - 2 * sd, ma, ma + 2 * sd


def detrended_price_oscillator(
    df: pd.DataFrame, *, indicator_kwargs: dict
) -> pd.Series:
    """
    Calculates the detrended price oscillator (DPO).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    indicator_kwargs : dict
        ``window``: controls the window size.

    Returns
    -------
    pd.Series
        The DPO indicator.
    """
    window = indicator_kwargs["window"]
    return df["close"].shift(window // 2 + 1) - df["close"].rolling(window).mean()


def moving_average(df: pd.DataFrame, *, indicator_kwargs: dict) -> pd.Series:
    """
    Calculates the moving average (MA).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    indicator_kwargs : dict
        ``window``: controls the window size.

    Returns
    -------
    pd.Series
        The MA.
    """
    return df["close"].rolling(indicator_kwargs["window"]).mean()


def moving_average_convergence_divergence(
    df: pd.DataFrame, *, indicator_kwargs: dict
) -> tuple[pd.Series]:
    """
    Calculates the moving average convergence/divergence (MACD).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    indicator_kwargs : dict
        ``spans``: 3-tuple controlling the spans.

    Returns
    -------
    tuple[pd.Series]
        The MACD indicator and MACD signal.
    """
    span_1, span_2, span_3 = indicator_kwargs["spans"]
    EMA_short = df["close"].ewm(span=span_1, adjust=False).mean()
    EMA_long = df["close"].ewm(span=span_2, adjust=False).mean()
    MACD = EMA_short - EMA_long
    return MACD, MACD.ewm(span=span_3, adjust=False).mean()


def money_flow_index(df: pd.DataFrame, *, indicator_kwargs: dict) -> pd.Series:
    """
    Calculates the money flow index (MFI).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    indicator_kwargs : dict
        ``window``: controls the window size.

    Returns
    -------
    pd.Series
        The MFI indicator.
    """
    window = indicator_kwargs["window"]
    money_flow = (df["high"] + df["low"] + df["close"]) / 3 * df["volume"]
    money_flow_shift = money_flow.shift()
    pos_money_flow = (money_flow > money_flow_shift).rolling(window).sum()
    neg_money_flow = (money_flow < money_flow_shift).rolling(window).sum()
    return 100 * pos_money_flow / (pos_money_flow + neg_money_flow)


def momentum(df: pd.DataFrame, *, indicator_kwargs: dict) -> pd.Series:
    """
    Calculates the momentum.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    indicator_kwargs : dict
        ``span``: controls the span.

    Returns
    -------
    pd.Series
        The momentum.
    """
    return 100 * df["close"] / df["close"].shift(indicator_kwargs["span"])


def relative_strength_index(df: pd.DataFrame, *, indicator_kwargs: dict) -> pd.Series:
    """
    Calculates the relative strength index (RSI).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    indicator_kwargs : dict
        ``window``: controls the window size.

    Returns
    -------
    pd.Series
        The RSI indicator.
    """
    window = indicator_kwargs["window"]
    diff = df["close"].diff()
    up_move = diff.where(diff > 0, 0)
    down_move = -diff.where(diff < 0, 0)
    up_SMA = up_move.rolling(window).mean()
    down_SMA = down_move.rolling(window).mean()
    return 100 - (100 / (1 + up_SMA / down_SMA))


def triple_exponential(df: pd.DataFrame, *, indicator_kwargs: dict) -> pd.Series:
    """
    Calculates the triple exponential indicator (TRIX).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    indicator_kwargs : dict
        ``windows``: 3-tuple of windows.

    Returns
    -------
    pd.Series
        The TRIX indicator.
    """
    window_1, window_2, window_3 = indicator_kwargs["windows"]
    EMA_1 = df["close"].ewm(alpha=1 / window_1, adjust=False).mean()
    EMA_2 = EMA_1.ewm(alpha=1 / window_2, adjust=False).mean()
    EMA_3 = EMA_2.ewm(alpha=1 / window_3, adjust=False).mean()
    return 100 * EMA_3.diff()


def vortex(df: pd.DataFrame, *, indicator_kwargs: dict) -> tuple[pd.Series]:
    """
    Calculates the vortex indicator (VI).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    indicator_kwargs : dict
        ``window``: controls the window size.

    Returns
    -------
    tuple[pd.Series]
        The VI+, VI- indicators and the difference of those two.
    """
    window = indicator_kwargs["window"]
    tr_sum = true_range(df).rolling(window).sum()
    VM_plus = (df["high"] - df["low"].shift()).abs()
    VM_minus = (df["low"] - df["high"].shift()).abs()
    VI_plus = VM_plus.rolling(window).sum() / tr_sum
    VI_minus = VM_minus.rolling(window).sum() / tr_sum
    return VI_plus, VI_minus, VI_plus - VI_minus


def volume_weight(df: pd.DataFrame, *, indicator_kwargs: dict) -> pd.Series:
    """
    Calculates the volume weight.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC and volume data.
    indicator_kwargs : dict
        ``minutes``: time span of the blocks in minutes.

    Returns
    -------
    pd.Series
        The volume weight.
    """
    minutes = indicator_kwargs["minutes"]
    shift = 24 * 60 / minutes
    if shift != int(shift):
        raise ValueError(
            f"The given amount of minutes ({minutes}) "
            "does not divide the amount of minutes in a day (1440)."
        )
    volume_rolling_mean = (
        df["volume"].groupby(df.index.date).sum().rolling(7).mean()
    ).shift()
    df_volume_mean = df.index.normalize().map(volume_rolling_mean)
    block_volume = (
        df["volume"].groupby(pd.Grouper(freq=f"{minutes}min")).sum().shift(int(shift))
    )
    block_volume = block_volume.reindex(df.index).ffill()
    return block_volume / df_volume_mean


def Williams_R(df: pd.DataFrame, *, indicator_kwargs: dict) -> pd.Series:
    """
    Calculates Williams' %R (%R).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    indicator_kwargs : dict
        ``window``: controls the window size.

    Returns
    -------
    pd.Series
        The %R indicator.
    """
    window = indicator_kwargs["window"]
    rolling_high = df["high"].rolling(window).max()
    rolling_low = df["low"].rolling(window).min()
    return -100 * (rolling_high - df["close"]) / (rolling_high - rolling_low)


def parabolic_SAR(df: pd.DataFrame, *, indicator_kwargs: dict) -> list:
    """
    Calculates the parabolic stop and reverse indicator (PSAR).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    indicator_kwargs : dict
        ``step``: how much to increment the acceleration factor by.

    Returns
    -------
    list
        The PSAR indicator.
    """
    step = indicator_kwargs["step"]
    max_accel_factor = indicator_kwargs["max_accel_factor"]

    length = len(df)
    high = list(df["high"])
    low = list(df["low"])
    psar = list(df["close"])  # faster than casting/copying to numpy
    up_trend = True
    accel_factor = step
    max_high = high[0]
    min_low = low[0]

    for i in range(2, length):
        if up_trend:
            psar[i] = psar[i - 1] + accel_factor * (max_high - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + accel_factor * (min_low - psar[i - 1])
        reverse = False

        if up_trend:
            if low[i] < psar[i]:
                up_trend = False
                reverse = True
                psar[i] = max_high
                min_low = low[i]
                accel_factor = step

        elif high[i] > psar[i]:
            up_trend = True
            reverse = True
            psar[i] = min_low
            max_high = high[i]
            accel_factor = step

        if not reverse:
            if up_trend:
                if high[i] > max_high:
                    max_high = high[i]
                    accel_factor = min(accel_factor + step, max_accel_factor)
                psar[i] = min(psar[i], low[i - 1])
                psar[i] = min(psar[i], low[i - 2])

            else:
                if low[i] < min_low:
                    min_low = low[i]
                    accel_factor = min(accel_factor + step, max_accel_factor)
                psar[i] = max(psar[i], high[i - 1])
                psar[i] = max(psar[i], high[i - 2])

    return psar


INDICATORS = {
    "ATR": average_true_range,
    "ADX": average_directional_movement_index,
    "BB": Bollinger_bands,
    "DPO": detrended_price_oscillator,
    "MA": moving_average,
    "MACD": moving_average_convergence_divergence,
    "MFI": money_flow_index,
    "momentum": momentum,
    "PSAR": parabolic_SAR,
    "RSI": relative_strength_index,
    "TRIX": triple_exponential,
    "VI": vortex,
    "VW": volume_weight,
    "%R": Williams_R,
}


def calculate_indicators(
    df: pd.DataFrame, *, indicator_kwargs: dict[dict]
) -> pd.DataFrame:
    """
    Calculate additional indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with OHLC data.
    indicator_kwargs : dict[dict]
        Additional parameters for the indicators.

    Returns
    -------
    pd.DataFrame
        Original df together with additional indicators as new columns.
    """
    t = time.perf_counter()
    indicator_kwargs = shared_functions.set_kwarg_defaults(
        indicator_kwargs,
        local_dict=INDICATORS,
        default_dict=constants.INDICATOR_DEFAULTS,
    )
    for indicator_name, indicator_function in INDICATORS.items():
        kwargs = indicator_kwargs[indicator_name]
        if indicator_name == "BB":
            df["BB_low"], df["BB_mid"], df["BB_high"] = indicator_function(
                df, indicator_kwargs=kwargs
            )
        elif indicator_name == "MACD":
            df["MACD"], df["MACD_signal"] = indicator_function(
                df, indicator_kwargs=kwargs
            )
        elif indicator_name == "VI":
            df["VI+"], df["VI-"], df["VI_diff"] = indicator_function(
                df, indicator_kwargs=kwargs
            )
        else:
            df[indicator_name] = indicator_function(df, indicator_kwargs=kwargs)
    print(
        f"Calculating additional indicators done in {time.perf_counter() - t:3.2f}s",
        end="\n\n",
    )
    return df
