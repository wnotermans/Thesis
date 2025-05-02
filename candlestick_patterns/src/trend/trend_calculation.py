import time

import numpy as np
import pandas as pd

from shared import constants, shared_functions


def moving_average(ser: pd.Series, *, averaging_kwargs: dict) -> pd.Series:
    return ser.rolling(averaging_kwargs["window"]).mean()


def weighted_moving_average(ser: pd.Series, *, averaging_kwargs: dict) -> pd.Series:
    window = averaging_kwargs["window"]
    weights = np.arange(window, 0, -1)
    sum_weights = np.sum(weights)
    return ser.rolling(window).apply(
        lambda x: np.sum(weights * x) / sum_weights, raw=True
    )


def exponential_moving_average(ser: pd.Series, *, averaging_kwargs: dict) -> pd.Series:
    return ser.rolling(averaging_kwargs["window"]).mean()


def monotonic(df: pd.DataFrame, *, decision_kwargs: dict) -> pd.Series:
    """
    Trend calculation based on consecutive strict in/decreases.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with rolling average.
    decision_kwargs : dict
        ``span``: span over which the monotonicity is checked.

    Returns
    -------
    pd.Series
        1 if is strictly increasing, -1 if strictly decreasing, 0 otherwise.
    """
    span = decision_kwargs["span"]

    def check_monotonicity(lst: list) -> int:
        increasing, decreasing = False, False

        for i in range(len(lst) - 1):
            if lst[i + 1] > lst[i]:
                increasing = True
            elif lst[i + 1] < lst[i]:
                decreasing = True
            else:
                return 0

            if increasing and decreasing:
                return 0

        if increasing:
            return 1
        if decreasing:
            return -1
        return 0

    return df["rolling_average"].rolling(span).apply(check_monotonicity, raw=True)


def counting(df: pd.DataFrame, *, decision_kwargs: dict) -> pd.Series:
    """
    Trend calculation based on counting in/decreases.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with rolling average.
    decision_kwargs : dict
        ``span``: span over which in/decreases are counted.
        ``fraction``: controls which fraction of ``span`` has to be in/decreases for it
        to count.

    Returns
    -------
    pd.Series
        1 if ``fraction`` of the ``span`` is increasing, -1 if decreasing, 0 otherwise.
    """
    span = decision_kwargs["span"]
    fraction = decision_kwargs["fraction"]
    threshold = span * fraction
    signs = pd.Series(np.sign(np.diff(df["rolling_average"], prepend=np.nan)))
    pos_count = signs.rolling(span).apply(lambda x: np.count_nonzero(x > 0), raw=True)
    neg_count = signs.rolling(span).apply(lambda x: np.count_nonzero(x < 0), raw=True)
    return np.where(pos_count >= threshold, 1, np.where(neg_count >= threshold, -1, 0))


def high_low(df: pd.DataFrame, *, decision_kwargs: dict) -> np.ndarray:
    """
    Trend calculation based on simultaneous in/decreases of the high and low.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    decision_kwargs : dict
        Unused, only present for compatibility reasons.

    Returns
    -------
    np.ndarray
        Trend array, with -1 for simultaneous decrease, 1 for simultaneous increase,
        0 otherwise.
    """
    H_diff_sign = np.sign(np.diff(df["high"]))
    L_diff_sign = np.sign(np.diff(df["low"]))
    return np.concat(
        [
            [0],
            np.where((H_diff_sign == L_diff_sign) & (H_diff_sign != 0), H_diff_sign, 0),
        ]
    )


def PSAR_trend(df: pd.DataFrame, *, decision_kwargs: dict) -> list[int]:
    """
    Trend calculation based on the parabolic SAR.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    decision_kwargs : dict
        ``step``: how much the acceleration factor increases by per step.
        ``max_accel_factor``: maximum possible acceleration factor.

    Returns
    -------
    list[int]
        List with 1 for uptrend, -1 for downtrend.
    """
    step = decision_kwargs["step"]
    max_accel_factor = decision_kwargs["max_accel_factor"]

    length = len(df)
    high = list(df["high"])
    low = list(df["low"])
    psar = list(df["close"])
    psar_trend = [None] * length
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

        psar_trend[i] = 1 if up_trend else -1
    return psar_trend


AVERAGING_METHODS = {
    "SMA": moving_average,
    "WMA": weighted_moving_average,
    "EMA": exponential_moving_average,
}
USES_AVERAGING = {"monotonic", "counting"}
DECISION_METHODS = {
    "monotonic": monotonic,
    "counting": counting,
    "high_low": high_low,
    "PSAR": PSAR_trend,
}


def calculate_trend(
    df: pd.DataFrame,
    *,
    averaging_method: str,
    averaging_kwargs: dict,
    decision_method: str,
    decision_kwargs: dict,
) -> pd.DataFrame:
    """
    Calculates the trend according to the specified method(s).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data.
    averaging_method : str
        - "SMA": simple moving average,
        - "WMA": weighted moving average,
        - "EMA": exponential moving average.
    averaging_kwargs : dict
        Kwargs for the averaging method.
    decision_method : str
        - "monotonic": monotonic in/decreases,
        - "counting": count in/decreases (not necessarily monotonic),
        - "high_low": simultaneous in/decrease of high and low,
        - "PSAR": parabolic SAR.
    decision_kwargs : dict
        Kwargs for the decision method.

    Returns
    -------
    pd.DataFrame
        Original df along with a "trend" column.
    """
    t = time.perf_counter()

    averaging_kwargs = shared_functions.set_kwarg_defaults(
        averaging_kwargs,
        local_dict=AVERAGING_METHODS,
        default_dict=constants.TREND_AVERAGING_DEFAULTS,
    )
    decision_kwargs = shared_functions.set_kwarg_defaults(
        decision_kwargs,
        local_dict=DECISION_METHODS,
        default_dict=constants.TREND_DECISION_DEFAULTS,
    )

    averaging_func = AVERAGING_METHODS[averaging_method]
    decision_func = DECISION_METHODS[decision_method]

    if decision_method in USES_AVERAGING:
        df["rolling_average"] = averaging_func(
            df["close"], averaging_kwargs=averaging_kwargs[averaging_method]
        )

    df["trend"] = decision_func(df, decision_kwargs=decision_kwargs[decision_method])

    if "rolling_average" in df.columns:
        del df["rolling_average"]

    print(f"Calculating trend done in {time.perf_counter() - t:3.2f}s", end="\n\n")

    return df
