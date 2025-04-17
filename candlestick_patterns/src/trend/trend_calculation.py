import time

import numpy as np
import pandas as pd

USES_AVERAGING = {"monotonic", "counting"}


def calculate_trend(
    df: pd.DataFrame,
    *,
    averaging_method: str,
    averaging_method_kwargs: dict,
    decision_method: str,
    decision_method_kwargs: dict,
) -> pd.DataFrame:
    """
    Calculate short-term trend of data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with at least the column `"close"`.
    averaging_method : str
        Method used to calculate a short-term average.
        - `"MA"`: simple moving average
        - `"WMA"`: weighted moving average with linearly decreasing weights
        - `"EWMA"`: exponentially weighted moving average, with `alpha=2/(1+span)`
    averaging_method_kwargs : dict
        Additional kwargs used in the calculation of the average.
        - `consecutive`: window size for the calculation of the rolling average.
        - `span`: span of the (EW/W)MA calculation.
    decision_method : str
        Method that decides the trend.
        - `"monotonic"`: based on consecutive in/decreases of the short-term average.
        Additional kwarg `consecutive` can be passed to decide the number of needed
        in/decreases. `consecutive` is 7 by default.
        - `"counting"`: counts the amount of in/decreases of the short-term average.
        Additional kwarg `fraction` controls what fraction of the data needs to be
        in/decreasing. `fraction` is 0.7 by default.
        - `"high_low"`: based on simultaneous increases of the high and low.
        No additional kwargs.
    decision_method_kwargs : dict
        Additional kwargs used by the trend decision method.

    Returns
    -------
    pd.DataFrame
        Original dataframe with an additional column `"trend"`.

    Raises
    ------
    ValueError
        When an unknown decision method is passed.
    """
    valid_decision_methods = globals()
    if decision_method not in valid_decision_methods:
        raise ValueError("Incorrect decision method")
    decision_method_function_call = globals()[decision_method]

    t = time.perf_counter()

    if decision_method == "counting":
        decision_method_kwargs.setdefault("fraction", 0.7)

    if decision_method in USES_AVERAGING:
        averaging_method_kwargs.setdefault("span", 5)
        averaging_method_kwargs.setdefault("consecutive", 7)

        df["rolling_average"] = calculate_rolling_average(
            df["close"],
            averaging_method,
            averaging_method_kwargs=averaging_method_kwargs,
        )
        df["trend"] = (
            df["rolling_average"]
            .rolling(averaging_method_kwargs["consecutive"])
            .apply(
                decision_method_function_call, kwargs=decision_method_kwargs, raw=True
            )
        )
        del df["rolling_average"]

    if decision_method == "high_low":
        df["trend"] = decision_method_function_call(
            df["high"].to_numpy(), df["low"].to_numpy()
        )

    print(f"Calculating trend done in {time.perf_counter() - t:>3.2f}s", end="\n\n")
    return df


def calculate_rolling_average(
    ser: pd.Series, averaging_method: str, *, averaging_method_kwargs: dict
) -> pd.Series:
    """
    Calculate different rolling averages.

    Parameters
    ----------
    ser : pd.Series
        Data to be averaged.
    averaging_method : str
        - `"MA"`: simple moving average
        - `"WMA"`: weighted moving average with linearly decreasing weights
        - `"EWMA"`: exponentially weighted moving average, with `alpha=2/(1+span)`
    averaging_method_kwargs : dict
        kwargs for rolling average.
        - `span`: span of the calculation.

    Returns
    -------
    pd.Series
        The averaged data.

    Raises
    ------
    ValueError
        If an unknown averaging method is given.
    """
    if averaging_method not in {"MA", "WMA", "EWMA"}:
        raise ValueError(
            f"averaging method '{averaging_method}' not recognized, "
            "must be 'MA', 'WMA' or 'EWMA' "
        )
    span = averaging_method_kwargs["span"]
    if averaging_method == "MA":
        return ser.rolling(span).mean()
    if averaging_method == "WMA":
        weights = np.arange(span, 0, -1)
        sum_weights = np.sum(weights)
        return ser.rolling(span).apply(
            lambda x: np.sum(weights * x) / sum_weights, raw=True
        )
    if averaging_method == "EWMA":
        return ser.ewm(span=span).mean()
    return None


def monotonic(C: list) -> int:
    """
    Trend calculation based on consecutive strict in/decreases.

    Parameters
    ----------
    C : list
        List of closes.

    Returns
    -------
    int
        1 if the list is strictly increasing, -1 if strictly decreasing, 0 otherwise.
    """
    increasing, decreasing = False, False

    for i in range(len(C) - 1):
        if C[i + 1] > C[i]:
            increasing = True
        elif C[i + 1] < C[i]:
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


def counting(C: list, *, fraction: float) -> int:
    """
    Trend calculation based on counting in/decreases. `fraction` controls how many
    in/decreases there need to be.

    Parameters
    ----------
    C : list
        List of closes.
    fraction : float, optional, default 0.7
        What fraction of the data needs to be in/decreasing.

    Returns
    -------
    int
        1 if `fraction` of the list is increasing, -1 if decreasing, 0 otherwise.
    """
    count_increase, count_decrease = 0, 0

    for i in range(len(C) - 1):
        if C[i + 1] > C[i]:
            count_increase += 1
        elif C[i + 1] < C[i]:
            count_decrease += 1

    if count_increase >= fraction * len(C):
        return 1
    if count_decrease >= fraction * len(C):
        return -1
    return 0


def high_low(H: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Trend calculation based on simultaneous in/decreases of the high and low.

    Parameters
    ----------
    H : np.ndarray
        Highs.
    L : np.ndarray
        Lows.

    Returns
    -------
    np.ndarray
        Trend array, with -1 for simultaneous decrease, 1 for simultaneous increase,
        0 otherwise.
    """
    H_diff_sign = np.sign(np.diff(H))
    L_diff_sign = np.sign(np.diff(L))
    return np.concat(
        [
            [0],
            np.where((H_diff_sign == L_diff_sign) & (H_diff_sign != 0), H_diff_sign, 0),
        ]
    )


def parabolic_SAR(df: pd.DataFrame, *, decision_kwargs: dict) -> list[int]:
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
