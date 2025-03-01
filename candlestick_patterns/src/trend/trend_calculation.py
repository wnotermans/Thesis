import time

import numba
import numpy as np
import pandas as pd


def calculate_rolling_average(
    ser: pd.Series, averaging_method: str = "MA", span: int = 5
) -> pd.Series:
    """
    Calculate different rolling averages.

    Parameters
    ----------
    ser : pd.Series
        Data to be averaged.
    averaging_method : str, optional, default `"MA"`
        - `"MA"`: simple moving average
        - `"WMA"`: weighted moving average with linearly decreasing weights
        - `"EWMA"`: exponentially weighted moving average, with `alpha=2/(1+span)`
    span : int, optional, default 5
        How many data points are included in (weighted) moving average, also governs the
        alpha used in the exponentially weighted moving average.

    Returns
    -------
    pd.Series
        The averaged data.

    Raises
    ------
    ValueError
        If an unknown averaging method is given.
    """
    if averaging_method == "MA":
        return ser.rolling(span).mean()
    if averaging_method == "WMA":
        weights = np.arange(span, 0, -1)
        sum_weights = sum(weights)
        return ser.rolling(span).apply(
            lambda x: np.sum(weights * x) / sum_weights, raw=True
        )
    if averaging_method == "EWMA":
        return ser.ewm(span=span).mean()


def calculate_trend(
    df: pd.DataFrame,
    averaging_method: str = "MA",
    span: int = 5,
    decision_method: str = "monotonic",
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate short-term trend of data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with at least the column `"close"`.
    averaging_method : str, optional, default "MA"
        Method used to calculate a short-term average.
        - `"MA"`: simple moving average
        - `"WMA"`: weighted moving average with linearly decreasing weights
        - `"EWMA"`: exponentially weighted moving average, with `alpha=2/(1+span)`
    span : int, optional, default 5
        Span over which the short-term average is calculated.
    decision_method : str, optional, default `"monotonic"`
        Method that decides the trend.
        - `"monotonic"`: based on consecutive in/decreases of the short-term average.
        Additional kwarg `consecutive` can be passed to decide the number of needed
        in/decreases.
    **kwargs : dict, optional
        Additional arguments to be passed into the decision method.

    Returns
    -------
    pd.DataFrame
        Original dataframe with an additional column `"trend"`.

    Raises
    ------
    ValueError
        When an unknown decision method is passed.
    """
    t = time.perf_counter()

    consecutive = kwargs.get("consecutive", 7)

    print(
        f"Calculating trend: {averaging_method=}, "
        + f"{decision_method=}, {span=}, {consecutive=}"
    )

    df["rolling_average"] = calculate_rolling_average(
        df["close"], averaging_method, span
    )

    decision_method_function_call = globals()[decision_method]
    df["trend"] = (
        df["rolling_average"]
        .rolling(consecutive)
        .apply(decision_method_function_call, raw=True, engine="numba")
    )

    del df["rolling_average"]

    print(f"Calculating trend done in {time.perf_counter() - t:>3.2f}s", end="\n\n")
    return df


@numba.jit
def monotonic(C: list) -> int:
    """
    Trend calculation based on strict in/decreases.

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

    for i in range(len(C)):
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
    elif decreasing:
        return -1
