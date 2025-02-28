import time

import numba
import numpy as np
import pandas as pd


def calculate_rolling_average(
    ser: pd.Series, averaging_method: str = "MA", span: int = 5
) -> pd.Series:
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
