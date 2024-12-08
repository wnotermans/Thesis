import numpy as np


def percentiles(df: np.ndarray) -> tuple:
    """Calculates percentiles of the data for calibration.

    Inputs
    ------
    df with OHLC data.

    Outputs
    -------
    A tuple with 25/50/75-percentiles of the body length, upper shadow length and lower
    shadow length, respectively.
    """

    def hb(O: float, C: float) -> float:
        return np.abs(O - C) / C

    def top_body(O: float, C: float) -> float:
        return np.maximum(O, C)

    def bottom_body(O: float, C: float) -> float:
        return np.minimum(O, C)

    def upper_shadow(O: float, H: float, C: float) -> float:
        return H - top_body(O, C)

    def lower_shadow(O: float, L: float, C: float) -> float:
        return bottom_body(O, C) - L

    O = df[:, 0]
    H = df[:, 1]
    L = df[:, 2]
    C = df[:, 3]
    return (
        np.nanpercentile(hb(O, C), [25, 50, 75]),
        np.nanpercentile(upper_shadow(O, H, C), [25, 50, 75]),
        np.nanpercentile(lower_shadow(O, L, C), [25, 50, 75]),
    )
