import numpy as np


def calculate_percentiles(df: np.ndarray) -> tuple[list, list, list, list]:
    """
    Calculates the percentiles of the data for calibration.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing OHLC (Open, High, Low, Close) data.

    Returns
    -------
    tuple
        A tuple containing lists of the 10th, 30th, and 70th percentiles for the
        black and white candles, as well as lists of the 10th, 30th, 70th and 90th
        percentiles of the upper and lower shadows.
    """

    def body_length(O: float, C: float) -> float:
        return np.abs(O / C - 1)

    def top_body(O: float, C: float) -> float:
        return np.maximum(O, C)

    def bottom_body(O: float, C: float) -> float:
        return np.minimum(O, C)

    def upper_shadow_length(O: float, H: float, C: float) -> float:
        return H - top_body(O, C)

    def lower_shadow_length(O: float, L: float, C: float) -> float:
        return bottom_body(O, C) - L

    O = df[:, 0]
    H = df[:, 1]
    L = df[:, 2]
    C = df[:, 3]

    black_idx = O > C
    white_idx = C > O

    return (
        np.percentile(body_length(O[black_idx], C[black_idx]), [10, 30, 70]),
        np.percentile(body_length(O[white_idx], C[white_idx]), [10, 30, 70]),
        np.percentile(upper_shadow_length(O, H, C), [10, 30, 70, 90]),
        np.percentile(lower_shadow_length(O, L, C), [10, 30, 70, 90]),
    )
