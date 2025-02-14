import numpy as np
from scipy.stats import ks_2samp


def calculate_percentiles(df: np.ndarray) -> tuple:
    """
    Calculates the percentiles of the data for calibration.

    Parameters
    ----------
    df : np.ndarray
        An array containing OHLC (Open, High, Low, Close) data.

    Returns
    -------
    tuple
        A tuple containing lists of the 10th, 30th, and 70th percentiles for the
        body lengths, split up into black and white candles separately if the lengths do
        not come from the same distribution (tested with a two-sample
        Kolmogorov-Smirnov test), as well as lists of the 10th, 30th, 70th and 90th
        percentiles of the upper and lower shadows.
    """

    def body_length(O: float, C: float) -> float:
        return np.abs(O - C)

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

    black_length = body_length(O[black_idx], C[black_idx])
    white_length = body_length(O[white_idx], C[white_idx])

    if ks_2samp(black_length, white_length).pvalue < 0.05:
        return (
            np.percentile(black_length, [10, 30, 70]),
            np.percentile(white_length, [10, 30, 70]),
            np.percentile(upper_shadow_length(O, H, C), [10, 30, 70, 90]),
            np.percentile(lower_shadow_length(O, L, C), [10, 30, 70, 90]),
        )
    else:
        return (
            np.percentile(body_length(O, C), [10, 30, 70]),
            np.percentile(upper_shadow_length(O, H, C), [10, 30, 70, 90]),
            np.percentile(lower_shadow_length(O, L, C), [10, 30, 70, 90]),
        )
