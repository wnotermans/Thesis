import numpy as np
from scipy.stats import ks_2samp

from shared import constants


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

    def body_length(OP: np.ndarray, C: np.ndarray) -> np.ndarray:
        return np.abs(OP - C)

    def upper_shadow_length(OP: np.ndarray, H: np.ndarray, C: np.ndarray) -> np.ndarray:
        return H - np.maximum(OP, C)

    def lower_shadow_length(OP: np.ndarray, L: np.ndarray, C: np.ndarray) -> np.ndarray:
        return np.minimum(OP, C) - L

    OP = df[:, 0]
    H = df[:, 1]
    L = df[:, 2]
    C = df[:, 3]

    black_idx = OP > C
    white_idx = C > OP
    black_lengths = body_length(OP[black_idx], C[black_idx])
    white_lengths = body_length(OP[white_idx], C[white_idx])
    combined_lengths = body_length(OP, C)
    upper_shadow_lengths = upper_shadow_length(OP, H, C)
    lower_shadow_lengths = lower_shadow_length(OP, L, C)

    if ks_2samp(black_lengths, white_lengths).pvalue < constants.ONE_STAR_SIGNIFICANCE:
        return (
            np.percentile(black_lengths, constants.BODY_PERCENTILES),
            np.percentile(white_lengths, constants.BODY_PERCENTILES),
            np.percentile(upper_shadow_lengths, constants.SHADOW_PERCENTILES),
            np.percentile(lower_shadow_lengths, constants.SHADOW_PERCENTILES),
        )
    return (
        np.percentile(combined_lengths, constants.BODY_PERCENTILES),
        np.percentile(upper_shadow_lengths, constants.SHADOW_PERCENTILES),
        np.percentile(lower_shadow_lengths, constants.SHADOW_PERCENTILES),
    )
