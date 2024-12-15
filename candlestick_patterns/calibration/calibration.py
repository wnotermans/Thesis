import numpy as np


def calculate_percentiles(df: np.ndarray) -> tuple[list, list, list]:
    """
    Calculates the percentiles of the data for calibration.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing OHLC (Open, High, Low, Close) data.

    Returns
    -------
    tuple
        A tuple containing a list of the 10th, 30th, and 70th percentiles for the
        following: body length, upper shadow length, and lower shadow length,
        respectively. These percentiles correspond to the ones given in
        "The Classification of Candlestick Charts: Laying the Foundation for Further
        Empirical Research" by Etschberger et al.
    """

    def hb(O: float, C: float) -> float:
        return np.abs(O / C - 1)

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
        np.nanpercentile(hb(O, C), [10, 30, 70]),
        np.nanpercentile(upper_shadow(O, H, C), [10, 30, 70]),
        np.nanpercentile(lower_shadow(O, L, C), [10, 30, 70]),
    )
