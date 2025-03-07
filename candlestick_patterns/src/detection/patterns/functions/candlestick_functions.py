import numpy as np

UNIFIED_PERCENTILE_LENGTH = 3


def body_height(OP: float, C: float) -> float:
    """
    Calculates the height of the real body without normalization.

    Parameters
    ----------
    OP : float
        Open.
    C : float
        Close.

    Returns
    -------
    float
        The height of the real body without normalization.
    """
    return np.abs(OP - C)


def near(x: float, y: float, percentile: tuple) -> bool:
    """
    Checks if x and y are near to each other.

    Parameters
    ----------
    x : float
    y : float
    percentile : tuple
        Tuple of length percentiles

    Returns
    -------
    bool
        True if x and y are near, which is defined through a percentile.
        False otherwise.
    """
    if len(percentile) == UNIFIED_PERCENTILE_LENGTH:
        return np.abs(x - y) / np.maximum(x, y) < percentile[0][1]
    return np.abs(x - y) / np.maximum(x, y) < (percentile[0][1] + percentile[1][1]) / 2


def near_up(x: float, y: float, percentile: tuple) -> bool:
    """
    Checks if x and y are near to each other, with x smaller than y.

    Parameters
    ----------
    x : float
    y : float
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if x and y are near and x is smaller than y, which is defined through a
        percentile. False otherwise.
    """
    return np.logical_and(near(x, y, percentile), x < y)


def doji(OP: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle is a doji.

    Parameters
    ----------
    OP : float
        Open.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if length of the body is in the appropriate percentile. False otherwise.
    """
    if len(percentile) == UNIFIED_PERCENTILE_LENGTH:
        return body_height(OP, C) < percentile[0][0]
    return body_height(OP, C) < (percentile[0][0] + percentile[1][0]) / 2


def short_body(OP: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle has a short body.

    Parameters
    ----------
    OP : float
        Open.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if length of the body is between the appropriate percentiles.
        False otherwise.
    """
    if len(percentile) == UNIFIED_PERCENTILE_LENGTH:
        return np.logical_and(
            body_height(OP, C) >= percentile[0][0],
            body_height(OP, C) < percentile[0][1],
        )
    return np.logical_and.reduce(
        (
            body_height(OP, C) >= percentile[0][0],
            body_height(OP, C) < percentile[0][1],
            body_height(OP, C) >= percentile[1][0],
            body_height(OP, C) < percentile[1][1],
        )
    )


def no_us(OP: float, H: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle has (almost) no upper shadow.

    Parameters
    ----------
    OP : float
        Open.
    H : float
        High.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if length of the upper shadow is in the appropriate percentile.
        False otherwise.
    """
    return upper_shadow_length(OP, H, C) <= percentile[-2][0]


def small_us(OP: float, H: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the upper shadow is small.

    Parameters
    ----------
    OP : float
        Open.
    H : float
        High.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if length of the upper shadow is between the appropriate percentiles.
        False otherwise.
    """
    return np.logical_and(
        upper_shadow_length(OP, H, C) > percentile[-2][0],
        upper_shadow_length(OP, H, C) <= percentile[-2][1],
    )


def normal_us(OP: float, H: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the upper shadow is normal.

    Parameters
    ----------
    OP : float
        Open.
    H : float
        High.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if length of the upper shadow is between the appropriate percentiles.
        False otherwise.
    """
    return np.logical_and(
        upper_shadow_length(OP, H, C) > percentile[-2][1],
        upper_shadow_length(OP, H, C) <= percentile[-2][2],
    )


def long_us(OP: float, H: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the upper shadow is long.

    Parameters
    ----------
    OP : float
        Open.
    H : float
        High.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if length of the upper shadow is between the appropriate percentiles.
        False otherwise.
    """
    return np.logical_and(
        upper_shadow_length(OP, H, C) > percentile[-2][2],
        upper_shadow_length(OP, H, C) <= percentile[-2][3],
    )


def exlong_us(OP: float, H: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the upper shadow is extremely long.

    Parameters
    ----------
    OP : float
        Open.
    H : float
        High.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if length of the upper shadow is between the appropriate percentiles.
        False otherwise.
    """
    return upper_shadow_length(OP, H, C) > percentile[-2][3]


def no_ls(OP: float, L: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle has (almost) no lower shadow.

    Parameters
    ----------
    OP : float
        Open.
    L : float
        Low.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if length of the lower shadow is in the appropriate percentile.
        False otherwise.
    """
    return lower_shadow_length(OP, L, C) <= percentile[-1][0]


def small_ls(OP: float, L: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the lower shadow is small.

    Parameters
    ----------
    OP : float
        Open.
    L : float
        Low.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if length of the lower shadow is between the appropriate percentiles.
        False otherwise.
    """
    return np.logical_and(
        lower_shadow_length(OP, L, C) > percentile[-1][0],
        lower_shadow_length(OP, L, C) <= percentile[-1][1],
    )


def normal_ls(OP: float, L: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the lower shadow is normal.

    Parameters
    ----------
    OP : float
        Open.
    L : float
        Low.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if length of the lower shadow is between the appropriate percentiles.
        False otherwise.
    """
    return np.logical_and(
        lower_shadow_length(OP, L, C) > percentile[-1][1],
        lower_shadow_length(OP, L, C) <= percentile[-1][2],
    )


def long_ls(OP: float, L: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the lower shadow is long.

    Parameters
    ----------
    OP : float
        Open.
    L : float
        Low.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if length of the lower shadow is between the appropriate percentiles.
        False otherwise.
    """
    return np.logical_and(
        lower_shadow_length(OP, L, C) > percentile[-1][2],
        lower_shadow_length(OP, L, C) <= percentile[-1][3],
    )


def exlong_ls(OP: float, L: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the lower shadow is extremely long.

    Parameters
    ----------
    OP : float
        Open.
    L : float
        Low.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if length of the lower shadow is between the appropriate percentiles.
        False otherwise.
    """
    return lower_shadow_length(OP, L, C) > percentile[-1][3]


def top_body(OP: float, C: float) -> float:
    """
    Computes the top of the body.

    Parameters
    ----------
    OP : float
        Open.
    C : float
        Close.

    Returns
    -------
    float
        The top of the body, which is the maximum of the open and close.
    """
    return np.maximum(OP, C)


def bottom_body(OP: float, C: float) -> float:
    """
    Computes the bottom of the body.

    Parameters
    ----------
    OP : float
        Open.
    C : float
        Close.

    Returns
    -------
    float
        The bottom of the body, which is the minimum of the open and close.
    """
    return np.minimum(OP, C)


def upper_shadow_length(OP: float, H: float, C: float) -> float:
    """
    Computes the length of the upper shadow.

    Parameters
    ----------
    OP : float
        Open.
    H : float
        High.
    C : float
        Close.

    Returns
    -------
    float
        The length of the upper shadow.
    """
    return H - top_body(OP, C)


def lower_shadow_length(OP: float, L: float, C: float) -> float:
    """
    Computes the length of the lower shadow.

    Parameters
    ----------
    OP : float
        Open.
    L : float
        Low.
    C : float
        Close.

    Returns
    -------
    float
        The length of the lower shadow.
    """
    return bottom_body(OP, C) - L


def total_shadow_length(OP: float, H: float, L: float, C: float) -> float:
    """
    Computes the length of both shadows added up.

    Parameters
    ----------
    OP : float
        Open.
    H : float
        High.
    L : float
        Low.
    C : float
        Close.

    Returns
    -------
    float
        The total length of the shadows, which is the sum of the upper and lower
        lengths.
    """
    return upper_shadow_length(OP, H, C) + lower_shadow_length(OP, L, C)


def black_body(OP: float, C: float) -> bool:
    """
    Checks if the candle is black.

    Parameters
    ----------
    OP : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if open is strictly larger than close. False otherwise.
    """
    return OP > C


def short_black_body(OP: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle is black and has a short body.

    Parameters
    ----------
    OP : float
        Open.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if open is strictly larger than close and length of the body is in the
        appropriate percentile. False otherwise.
    """
    return np.logical_and.reduce(
        (
            OP > C,
            body_height(OP, C) >= percentile[0][0],
            body_height(OP, C) < percentile[0][1],
        )
    )


def normal_black_body(OP: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle is black and has a normal body.

    Parameters
    ----------
    OP : float
        Open.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if open is strictly larger than close and length of the body is in the
        appropriate percentile. False otherwise.
    """
    return np.logical_and.reduce(
        (
            OP > C,
            body_height(OP, C) >= percentile[0][1],
            body_height(OP, C) < percentile[0][2],
        )
    )


def tall_black_body(OP: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle is black and has a tall body.

    Parameters
    ----------
    OP : float
        Open.
    C : float
        Close.
    percentile: tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if open is strictly larger than close and length of the body is in the
        appropriate percentile. False otherwise.
    """
    return np.logical_and(
        OP > C,
        body_height(OP, C) >= percentile[0][2],
    )


def white_body(OP: float, C: float) -> bool:
    """
    Checks if the candle is white.

    Parameters
    ----------
    OP : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if open is strictly smaller than close. False otherwise.
    """
    return OP < C


def short_white_body(OP: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle is white and the body is short.

    Parameters
    ----------
    OP : float
        Open.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if open is strictly smaller than close and length of the body is in the
        appropriate percentile. False otherwise.
    """
    if len(percentile) == UNIFIED_PERCENTILE_LENGTH:
        return np.logical_and.reduce(
            (
                OP < C,
                body_height(OP, C) >= percentile[0][0],
                body_height(OP, C) < percentile[0][1],
            )
        )
    return np.logical_and.reduce(
        (
            OP < C,
            body_height(OP, C) >= percentile[1][0],
            body_height(OP, C) < percentile[1][1],
        )
    )


def normal_white_body(OP: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle is white and the body is normal.

    Parameters
    ----------
    OP : float
        Open.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if open is strictly smaller than close and length of the body is in the
        appropriate percentile. False otherwise.
    """
    if len(percentile) == UNIFIED_PERCENTILE_LENGTH:
        return np.logical_and.reduce(
            (
                OP < C,
                body_height(OP, C) >= percentile[0][1],
                body_height(OP, C) < percentile[0][2],
            )
        )
    return np.logical_and.reduce(
        (
            OP < C,
            body_height(OP, C) >= percentile[1][1],
            body_height(OP, C) < percentile[1][2],
        )
    )


def tall_white_body(OP: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle is white and the body is tall.

    Parameters
    ----------
    OP : float
        Open.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if open is strictly smaller than close and length of the body is in the
        appropriate percentile. False otherwise.
    """
    if len(percentile) == UNIFIED_PERCENTILE_LENGTH:
        return np.logical_and(OP < C, body_height(OP, C) >= percentile[0][2])
    return np.logical_and(
        OP < C,
        body_height(OP, C) >= percentile[1][2],
    )


def down_shadow_gap(first_L: float, second_H: float) -> bool:
    """
    Checks for a downwards shadow gap between two candles.

    Parameters
    ----------
    first_L : float
        Low of the first candle.
    second_H : float
        High of the second candle.

    Returns
    -------
    bool
        True if the low of the first candle is strictly greater than the high of the
        second candle. False otherwise.
    """
    return first_L > second_H


def up_shadow_gap(first_H: float, second_L: float) -> bool:
    """
    Checks for an upwards shadow gap between two candles.

    Parameters
    ----------
    first_H : float
        High of the first candle.
    second_L : float
        Low of the second candle.

    Returns
    -------
    bool
        True if the high of the first candle is strictly less than the low of the
        second candle. False otherwise.
    """
    return first_H < second_L


def down_body_gap(
    first_O: float, first_C: float, second_O: float, second_C: float
) -> bool:
    """
    Checks for a downwards body gap between two candles.

    Parameters
    ----------
    first_O : float
        Open of the first candle.
    first_C : float
        Close of the first candle.
    second_O : float
        Open of the second candle.
    second_C : float
        Close of the second candle.

    Returns
    -------
    bool
        True if the bottom of the first candles body is strictly greater than the top
        of the second candles body. False otherwise.
    """
    return bottom_body(first_O, first_C) > top_body(second_O, second_C)


def up_body_gap(
    first_O: float, first_C: float, second_O: float, second_C: float
) -> bool:
    """
    Checks for an upwards body gap between two candles.

    Parameters
    ----------
    first_O : float
        Open of the first candle.
    first_C : float
        Close of the first candle.
    second_O : float
        Open of the second candle.
    second_C : float
        Close of the second candle.

    Returns
    -------
    bool
        True if the top of the first candles body is strictly less than the bottom
        of the second candles body. False otherwise.
    """
    return top_body(first_O, first_C) < bottom_body(second_O, second_C)
