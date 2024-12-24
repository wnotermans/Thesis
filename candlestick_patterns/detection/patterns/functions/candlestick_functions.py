import numpy as np


def body_height(O: float, C: float) -> float:
    """
    Calculates the height of the real body without normalization.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    float
        The height of the real body without normalization.
    """
    return np.abs(O - C)


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
    if len(percentile) == 3:
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


def doji(O: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle is a doji.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if length of the body is in the appropiate percentile. False otherwise.
    """
    if len(percentile) == 3:
        return body_height(O, C) < percentile[0][0]
    return body_height(O, C) < (percentile[0][0] + percentile[1][0]) / 2


def short_body(O: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle has a short body.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if length of the body is between the appropiate percentiles.
        False otherwise.
    """
    if len(percentile) == 3:
        return np.logical_and(
            body_height(O, C) >= percentile[0][0], body_height(O, C) < percentile[0][1]
        )
    return np.logical_and.reduce(
        (
            body_height(O, C) >= percentile[0][0],
            body_height(O, C) < percentile[0][1],
            body_height(O, C) >= percentile[1][0],
            body_height(O, C) < percentile[1][1],
        )
    )


def no_us(O: float, H: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle has (almost) no upper shadow.

    Parameters
    ----------
    O : float
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
        True if length of the upper shadow is in the appropiate percentile.
        False otherwise.
    """
    return upper_shadow_length(O, H, C) <= percentile[-2][0]


def small_us(O: float, H: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the upper shadow is small.

    Parameters
    ----------
    O : float
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
        True if length of the upper shadow is between the appropiate percentiles.
        False otherwise.
    """
    return np.logical_and(
        upper_shadow_length(O, H, C) > percentile[-2][0],
        upper_shadow_length(O, H, C) <= percentile[-2][1],
    )


def normal_us(O: float, H: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the upper shadow is normal.

    Parameters
    ----------
    O : float
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
        True if length of the upper shadow is between the appropiate percentiles.
        False otherwise.
    """
    return np.logical_and(
        upper_shadow_length(O, H, C) > percentile[-2][1],
        upper_shadow_length(O, H, C) <= percentile[-2][2],
    )


def long_us(O: float, H: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the upper shadow is long.

    Parameters
    ----------
    O : float
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
        True if length of the upper shadow is between the appropiate percentiles.
        False otherwise.
    """
    return np.logical_and(
        upper_shadow_length(O, H, C) > percentile[-2][2],
        upper_shadow_length(O, H, C) <= percentile[-2][3],
    )


def exlong_us(O: float, H: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the upper shadow is extremely long.

    Parameters
    ----------
    O : float
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
        True if length of the upper shadow is between the appropiate percentiles.
        False otherwise.
    """
    return upper_shadow_length(O, H, C) > percentile[-2][3]


def no_ls(O: float, L: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle has (almost) no lower shadow.

    Parameters
    ----------
    O : float
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
        True if length of the lower shadow is in the appropiate percentile.
        False otherwise.
    """
    return lower_shadow_length(O, L, C) <= percentile[-1][0]


def small_ls(O: float, L: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the lower shadow is small.

    Parameters
    ----------
    O : float
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
        True if length of the lower shadow is between the appropiate percentiles.
        False otherwise.
    """
    return np.logical_and(
        lower_shadow_length(O, L, C) > percentile[-1][0],
        lower_shadow_length(O, L, C) <= percentile[-1][1],
    )


def normal_ls(O: float, L: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the lower shadow is normal.

    Parameters
    ----------
    O : float
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
        True if length of the lower shadow is between the appropiate percentiles.
        False otherwise.
    """
    return np.logical_and(
        lower_shadow_length(O, L, C) > percentile[-1][1],
        lower_shadow_length(O, L, C) <= percentile[-1][2],
    )


def long_ls(O: float, L: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the lower shadow is long.

    Parameters
    ----------
    O : float
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
        True if length of the lower shadow is between the appropiate percentiles.
        False otherwise.
    """
    return np.logical_and(
        lower_shadow_length(O, L, C) > percentile[-1][2],
        lower_shadow_length(O, L, C) <= percentile[-1][3],
    )


def exlong_ls(O: float, L: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the lower shadow is extremely long.

    Parameters
    ----------
    O : float
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
        True if length of the lower shadow is between the appropiate percentiles.
        False otherwise.
    """
    return lower_shadow_length(O, L, C) > percentile[-1][3]


def top_body(O: float, C: float) -> float:
    """
    Computes the top of the body.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    float
        The top of the body, which is the maximum of the open and close.
    """
    return np.maximum(O, C)


def bottom_body(O: float, C: float) -> float:
    """
    Computes the bottom of the body.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    float
        The bottom of the body, which is the minimum of the open and close.
    """
    return np.minimum(O, C)


def upper_shadow_length(O: float, H: float, C: float) -> float:
    """
    Computes the length of the upper shadow.

    Parameters
    ----------
    O : float
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
    return H - top_body(O, C)


def lower_shadow_length(O: float, L: float, C: float) -> float:
    """
    Computes the length of the lower shadow.

    Parameters
    ----------
    O : float
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
    return bottom_body(O, C) - L


def total_shadow_length(O: float, H: float, L: float, C: float) -> float:
    """
    Computes the length of both shadows added up.

    Parameters
    ----------
    O : float
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
    return upper_shadow_length(O, H, C) + lower_shadow_length(O, L, C)


def black_body(O: float, C: float) -> bool:
    """
    Checks if the candle is black.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if open is strictly larger than close. False otherwise.
    """
    return O > C


def short_black_body(O: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle is black and has a short body.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if open is strictly larger than close and length of the body is in the
        appropiate percentile. False otherwise.
    """
    return np.logical_and.reduce(
        (
            O > C,
            body_height(O, C) >= percentile[0][0],
            body_height(O, C) < percentile[0][1],
        )
    )


def normal_black_body(O: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle is black and has a normal body.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if open is strictly larger than close and length of the body is in the
        appropiate percentile. False otherwise.
    """
    return np.logical_and.reduce(
        (
            O > C,
            body_height(O, C) >= percentile[0][1],
            body_height(O, C) < percentile[0][2],
        )
    )


def tall_black_body(O: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle is black and has a tall body.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.
    percentile: tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if open is strictly larger than close and length of the body is in the
        appropiate percentile. False otherwise.
    """
    return np.logical_and(
        O > C,
        body_height(O, C) >= percentile[0][2],
    )


def white_body(O: float, C: float) -> bool:
    """
    Checks if the candle is white.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if open is strictly smaller than close. False otherwise.
    """
    return O < C


def short_white_body(O: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle is white and the body is short.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if open is strictly smaller than close and length of the body is in the
        appropiate percentile. False otherwise.
    """
    if len(percentile) == 3:
        return np.logical_and.reduce(
            (
                O < C,
                body_height(O, C) >= percentile[0][0],
                body_height(O, C) < percentile[0][1],
            )
        )
    return np.logical_and.reduce(
        (
            O < C,
            body_height(O, C) >= percentile[1][0],
            body_height(O, C) < percentile[1][1],
        )
    )


def normal_white_body(O: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle is white and the body is normal.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if open is strictly smaller than close and length of the body is in the
        appropiate percentile. False otherwise.
    """
    if len(percentile) == 3:
        return np.logical_and.reduce(
            (
                O < C,
                body_height(O, C) >= percentile[0][1],
                body_height(O, C) < percentile[0][2],
            )
        )
    return np.logical_and.reduce(
        (
            O < C,
            body_height(O, C) >= percentile[1][1],
            body_height(O, C) < percentile[1][2],
        )
    )


def tall_white_body(O: float, C: float, percentile: tuple) -> bool:
    """
    Checks if the candle is white and the body is tall.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.
    percentile : tuple
        Tuple of length percentiles.

    Returns
    -------
    bool
        True if open is strictly smaller than close and length of the body is in the
        appropiate percentile. False otherwise.
    """
    if len(percentile) == 3:
        return np.logical_and(O < C, body_height(O, C) >= percentile[0][2])
    return np.logical_and(
        O < C,
        body_height(O, C) >= percentile[1][2],
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
    Checks for a downwards shadow gap between two candles.

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
