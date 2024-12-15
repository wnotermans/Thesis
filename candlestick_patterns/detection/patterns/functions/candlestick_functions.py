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


def sli_greater(x: float, y: float) -> bool:
    """
    Checks if x is slightly greater than y.

    Parameters
    ----------
    x : float
    y : float

    Returns
    -------
    bool
        True if x is slightly greater than y, which is defined through a percentile.
        False otherwise.
    """
    return np.logical_and(0 < (x - y) / y, (x - y) / y < 0.0001148)


def mod_greater(x: float, y: float) -> bool:
    """
    Checks if x is moderately greater than y.

    Parameters
    ----------
    x : float
    y : float

    Returns
    -------
    bool
        True if x is moderately greater than y, which is defined through a percentile.
        False otherwise.
    """
    return np.logical_and(0.0001148 <= (x - y) / y, (x - y) / y < 0.00017857)


def lar_greater(x: float, y: float) -> bool:
    """
    Checks if x is largely greater than y.

    Parameters
    ----------
    x : float
    y : float

    Returns
    -------
    bool
        True if x is largely greater than y, which is defined through a percentile.
        False otherwise.
    """
    return np.logical_and(0.00017857 <= (x - y) / y, (x - y) / y < 0.00027445)


def ext_greater(x: float, y: float) -> bool:
    """
    Checks if x is extremely greater than y.

    Parameters
    ----------
    x : float
    y : float

    Returns
    -------
    bool
        True if x is extremely greater than y, which is defined through a percentile.
        False otherwise.
    """
    return (x - y) / y > 0.00027445


def sli_less(x: float, y: float) -> bool:
    """
    Checks if x is slightly less than y.

    Parameters
    ----------
    x : float
    y : float

    Returns
    -------
    bool
        True if x is slightly less than y, which is defined through a percentile.
        False otherwise.
    """
    return np.logical_and(0 < (y - x) / x, (y - x) / x < 0.0001148)


def mod_less(x: float, y: float) -> bool:
    """
    Checks if x is moderately less than y.

    Parameters
    ----------
    x : float
    y : float

    Returns
    -------
    bool
        True if x is moderately less than y, which is defined through a percentile.
        False otherwise.
    """
    return np.logical_and(0.0001148 <= (y - x) / x, (y - x) / x < 0.00017857)


def lar_less(x: float, y: float) -> bool:
    """
    Checks if x is largely less than y.

    Parameters
    ----------
    x : float
    y : float

    Returns
    -------
    bool
        True if x is largely less than y, which is defined through a percentile.
        False otherwise.
    """
    return np.logical_and(0.00017857 <= (y - x) / x, (y - x) / x < 0.00027445)


def ext_less(x: float, y: float) -> bool:
    """
    Checks if x is extremely less than y.

    Parameters
    ----------
    x : float
    y : float

    Returns
    -------
    bool
        True if x is extremely less than y, which is defined through a percentile.
        False otherwise.
    """
    return (y - x) / x > 0.00027445


def mod_near(x: float, y: float) -> bool:
    """
    Checks if x and y are moderately near to each other.

    Parameters
    ----------
    x : float
    y : float

    Returns
    -------
    bool
        True if x and y are moderately near, which is defined through a percentile.
        False otherwise.
    """
    return np.logical_and(
        0 < np.abs(x - y) / np.maximum(x, y),
        np.abs(x - y) / np.maximum(x, y) < 0.0001148,
    )


def near(x: float, y: float) -> bool:
    """
    Checks if x and y are near to each other.

    Parameters
    ----------
    x : float
    y : float

    Returns
    -------
    bool
        True if x and y are near, which is defined through a percentile.
        False otherwise.
    """
    return np.abs(x - y) / np.maximum(x, y) <= 0.00017857


def near_up(x: float, y: float) -> bool:
    """
    Checks if x and y are near to each other, with x smaller than y.

    Parameters
    ----------
    x : float
    y : float

    Returns
    -------
    bool
        True if x and y are near and x is smaller than y, which is defined through a
        percentile. False otherwise.
    """
    return (y - x) / y < 0.00017857


def doji(O: float, C: float) -> bool:
    """
    Checks if the candle is a doji.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if open == close.
    """
    return O == C


def short_body(O: float, C: float) -> bool:
    """
    Checks if the candle has a short body.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if bottom of the body is slightly less than the top. False otherwise.
    """
    return sli_less(bottom_body(O, C), top_body(O, C))


def normal_body(O: float, C: float) -> bool:
    """
    Checks if the candle has a normal body.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if bottom of the body is moderately less than the top. False otherwise.
    """
    return mod_less(bottom_body(O, C), top_body(O, C))


def tall_body(O: float, C: float) -> bool:
    """
    Checks if the candle has a tall body.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if bottom of the body is largely less than the top. False otherwise.
    """
    return lar_less(bottom_body(O, C), top_body(O, C))


def extall_body(O: float, C: float) -> bool:
    """
    Checks if the candle has a extremely tall body.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if bottom of the body is extremely less than the top. False otherwise.
    """
    return ext_less(bottom_body(O, C), top_body(O, C))


def no_us(O: float, H: float, C: float) -> bool:
    """
    Checks if the candle has no upper shadow.

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
    bool
        True if high equals the top of the body. False otherwise.
    """
    return H == top_body(O, C)


def small_us(O: float, H: float, C: float) -> bool:
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

    Returns
    -------
    bool
        True if high is slightly greater than the top of the body. False otherwise.
    """
    return sli_greater(H, top_body(O, C))


def normal_us(O: float, H: float, C: float) -> bool:
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

    Returns
    -------
    bool
        True if high is moderately greater than the top of the body. False otherwise.
    """
    return mod_greater(H, top_body(O, C))


def long_us(O: float, H: float, C: float) -> bool:
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

    Returns
    -------
    bool
        True if high is largely greater than the top of the body. False otherwise.
    """
    return lar_greater(H, top_body(O, C))


def exlong_us(O: float, H: float, C: float) -> bool:
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

    Returns
    -------
    bool
        True if high is extremely greater than the top of the body. False otherwise.
    """
    return ext_greater(H, top_body(O, C))


def no_ls(O: float, L: float, C: float) -> bool:
    """
    Checks if the candle has no lower shadow.

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
    bool
        True if low equals the bottom of the body. False otherwise
    """
    return L == bottom_body(O, C)


def small_ls(O: float, L: float, C: float) -> bool:
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

    Returns
    -------
    bool
        True if low is slightly less than the bottom of the body. False otherwise.
    """
    return sli_less(L, bottom_body(O, C))


def normal_ls(O: float, L: float, C: float) -> bool:
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

    Returns
    -------
    bool
        True if low is moderately less than the bottom of the body. False otherwise.
    """
    return mod_less(L, bottom_body(O, C))


def long_ls(O: float, L: float, C: float) -> bool:
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

    Returns
    -------
    bool
        True if low is largely less than the bottom of the body. False otherwise.
    """
    return lar_less(L, bottom_body(O, C))


def exlong_ls(O: float, L: float, C: float) -> bool:
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

    Returns
    -------
    bool
        True if low is extremely less than the bottom of the body. False otherwise.
    """
    return ext_less(L, bottom_body(O, C))


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


def short_black_body(O: float, C: float) -> bool:
    """
    Checks if the candle is black and has a short body.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if open is strictly largar than close and body is short. False otherwise.
    """
    return np.logical_and(O > C, short_body(O, C))


def normal_black_body(O: float, C: float) -> bool:
    """
    Checks if the candle is black and has a normal body.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if open is strictly largar than close and body is normal. False otherwise.
    """
    return np.logical_and(O > C, normal_body(O, C))


def tall_black_body(O: float, C: float) -> bool:
    """
    Checks if the candle is black and has a tall body.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if open is strictly largar than close and body is tall. False otherwise.
    """
    return np.logical_and(O > C, tall_body(O, C))


def extall_black_body(O: float, C: float) -> bool:
    """
    Checks if the candle is black and has an extremely tall body.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if open is strictly largar than close and body is extremely tall.
        False otherwise.
    """
    return np.logical_and(O > C, extall_body(O, C))


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


def short_white_body(O: float, C: float) -> bool:
    """
    Checks if the candle is white and the body is short.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if open is strictly smaller than close and body is short. False otherwise.
    """
    return np.logical_and(O < C, short_body(O, C))


def normal_white_body(O: float, C: float) -> bool:
    """
    Checks if the candle is white and the body is normal.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if open is strictly smaller than close and body is normal. False otherwise.
    """
    return np.logical_and(O < C, normal_body(O, C))


def tall_white_body(O: float, C: float) -> bool:
    """
    Checks if the candle is white and the body is tall.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if open is strictly smaller than close and body is tall. False otherwise.
    """
    return np.logical_and(O < C, tall_body(O, C))


def extall_white_body(O: float, C: float) -> bool:
    """
    Checks if the candle is white and the body is extremely tall.

    Parameters
    ----------
    O : float
        Open.
    C : float
        Close.

    Returns
    -------
    bool
        True if open is strictly smaller than close and body is extremely tall.
        False otherwise.
    """
    return np.logical_and(O < C, extall_body(O, C))


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
