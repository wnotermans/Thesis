"""Functions related to candlestick parameters, such as height of the body, shadows...
"""

import numpy as np


def hb(O: float, C: float) -> float:
    """Inputs: open and close.

    Outputs: the height of the real body.

    NOTE: Normalization: none.
    """
    return np.abs(O - C)


def sli_greater(x: float, y: float) -> bool:
    """Inputs: x and y, floats.

    Outputs: true if x is slightly greater then y.

    NOTE: Normalization: y.
    """
    return np.logical_and(0 < (x - y) / y, (x - y) / y < 0.0001148)


def mod_greater(x: float, y: float) -> bool:
    """Inputs: x and y, floats.

    Outputs: true if x is moderately greater then y.

    NOTE: Normalization: y.
    """
    return np.logical_and(0.0001148 <= (x - y) / y, (x - y) / y < 0.00017857)


def lar_greater(x: float, y: float) -> bool:
    """Inputs: x and y, floats.

    Outputs: true if x is largely greater then y.

    NOTE: Normalization: y.
    """
    return np.logical_and(0.00017857 <= (x - y) / y, (x - y) / y < 0.00027445)


def ext_greater(x: float, y: float) -> bool:
    """Inputs: x and y, floats.

    Outputs: true if x is extremely greater then y.

    NOTE: Normalization: y.
    """
    return (x - y) / y > 0.00027445


def sli_less(x: float, y: float) -> bool:
    """Inputs: x and y, floats.

    Outputs: true if x is slightly less then y.

    NOTE: Normalization: x.
    """
    return np.logical_and(0 < (y - x) / x, (y - x) / x < 0.0001148)


def mod_less(x: float, y: float) -> bool:
    """Inputs: x and y, floats.

    Outputs: true if x is moderately less then y.

    NOTE: Normalization: x.
    """
    return np.logical_and(0.0001148 <= (y - x) / x, (y - x) / x < 0.00017857)


def lar_less(x: float, y: float) -> bool:
    """Inputs: x and y, floats.

    Outputs: true if x is largely less then y.

    NOTE: Normalization: x.
    """
    return np.logical_and(0.00017857 <= (y - x) / x, (y - x) / x < 0.00027445)


def ext_less(x: float, y: float) -> bool:
    """Inputs: x and y, floats.

    Outputs: true if x is extremely less then y.

    NOTE: Normalization: x.
    """
    return (y - x) / x > 0.00027445


def mod_near(x: float, y: float) -> bool:
    """Inputs: x and y, floats.

    Outputs: true if x and y are moderately near.

    NOTE: Normalization: max(x,y)
    """
    return np.logical_and(
        0 < np.abs(x - y) / np.maximum(x, y),
        np.abs(x - y) / np.maximum(x, y) < 0.0001148,
    )


def near(x: float, y: float) -> bool:
    """Inputs: x and y, floats.

    Outputs: true if x and y are near.

    NOTE: Normalization: max(x,y)
    """
    return np.abs(x - y) / np.maximum(x, y) <= 0.00017857


def near_up(x: float, y: float) -> bool:
    """Inputs: x and y, floats.

    Outputs: true if x and y are near, with x < y.

    NOTE: Normalization: y
    """
    return (y - x) / y < 0.00017857


def doji(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: true if open == close.
    """
    return O == C


def short_body(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: true if bottom of the body is slightly less than the top.
    """
    return sli_less(bottom_body(O, C), top_body(O, C))


def normal_body(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: true if bottom of the body is moderately less than the top.
    """
    return mod_less(bottom_body(O, C), top_body(O, C))


def tall_body(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: true if bottom of the body is largely less than the top.
    """
    return lar_less(bottom_body(O, C), top_body(O, C))


def extall_body(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: true if bottom of the body is extremely less than the top.
    """
    return ext_less(bottom_body(O, C), top_body(O, C))


def no_us(O: float, H: float, C: float) -> bool:
    """Inputs: open, high and close.

    Outputs: true if high == the top of the body.
    """
    return H == top_body(O, C)


def small_us(O: float, H: float, C: float) -> bool:
    """Inputs: open, high and close.

    Outputs: true if high is slightly greater than the top of the body.
    """
    return sli_greater(H, top_body(O, C))


def normal_us(O: float, H: float, C: float) -> bool:
    """Inputs: open, high and close.

    Outputs: true if high is moderately greater than the top of the body.
    """
    return mod_greater(H, top_body(O, C))


def long_us(O: float, H: float, C: float) -> bool:
    """Inputs: open, high and close.

    Outputs: true if high is largely greater than the top of the body.
    """
    return lar_greater(H, top_body(O, C))


def exlong_us(O: float, H: float, C: float) -> bool:
    """Inputs: open, high and close.

    Outputs: true if high is extremely greater than the top of the body.
    """
    return ext_greater(H, top_body(O, C))


def no_ls(O: float, L: float, C: float) -> bool:
    """Inputs: open, low and close.

    Outputs: true if low == the bottom of the body.
    """
    return L == bottom_body(O, C)


def small_ls(O: float, L: float, C: float) -> bool:
    """Inputs: open, low and close.

    Outputs: true if low is slightly less than the bottom of the body.
    """
    return sli_less(L, bottom_body(O, C))


def normal_ls(O: float, L: float, C: float) -> bool:
    """Inputs: open, low and close.

    Outputs: true if low is moderately less than the bottom of the body.
    """
    return mod_less(L, bottom_body(O, C))


def long_ls(O: float, L: float, C: float) -> bool:
    """Inputs: open, low and close.

    Outputs: true if low is largely less than the bottom of the body.
    """
    return lar_less(L, bottom_body(O, C))


def exlong_ls(O: float, L: float, C: float) -> bool:
    """Inputs: open, low and close.

    Outputs: true if low is extremely less than the bottom of the body.
    """
    return ext_less(L, bottom_body(O, C))


def top_body(O: float, C: float) -> float:
    """Inputs: open and close.

    Outputs: the top of the body.
    """
    return np.maximum(O, C)


def bottom_body(O: float, C: float) -> float:
    """Inputs: open and close.

    Outputs: the bottom of the body.
    """
    return np.minimum(O, C)


def upper_shadow(O: float, H: float, C: float) -> float:
    """Inputs: open, high and close.

    Outputs: the length of the upper shadow.
    """
    return H - top_body(O, C)


def lower_shadow(O: float, L: float, C: float) -> float:
    """Inputs: open, low and close.

    Outputs: the length of the lower shadow.
    """
    return bottom_body(O, C) - L


def shadow_length(O: float, H: float, L: float, C: float) -> float:
    """Inputs: open, high, low and close.

    Outputs: the total length of the shadows.
    """
    return upper_shadow(O, H, C) + lower_shadow(O, L, C)


def black_body(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: True if open > close.
    """
    return O > C


def short_black_body(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: True if open > close and body is small.
    """
    return np.logical_and(O > C, short_body(O, C))


def normal_black_body(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: True if open > close and body is normal.
    """
    return np.logical_and(O > C, normal_body(O, C))


def tall_black_body(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: True if open > close and body is tall.
    """
    return np.logical_and(O > C, tall_body(O, C))


def extall_black_body(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: True if open > close and body is extremely tall.
    """
    return np.logical_and(O > C, extall_body(O, C))


def white_body(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: True if open < close.
    """
    return O < C


def short_white_body(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: True if open < close and body is small.
    """
    return np.logical_and(O < C, short_body(O, C))


def normal_white_body(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: True if open < close and body is normal.
    """
    return np.logical_and(O < C, normal_body(O, C))


def tall_white_body(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: True if open < close and body is tall.
    """
    return np.logical_and(O < C, tall_body(O, C))


def extall_white_body(O: float, C: float) -> bool:
    """Inputs: open and close.

    Outputs: True if open < close and body is extremely tall.
    """
    return np.logical_and(O < C, extall_body(O, C))


def down_shadow_gap(first_L: float, second_H: float) -> bool:
    """Inputs: first candles low, second candles high.

    Outputs: True if first low > second high.
    """
    return first_L > second_H


def up_shadow_gap(first_H: float, second_L: float) -> bool:
    """Inputs: first candles high, second candles low.

    Outputs: True if first high < second low.
    """
    return first_H < second_L


def down_body_gap(
    first_O: float, first_C: float, second_O: float, second_C: float
) -> bool:
    """Inputs: first and second open and close.

    Outputs: True if first bottom of body > second top of body.
    """
    return bottom_body(first_O, first_C) > top_body(second_O, second_C)


def up_body_gap(
    first_O: float, first_C: float, second_O: float, second_C: float
) -> bool:
    """Inputs: first and second open and close.

    Outputs: True if previous top of body < current bottom of body.
    """
    return top_body(first_O, first_C) < bottom_body(second_O, second_C)
