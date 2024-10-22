"""Module that provides functions to detect candlestick patterns.
"""

import candlestick_functions as cf
import numba


@numba.jit
def candle_black(O: list, H: list, L: list, C: list) -> bool:
    """Definition: black candle without shadows that do not exceed the length of the body.
    Trend: either.
    Prediction: continuation.
    """
    return (
        cf.black_body(O, C)
        and cf.upper_shadow(O, H, C) < cf.hb(O, C)
        and cf.lower_shadow(O, H, C) < cf.hb(O, C)
        and cf.upper_shadow(O, H, C) != 0
        and cf.lower_shadow(O, L, C) != 0
    )


@numba.jit
def candle_white(O: list, H: list, L: list, C: list) -> bool:
    """Definition: white candle without shadows that do not exceed the length of the body.
    Trend: either.
    Prediction: continuation.
    """
    return (
        cf.white_body(O, C)
        and cf.upper_shadow(O, H, C) < cf.hb(O, C)
        and cf.lower_shadow(O, H, C) < cf.hb(O, C)
        and cf.upper_shadow(O, H, C) != 0
        and cf.lower_shadow(O, L, C) != 0
    )


@numba.jit
def marubozu_black(O: list, H: list, L: list, C: list) -> bool:
    """Definition: tall black candle without shadows.
    Trend: either.
    Prediction: continuation.
    """
    return (
        cf.black_body(O, C)
        and cf.upper_shadow(O, H, C) == 0
        and cf.lower_shadow(O, L, C) == 0
    )


@numba.jit
def marubozu_closing_black(O: list, H: list, L: list, C: list) -> bool:
    """Definition: tall black candle without lower shadow.
    Trend: either.
    Prediction: continuation.
    """
    return (
        cf.black_body(O, C)
        and cf.lower_shadow(O, L, C) == 0
        and cf.upper_shadow(O, H, C) != 0
    )


@numba.jit
def marubozu_closing_white(O: list, H: list, L: list, C: list) -> bool:
    """Definition: tall white candle without upper shadow.
    Trend: either.
    Prediction: continuation.
    """
    return (
        cf.white_body(O, C)
        and cf.upper_shadow(O, H, C) == 0
        and cf.lower_shadow(O, L, C) != 0
    )


@numba.jit
def marubozu_opening_black(O: list, H: list, L: list, C: list) -> bool:
    """Definition: tall black candle without upper shadow.
    Trend: either.
    Prediction: continuation.
    """
    return (
        cf.black_body(O, C)
        and cf.upper_shadow(O, H, C) == 0
        and cf.lower_shadow(O, L, C) != 0
    )


@numba.jit
def marubozu_opening_white(O: list, H: list, L: list, C: list) -> bool:
    """Definition: tall white candle without lower shadow.
    Trend: either.
    Prediction: continuation.
    """
    return (
        cf.white_body(O, C)
        and cf.lower_shadow(O, L, C) == 0
        and cf.upper_shadow(O, H, C) != 0
    )


@numba.jit
def marubozu_white(O: list, H: list, L: list, C: list) -> bool:
    """Definition: tall white candle without shadows.
    Trend: either.
    Prediction: continuation.
    """
    return (
        cf.white_body(O, C)
        and cf.upper_shadow(O, H, C) == 0
        and cf.lower_shadow(O, L, C) == 0
    )
