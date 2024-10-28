"""Module that provides functions to detect candlestick patterns.
"""

import candlestick_functions as cf
import numba


@numba.jit
def belt_hold_bearish(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: tall black candle in an uptrend with no upper shadow that closes near
    the low.
    Trend: up.
    Prediction: reversal.
    """
    return (
        T == 1 and cf.tall_black_body(O, C) and cf.no_us(O, H, C) and cf.mod_near(L, C)
    )


@numba.jit
def belt_hold_bearish_no_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall black candle in an uptrend with no upper shadow that closes near
    the low.
    Trend: up.
    Prediction: reversal.
    """
    return cf.tall_black_body(O, C) and cf.no_us(O, H, C) and cf.mod_near(L, C)


@numba.jit
def belt_hold_bearish_opp_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall black candle in an uptrend with no upper shadow that closes near
    the low.
    Trend: up.
    Prediction: reversal.
    """
    return (
        T == -1 and cf.tall_black_body(O, C) and cf.no_us(O, H, C) and cf.mod_near(L, C)
    )


@numba.jit
def belt_hold_bullish(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: tall white candle in a downtrend with no lower shadow that closes
    near the high.
    Trend: down.
    Prediction: reversal.
    """
    return (
        T == -1 and cf.tall_white_body(O, C) and cf.no_ls(O, L, C) and cf.mod_near(H, C)
    )


@numba.jit
def belt_hold_bullish_no_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall white candle in a downtrend with no lower shadow that closes
    near the high.
    Trend: down.
    Prediction: reversal.
    """
    return cf.tall_white_body(O, C) and cf.no_ls(O, L, C) and cf.mod_near(H, C)


@numba.jit
def belt_hold_bullish_opp_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall white candle in a downtrend with no lower shadow that closes
    near the high.
    Trend: down.
    Prediction: reversal.
    """
    return (
        T == 1 and cf.tall_white_body(O, C) and cf.no_ls(O, L, C) and cf.mod_near(H, C)
    )


@numba.jit
def candle_black(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.
    Trend: either.
    Prediction: either.
    """
    return (
        cf.normal_black_body(O, C)
        and not cf.no_us(O, H, C)
        and not cf.no_ls(O, L, C)
        and cf.upper_shadow(O, H, C) < cf.hb(O, C)
        and cf.lower_shadow(O, L, C) < cf.hb(O, C)
    )


@numba.jit
def candle_black_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.
    Trend: either.
    Prediction: either.
    """
    return (
        T == -1
        and cf.normal_black_body(O, C)
        and not cf.no_us(O, H, C)
        and not cf.no_ls(O, L, C)
        and cf.upper_shadow(O, H, C) < cf.hb(O, C)
        and cf.lower_shadow(O, L, C) < cf.hb(O, C)
    )


@numba.jit
def candle_black_up_trend(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.
    Trend: either.
    Prediction: either.
    """
    return (
        T == 1
        and cf.normal_black_body(O, C)
        and not cf.no_us(O, H, C)
        and not cf.no_ls(O, L, C)
        and cf.upper_shadow(O, H, C) < cf.hb(O, C)
        and cf.lower_shadow(O, L, C) < cf.hb(O, C)
    )


@numba.jit
def candle_short_black(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.
    Trend: either.
    Prediction: either.
    """
    return (
        cf.short_black_body(O, C)
        and not cf.no_us(O, H, C)
        and not cf.no_ls(O, L, C)
        and cf.upper_shadow(O, H, C) < cf.hb(O, C)
        and cf.lower_shadow(O, L, C) < cf.hb(O, C)
    )


@numba.jit
def candle_short_black_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.
    Trend: either.
    Prediction: either.
    """
    return (
        T == -1
        and cf.short_black_body(O, C)
        and not cf.no_us(O, H, C)
        and not cf.no_ls(O, L, C)
        and cf.upper_shadow(O, H, C) < cf.hb(O, C)
        and cf.lower_shadow(O, L, C) < cf.hb(O, C)
    )


@numba.jit
def candle_short_black_up_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.
    Trend: either.
    Prediction: either.
    """
    return (
        T == 1
        and cf.short_black_body(O, C)
        and not cf.no_us(O, H, C)
        and not cf.no_ls(O, L, C)
        and cf.upper_shadow(O, H, C) < cf.hb(O, C)
        and cf.lower_shadow(O, L, C) < cf.hb(O, C)
    )


@numba.jit
def candle_short_white(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.
    Trend: either.
    Prediction: either.
    """
    return (
        cf.short_white_body(O, C)
        and not cf.no_us(O, H, C)
        and not cf.no_ls(O, L, C)
        and cf.upper_shadow(O, H, C) < cf.hb(O, C)
        and cf.lower_shadow(O, L, C) < cf.hb(O, C)
    )


@numba.jit
def candle_short_white_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.
    Trend: either.
    Prediction: either.
    """
    return (
        T == -1
        and cf.short_white_body(O, C)
        and not cf.no_us(O, H, C)
        and not cf.no_ls(O, L, C)
        and cf.upper_shadow(O, H, C) < cf.hb(O, C)
        and cf.lower_shadow(O, L, C) < cf.hb(O, C)
    )


@numba.jit
def candle_short_white_up_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.
    Trend: either.
    Prediction: either.
    """
    return (
        T == 1
        and cf.short_white_body(O, C)
        and not cf.no_us(O, H, C)
        and not cf.no_ls(O, L, C)
        and cf.upper_shadow(O, H, C) < cf.hb(O, C)
        and cf.lower_shadow(O, L, C) < cf.hb(O, C)
    )


@numba.jit
def candle_white(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.
    Trend: either.
    Prediction: either.
    """
    return (
        cf.normal_white_body(O, C)
        and not cf.no_us(O, H, C)
        and not cf.no_ls(O, L, C)
        and cf.upper_shadow(O, H, C) < cf.hb(O, C)
        and cf.lower_shadow(O, L, C) < cf.hb(O, C)
    )


@numba.jit
def candle_white_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.
    Trend: either.
    Prediction: either.
    """
    return (
        T == -1
        and cf.normal_white_body(O, C)
        and not cf.no_us(O, H, C)
        and not cf.no_ls(O, L, C)
        and cf.upper_shadow(O, H, C) < cf.hb(O, C)
        and cf.lower_shadow(O, L, C) < cf.hb(O, C)
    )


@numba.jit
def candle_white_up_trend(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.
    Trend: either.
    Prediction: either.
    """
    return (
        T == 1
        and cf.normal_white_body(O, C)
        and not cf.no_us(O, H, C)
        and not cf.no_ls(O, L, C)
        and cf.upper_shadow(O, H, C) < cf.hb(O, C)
        and cf.lower_shadow(O, L, C) < cf.hb(O, C)
    )


@numba.jit
def doji_dragonfly(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: doji with long lower shadow and small upper shadow.
    Trend: either.
    Prediction: reversal if down.
    """
    return cf.doji(O, C) and cf.small_us(O, H, C) and cf.long_ls(O, L, C)


@numba.jit
def doji_dragonfly_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: doji with long lower shadow and small upper shadow.
    Trend: either.
    Prediction: reversal if down.
    """
    return T == -1 and cf.doji(O, C) and cf.small_us(O, H, C) and cf.long_ls(O, L, C)


@numba.jit
def doji_dragonfly_up_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: doji with long lower shadow and small upper shadow.
    Trend: either.
    Prediction: reversal if down.
    """
    return T == 1 and cf.doji(O, C) and cf.small_us(O, H, C) and cf.long_ls(O, L, C)


@numba.jit
def marubozu_black(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: tall black candle without shadows.
    Trend: either.
    Prediction: continuation.
    """
    return cf.tall_black_body(O, C) and cf.no_us(O, H, C) and cf.no_ls(O, L, C)


@numba.jit
def marubozu_black_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall black candle without shadows.
    Trend: either.
    Prediction: continuation.
    """
    return (
        T == -1 and cf.tall_black_body(O, C) and cf.no_us(O, H, C) and cf.no_ls(O, L, C)
    )


@numba.jit
def marubozu_black_up_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall black candle without shadows.
    Trend: either.
    Prediction: continuation.
    """
    return (
        T == 1 and cf.tall_black_body(O, C) and cf.no_us(O, H, C) and cf.no_ls(O, L, C)
    )


@numba.jit
def marubozu_closing_black(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall black candle without lower shadow.
    Trend: either.
    Prediction: continuation.
    """
    return cf.tall_black_body(O, C) and not cf.no_us(O, H, C) and cf.no_ls(O, L, C)


@numba.jit
def marubozu_closing_black_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall black candle without lower shadow.
    Trend: either.
    Prediction: continuation.
    """
    return (
        T == -1
        and cf.tall_black_body(O, C)
        and not cf.no_us(O, H, C)
        and cf.no_ls(O, L, C)
    )


@numba.jit
def marubozu_closing_black_up_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall black candle without lower shadow.
    Trend: either.
    Prediction: continuation.
    """
    return (
        T == 1
        and cf.tall_black_body(O, C)
        and not cf.no_us(O, H, C)
        and cf.no_ls(O, L, C)
    )


@numba.jit
def marubozu_closing_white(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall white candle without upper shadow.
    Trend: either.
    Prediction: continuation.
    """
    return cf.tall_white_body(O, C) and not cf.no_ls(O, L, C) and cf.no_us(O, H, C)


@numba.jit
def marubozu_closing_white_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall white candle without upper shadow.
    Trend: either.
    Prediction: continuation.
    """
    return (
        T == -1
        and cf.tall_white_body(O, C)
        and not cf.no_ls(O, L, C)
        and cf.no_us(O, H, C)
    )


@numba.jit
def marubozu_closing_white_up_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall white candle without upper shadow.
    Trend: either.
    Prediction: continuation.
    """
    return (
        T == 1
        and cf.tall_white_body(O, C)
        and not cf.no_ls(O, L, C)
        and cf.no_us(O, H, C)
    )


@numba.jit
def marubozu_opening_black(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall black candle without upper shadow.
    Trend: either.
    Prediction: continuation.
    """
    return cf.tall_black_body(O, C) and not cf.no_ls(O, L, C) and cf.no_us(O, H, C)


@numba.jit
def marubozu_opening_black_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall black candle without upper shadow.
    Trend: either.
    Prediction: continuation.
    """
    return (
        T == -1
        and cf.tall_black_body(O, C)
        and not cf.no_ls(O, L, C)
        and cf.no_us(O, H, C)
    )


@numba.jit
def marubozu_opening_black_up_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall black candle without upper shadow.
    Trend: either.
    Prediction: continuation.
    """
    return (
        T == 1
        and cf.tall_black_body(O, C)
        and not cf.no_ls(O, L, C)
        and cf.no_us(O, H, C)
    )


@numba.jit
def marubozu_opening_white(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall white candle without lower shadow.
    Trend: either.
    Prediction: continuation.
    """
    return cf.tall_white_body(O, C) and not cf.no_us(O, H, C) and cf.no_ls(O, L, C)


@numba.jit
def marubozu_opening_white_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall white candle without lower shadow.
    Trend: either.
    Prediction: continuation.
    """
    return (
        T == -1
        and cf.tall_white_body(O, C)
        and not cf.no_us(O, H, C)
        and cf.no_ls(O, L, C)
    )


@numba.jit
def marubozu_opening_white_up_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall white candle without lower shadow.
    Trend: either.
    Prediction: continuation.
    """
    return (
        T == 1
        and cf.tall_white_body(O, C)
        and not cf.no_us(O, H, C)
        and cf.no_ls(O, L, C)
    )


@numba.jit
def marubozu_white(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: tall white candle without shadows.
    Trend: either.
    Prediction: continuation.
    """
    return cf.tall_white_body(O, C) and cf.no_us(O, H, C) and cf.no_ls(O, L, C)


@numba.jit
def marubozu_white_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall white candle without shadows.
    Trend: either.
    Prediction: continuation.
    """
    return (
        T == -1 and cf.tall_white_body(O, C) and cf.no_us(O, H, C) and cf.no_ls(O, L, C)
    )


@numba.jit
def marubozu_white_up_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: tall white candle without shadows.
    Trend: either.
    Prediction: continuation.
    """
    return (
        T == 1 and cf.tall_white_body(O, C) and cf.no_us(O, H, C) and cf.no_ls(O, L, C)
    )
