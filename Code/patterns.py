"""Module that provides functions to detect candlestick patterns.
"""

import candlestick_functions as cf
import numba


@numba.jit
def belt_hold_bearish_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
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
def belt_hold_bullish_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
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
def candle_black_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
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
def candle_short_black_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
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
def candle_short_white_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
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
def candle_white_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
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
def doji_dragonfly_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
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
def doji_gravestone_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: doji with small lower shadow and long upper shadow.
    Trend: either.
    Prediction: reversal if up.
    """
    return cf.doji(O, C) and cf.small_ls(O, L, C) and cf.long_us(O, H, C)


@numba.jit
def doji_gravestone_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: doji with small lower shadow and long upper shadow.
    Trend: either.
    Prediction: reversal if up.
    """
    return T == -1 and cf.doji(O, C) and cf.small_ls(O, L, C) and cf.long_us(O, H, C)


@numba.jit
def doji_gravestone_up_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: doji with small lower shadow and long upper shadow.
    Trend: either.
    Prediction: reversal if up.
    """
    return T == 1 and cf.doji(O, C) and cf.small_ls(O, L, C) and cf.long_us(O, H, C)


@numba.jit
def doji_long_legged_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: doji with long shadows.
    Trend: either.
    Prediction: either.
    """
    return cf.doji(O, C) and cf.long_us(O, H, C) and cf.long_ls(O, L, C)


@numba.jit
def doji_long_legged_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: doji with long shadows.
    Trend: either.
    Prediction: either.
    """
    return T == -1 and cf.doji(O, C) and cf.long_us(O, H, C) and cf.long_ls(O, L, C)


@numba.jit
def doji_long_legged_up_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: doji with long shadows.
    Trend: either.
    Prediction: either.
    """
    return T == 1 and cf.doji(O, C) and cf.long_us(O, H, C) and cf.long_ls(O, L, C)


@numba.jit
def doji_nothern_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: doji in an uptrend.
    Trend: up.
    Prediction: reversal.
    """
    return T == 1 and cf.doji(O, C)


@numba.jit
def doji_nothern_no_trend(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: doji in an uptrend.
    Trend: up.
    Prediction: reversal.
    """
    return cf.doji(O, C)


@numba.jit
def doji_nothern_opp_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: doji in an uptrend.
    Trend: up.
    Prediction: reversal.
    """
    return T == -1 and cf.doji(O, C)


@numba.jit
def doji_southern_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: doji in a downtrend.
    Trend: down.
    Prediction: reversal.
    """
    return T == -1 and cf.doji(O, C)


@numba.jit
def doji_southern_no_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: doji in a downtrend.
    Trend: down.
    Prediction: reversal.
    """
    return cf.doji(O, C)


@numba.jit
def doji_southern_opp_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: doji in a downtrend.
    Trend: down.
    Prediction: reversal.
    """
    return T == 1 and cf.doji(O, C)


@numba.jit
def hammer_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: candle with a small body of either color, lower shadow between 2 and
    3 times the length of the body and a small or no upper shadow.
    Trend: down.
    Prediction: reversal.
    """
    return (
        T == -1
        and cf.short_body(O, C)
        and not cf.no_ls(O, L, C)
        and (2 * cf.hb(O, C) < cf.lower_shadow(O, L, C) < 3 * cf.hb(O, C))
        and (cf.small_us(O, H, C) or cf.no_us(O, H, C))
    )


@numba.jit
def hammer_no_trend(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: candle with a small body of either color, lower shadow between 2 and
    3 times the length of the body and a small or no upper shadow.
    Trend: down.
    Prediction: reversal.
    """
    return (
        cf.short_body(O, C)
        and not cf.no_ls(O, L, C)
        and (2 * cf.hb(O, C) < cf.lower_shadow(O, L, C) < 3 * cf.hb(O, C))
        and (cf.small_us(O, H, C) or cf.no_us(O, H, C))
    )


@numba.jit
def hammer_opp_trend(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: candle with a small body of either color, lower shadow between 2 and
    3 times the length of the body and a small or no upper shadow.
    Trend: down.
    Prediction: reversal.
    """
    return (
        T == 1
        and cf.short_body(O, C)
        and not cf.no_ls(O, L, C)
        and (2 * cf.hb(O, C) < cf.lower_shadow(O, L, C) < 3 * cf.hb(O, C))
        and (cf.small_us(O, H, C) or cf.no_us(O, H, C))
    )


@numba.jit
def hanging_man_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: small candle of either color with no upper shadow and long lower
    shadow.
    Trend: up.
    Prediction: reversal.
    """
    return T == 1 and cf.no_us(O, H, C) and cf.long_ls(O, L, C) and cf.short_body(O, C)


@numba.jit
def hanging_man_no_trend(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: small candle of either color with no upper shadow and long lower
    shadow.
    Trend: up.
    Prediction: reversal.
    """
    return cf.no_us(O, H, C) and cf.long_ls(O, L, C) and cf.short_body(O, C)


@numba.jit
def hanging_man_opp_trend(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: small candle of either color with no upper shadow and long lower
    shadow.
    Trend: up.
    Prediction: reversal.
    """
    return T == -1 and cf.no_us(O, H, C) and cf.long_ls(O, L, C) and cf.short_body(O, C)


@numba.jit
def high_wave_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: small candle with extremely long shadows
    Trend: either.
    Prediction: either.
    """
    return cf.exlong_us(O, H, C) and cf.exlong_ls(O, L, C) and cf.short_body(O, C)


@numba.jit
def high_wave_down_trend(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: small candle with extremely long shadows
    Trend: either.
    Prediction: either.
    """
    return (
        T == -1
        and cf.exlong_us(O, H, C)
        and cf.exlong_ls(O, L, C)
        and cf.short_body(O, C)
    )


@numba.jit
def high_wave_up_trend(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: small candle with extremely long shadows
    Trend: either.
    Prediction: either.
    """
    return (
        T == 1
        and cf.exlong_us(O, H, C)
        and cf.exlong_ls(O, L, C)
        and cf.short_body(O, C)
    )


@numba.jit
def marubozu_black_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
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
def marubozu_closing_black_(
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
def marubozu_closing_white_(
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
def marubozu_opening_black_(
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
def marubozu_opening_white_(
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
def marubozu_white_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
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


@numba.jit
def shooting_star_one_candle_(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: small candle of either color with long upper shadow at least twice
    the height of the body and no lower shadow.
    Trend: up.
    Prediction: reversal.
    """
    return (
        T == 1
        and cf.long_us(O, H, C)
        and (cf.upper_shadow(O, H, C) > 2 * cf.hb(O, C))
        and cf.short_body(O, C)
        and cf.no_ls(O, L, C)
    )


@numba.jit
def shooting_star_one_candle_no_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: small candle of either color with long upper shadow at least twice
    the height of the body and no lower shadow.
    Trend: up.
    Prediction: reversal.
    """
    return (
        cf.long_us(O, H, C)
        and (cf.upper_shadow(O, H, C) > 2 * cf.hb(O, C))
        and cf.short_body(O, C)
        and cf.no_ls(O, L, C)
    )


@numba.jit
def shooting_star_one_candle_opp_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: small candle of either color with long upper shadow at least twice
    the height of the body and no lower shadow.
    Trend: up.
    Prediction: reversal.
    """
    return (
        T == -1
        and cf.long_us(O, H, C)
        and (cf.upper_shadow(O, H, C) > 2 * cf.hb(O, C))
        and cf.short_body(O, C)
        and cf.no_ls(O, L, C)
    )


@numba.jit
def spinning_top_black_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: small black candle with shadows longer than the body.
    Trend: either.
    Prediction: either.
    """
    return (
        cf.short_black_body(O, C)
        and (cf.upper_shadow(O, H, C) > cf.hb(O, C))
        and (cf.lower_shadow(O, L, C) > cf.hb(O, C))
        and not cf.no_ls(O, L, C)
        and not cf.no_us(O, H, C)
    )


@numba.jit
def spinning_top_black_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: small black candle with shadows longer than the body.
    Trend: either.
    Prediction: either.
    """
    return (
        T == -1
        and cf.short_black_body(O, C)
        and (cf.upper_shadow(O, H, C) > cf.hb(O, C))
        and (cf.lower_shadow(O, L, C) > cf.hb(O, C))
        and not cf.no_ls(O, L, C)
        and not cf.no_us(O, H, C)
    )


@numba.jit
def spinning_top_black_up_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: small black candle with shadows longer than the body.
    Trend: either.
    Prediction: either.
    """
    return (
        T == 1
        and cf.short_black_body(O, C)
        and (cf.upper_shadow(O, H, C) > cf.hb(O, C))
        and (cf.lower_shadow(O, L, C) > cf.hb(O, C))
        and not cf.no_ls(O, L, C)
        and not cf.no_us(O, H, C)
    )


@numba.jit
def spinning_top_white_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: small white candle with shadows longer than the body.
    Trend: either.
    Prediction: either.
    """
    return (
        cf.short_white_body(O, C)
        and (cf.upper_shadow(O, H, C) > cf.hb(O, C))
        and (cf.lower_shadow(O, L, C) > cf.hb(O, C))
        and not cf.no_ls(O, L, C)
        and not cf.no_us(O, H, C)
    )


@numba.jit
def spinning_top_white_down_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: small white candle with shadows longer than the body.
    Trend: either.
    Prediction: either.
    """
    return (
        T == -1
        and cf.short_white_body(O, C)
        and (cf.upper_shadow(O, H, C) > cf.hb(O, C))
        and (cf.lower_shadow(O, L, C) > cf.hb(O, C))
        and not cf.no_ls(O, L, C)
        and not cf.no_us(O, H, C)
    )


@numba.jit
def spinning_top_white_up_trend(
    O: list, H: list, L: list, C: list, V: list, T: list
) -> bool:
    """Definition: small white candle with shadows longer than the body.
    Trend: either.
    Prediction: either.
    """
    return (
        T == 1
        and cf.short_white_body(O, C)
        and (cf.upper_shadow(O, H, C) > cf.hb(O, C))
        and (cf.lower_shadow(O, L, C) > cf.hb(O, C))
        and not cf.no_ls(O, L, C)
        and not cf.no_us(O, H, C)
    )


@numba.jit
def takuri_line_(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: small  candle of either color with no upper shadow and lower shadow
    at least three times the length of the body.
    Trend: down.
    Prediction: reversal.
    """
    return (
        T == -1
        and cf.short_body(O, C)
        and cf.no_us(O, H, C)
        and (cf.lower_shadow(O, L, C) > 3 * cf.hb(O, C))
    )


@numba.jit
def takuri_line_no_trend(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: small  candle of either color with no upper shadow and lower shadow
    at least three times the length of the body.
    Trend: down.
    Prediction: reversal.
    """
    return (
        cf.short_body(O, C)
        and cf.no_us(O, H, C)
        and (cf.lower_shadow(O, L, C) > 3 * cf.hb(O, C))
    )


@numba.jit
def takuri_line_opp_trend(O: list, H: list, L: list, C: list, V: list, T: list) -> bool:
    """Definition: small  candle of either color with no upper shadow and lower shadow
    at least three times the length of the body.
    Trend: down.
    Prediction: reversal.
    """
    return (
        T == 1
        and cf.short_body(O, C)
        and cf.no_us(O, H, C)
        and (cf.lower_shadow(O, L, C) > 3 * cf.hb(O, C))
    )
