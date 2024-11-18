import candlestick_functions as cf
import numpy as np


def belt_hold_bearish_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle in an uptrend with no upper shadow that closes near
    the low.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.tall_black_body(O, C), cf.no_us(O, H, C), cf.mod_near(L, C))
    )


def belt_hold_bearish_no_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle in an uptrend with no upper shadow that closes near
    the low.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (cf.tall_black_body(O, C), cf.no_us(O, H, C), cf.mod_near(L, C))
    )


def belt_hold_bearish_opp_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle in an uptrend with no upper shadow that closes near
    the low.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == -1, cf.tall_black_body(O, C), cf.no_us(O, H, C), cf.mod_near(L, C))
    )


def belt_hold_bullish_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle in a downtrend with no lower shadow that closes
    near the high.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == -1, cf.tall_white_body(O, C), cf.no_ls(O, L, C), cf.mod_near(H, C))
    )


def belt_hold_bullish_no_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle in a downtrend with no lower shadow that closes
    near the high.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (cf.tall_white_body(O, C), cf.no_ls(O, L, C), cf.mod_near(H, C))
    )


def belt_hold_bullish_opp_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle in a downtrend with no lower shadow that closes
    near the high.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.tall_white_body(O, C), cf.no_ls(O, L, C), cf.mod_near(H, C))
    )


def candle_black_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.normal_black_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.upper_shadow(O, H, C) < cf.hb(O, C),
            cf.lower_shadow(O, L, C) < cf.hb(O, C),
        )
    )


def candle_black_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.normal_black_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.upper_shadow(O, H, C) < cf.hb(O, C),
            cf.lower_shadow(O, L, C) < cf.hb(O, C),
        )
    )


def candle_black_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.normal_black_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.upper_shadow(O, H, C) < cf.hb(O, C),
            cf.lower_shadow(O, L, C) < cf.hb(O, C),
        )
    )


def candle_short_black_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.short_black_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.upper_shadow(O, H, C) < cf.hb(O, C),
            cf.lower_shadow(O, L, C) < cf.hb(O, C),
        )
    )


def candle_short_black_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.short_black_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.upper_shadow(O, H, C) < cf.hb(O, C),
            cf.lower_shadow(O, L, C) < cf.hb(O, C),
        )
    )


def candle_short_black_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.short_black_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.upper_shadow(O, H, C) < cf.hb(O, C),
            cf.lower_shadow(O, L, C) < cf.hb(O, C),
        )
    )


def candle_short_white_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.short_white_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.upper_shadow(O, H, C) < cf.hb(O, C),
            cf.lower_shadow(O, L, C) < cf.hb(O, C),
        )
    )


def candle_short_white_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.short_white_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.upper_shadow(O, H, C) < cf.hb(O, C),
            cf.lower_shadow(O, L, C) < cf.hb(O, C),
        )
    )


def candle_short_white_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.short_white_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.upper_shadow(O, H, C) < cf.hb(O, C),
            cf.lower_shadow(O, L, C) < cf.hb(O, C),
        )
    )


def candle_white_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.normal_white_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.upper_shadow(O, H, C) < cf.hb(O, C),
            cf.lower_shadow(O, L, C) < cf.hb(O, C),
        )
    )


def candle_white_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.normal_white_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.upper_shadow(O, H, C) < cf.hb(O, C),
            cf.lower_shadow(O, L, C) < cf.hb(O, C),
        )
    )


def candle_white_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.normal_white_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.upper_shadow(O, H, C) < cf.hb(O, C),
            cf.lower_shadow(O, L, C) < cf.hb(O, C),
        )
    )


def doji_dragonfly_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with long lower shadow , small upper shadow.

    Trend: either.

    Prediction: reversal if down.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (cf.doji(O, C), cf.small_us(O, H, C), cf.long_ls(O, L, C))
    )


def doji_dragonfly_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with long lower shadow , small upper shadow.

    Trend: either.

    Prediction: reversal if down.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == -1, cf.doji(O, C), cf.small_us(O, H, C), cf.long_ls(O, L, C))
    )


def doji_dragonfly_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with long lower shadow , small upper shadow.

    Trend: either.

    Prediction: reversal if down.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.doji(O, C), cf.small_us(O, H, C), cf.long_ls(O, L, C))
    )


def doji_gravestone_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with small lower shadow , long upper shadow.

    Trend: either.

    Prediction: reversal if up.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (cf.doji(O, C), cf.no_ls(O, L, C), cf.long_us(O, H, C))
    )


def doji_gravestone_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with small lower shadow , long upper shadow.

    Trend: either.

    Prediction: reversal if up.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == -1, cf.doji(O, C), cf.no_ls(O, L, C), cf.long_us(O, H, C))
    )


def doji_gravestone_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with small lower shadow , long upper shadow.

    Trend: either.

    Prediction: reversal if up.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.doji(O, C), cf.no_ls(O, L, C), cf.long_us(O, H, C))
    )


def doji_long_legged_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with long shadows.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (cf.doji(O, C), cf.long_us(O, H, C), cf.long_ls(O, L, C))
    )


def doji_long_legged_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with long shadows.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == -1, cf.doji(O, C), cf.long_us(O, H, C), cf.long_ls(O, L, C))
    )


def doji_long_legged_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with long shadows.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.doji(O, C), cf.long_us(O, H, C), cf.long_ls(O, L, C))
    )


def doji_nothern_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji in an uptrend.

    Trend: up.

    Prediction: reversal.
    """
    O, C = candle[:, 0], candle[:, 3]
    return np.logical_and.reduce((T == 1, cf.doji(O, C)))


def doji_nothern_no_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji in an uptrend.

    Trend: up.

    Prediction: reversal.
    """
    O, C = candle[:, 0], candle[:, 3]
    return cf.doji(O, C)


def doji_nothern_opp_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji in an uptrend.

    Trend: up.

    Prediction: reversal.
    """
    O, C = candle[:, 0], candle[:, 3]
    return np.logical_and.reduce((T == -1, cf.doji(O, C)))


def doji_southern_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji in a downtrend.

    Trend: down.

    Prediction: reversal.
    """
    O, C = candle[:, 0], candle[:, 3]
    return np.logical_and.reduce((T == -1, cf.doji(O, C)))


def doji_southern_no_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji in a downtrend.

    Trend: down.

    Prediction: reversal.
    """
    O, C = candle[:, 0], candle[:, 3]
    return cf.doji(O, C)


def doji_southern_opp_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji in a downtrend.

    Trend: down.

    Prediction: reversal.
    """
    O, C = candle[:, 0], candle[:, 3]
    return np.logical_and.reduce((T == 1, cf.doji(O, C)))


def hammer_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: candle with a small body of either color, lower shadow between 2, 3
    times the length of the body, and a small or no upper shadow.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.short_body(O, C),
            np.logical_not(cf.no_ls(O, L, C)),
            (2 * cf.hb(O, C) < cf.lower_shadow(O, L, C)),
            (cf.lower_shadow(O, L, C) < 3 * cf.hb(O, C)),
            np.logical_or(cf.small_us(O, H, C), cf.no_us(O, H, C)),
        )
    )


def hammer_no_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: candle with a small body of either color, lower shadow between 2, 3
    times the length of the body, and a small or no upper shadow.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.short_body(O, C),
            np.logical_not(cf.no_ls(O, L, C)),
            (2 * cf.hb(O, C) < cf.lower_shadow(O, L, C)),
            (cf.lower_shadow(O, L, C) < 3 * cf.hb(O, C)),
            np.logical_or(cf.small_us(O, H, C), cf.no_us(O, H, C)),
        )
    )


def hammer_opp_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: candle with a small body of either color, lower shadow between 2, 3
    times the length of the body, and a small or no upper shadow.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.short_body(O, C),
            np.logical_not(cf.no_ls(O, L, C)),
            (2 * cf.hb(O, C) < cf.lower_shadow(O, L, C)),
            (cf.lower_shadow(O, L, C) < 3 * cf.hb(O, C)),
            np.logical_or(cf.small_us(O, H, C), cf.no_us(O, H, C)),
        )
    )


def hanging_man_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small candle of either color with no upper shadow , long lower
    shadow.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.no_us(O, H, C), cf.long_ls(O, L, C), cf.short_body(O, C))
    )


def hanging_man_no_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small candle of either color with no upper shadow , long lower
    shadow.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (cf.no_us(O, H, C), cf.long_ls(O, L, C), cf.short_body(O, C))
    )


def hanging_man_opp_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small candle of either color with no upper shadow , long lower
    shadow.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == -1, cf.no_us(O, H, C), cf.long_ls(O, L, C), cf.short_body(O, C))
    )


def high_wave_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small candle with extremely long shadows.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (cf.exlong_us(O, H, C), cf.exlong_ls(O, L, C), cf.short_body(O, C))
    )


def high_wave_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small candle with extremely long shadows.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == -1, cf.exlong_us(O, H, C), cf.exlong_ls(O, L, C), cf.short_body(O, C))
    )


def high_wave_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small candle with extremely long shadows.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.exlong_us(O, H, C), cf.exlong_ls(O, L, C), cf.short_body(O, C))
    )


def marubozu_black_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle without shadows.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (cf.tall_black_body(O, C), cf.no_us(O, H, C), cf.no_ls(O, L, C))
    )


def marubozu_black_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle without shadows.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == -1, cf.tall_black_body(O, C), cf.no_us(O, H, C), cf.no_ls(O, L, C))
    )


def marubozu_black_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle without shadows.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.tall_black_body(O, C), cf.no_us(O, H, C), cf.no_ls(O, L, C))
    )


def marubozu_closing_black_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle without lower shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            cf.no_ls(O, L, C),
        )
    )


def marubozu_closing_black_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle without lower shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            cf.no_ls(O, L, C),
        )
    )


def marubozu_closing_black_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle without lower shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            cf.no_ls(O, L, C),
        )
    )


def marubozu_closing_white_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle without upper shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O, C),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.no_us(O, H, C),
        )
    )


def marubozu_closing_white_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle without upper shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O, C),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.no_us(O, H, C),
        )
    )


def marubozu_closing_white_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle without upper shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O, C),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.no_us(O, H, C),
        )
    )


def marubozu_opening_black_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle without upper shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O, C),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.no_us(O, H, C),
        )
    )


def marubozu_opening_black_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle without upper shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O, C),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.no_us(O, H, C),
        )
    )


def marubozu_opening_black_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle without upper shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O, C),
            np.logical_not(cf.no_ls(O, L, C)),
            cf.no_us(O, H, C),
        )
    )


def marubozu_opening_white_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle without lower shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            cf.no_ls(O, L, C),
        )
    )


def marubozu_opening_white_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle without lower shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            cf.no_ls(O, L, C),
        )
    )


def marubozu_opening_white_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle without lower shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O, C),
            np.logical_not(cf.no_us(O, H, C)),
            cf.no_ls(O, L, C),
        )
    )


def marubozu_white_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle without shadows.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (cf.tall_white_body(O, C), cf.no_us(O, H, C), cf.no_ls(O, L, C))
    )


def marubozu_white_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle without shadows.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == -1, cf.tall_white_body(O, C), cf.no_us(O, H, C), cf.no_ls(O, L, C))
    )


def marubozu_white_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle without shadows.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.tall_white_body(O, C), cf.no_us(O, H, C), cf.no_ls(O, L, C))
    )


def rickshaw_man_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with midpoint of the body near the midpoint of the shadows,
    those shadows being exceedingly long.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.doji(O, C),
            cf.exlong_us(O, H, C),
            cf.exlong_ls(O, L, C),
            cf.near(0.5 * (O + C), 0.5 * (H + L)),
        )
    )


def rickshaw_man_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with midpoint of the body near the midpoint of the shadows,
    those shadows being exceedingly long.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.doji(O, C),
            cf.exlong_us(O, H, C),
            cf.exlong_ls(O, L, C),
            cf.near(0.5 * (O + C), 0.5 * (H + L)),
        )
    )


def rickshaw_man_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with midpoint of the body near the midpoint of the shadows,
    those shadows being exceedingly long.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.doji(O, C),
            cf.exlong_us(O, H, C),
            cf.exlong_ls(O, L, C),
            cf.near(0.5 * (O + C), 0.5 * (H + L)),
        )
    )


def shooting_star_one_candle_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small candle of either color with long upper shadow at least twice
    the height of the body, no lower shadow.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.long_us(O, H, C),
            (cf.upper_shadow(O, H, C) > 2 * cf.hb(O, C)),
            cf.short_body(O, C),
            cf.no_ls(O, L, C),
        )
    )


def shooting_star_one_candle_no_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small candle of either color with long upper shadow at least twice
    the height of the body, no lower shadow.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.long_us(O, H, C),
            (cf.upper_shadow(O, H, C) > 2 * cf.hb(O, C)),
            cf.short_body(O, C),
            cf.no_ls(O, L, C),
        )
    )


def shooting_star_one_candle_opp_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small candle of either color with long upper shadow at least twice
    the height of the body, no lower shadow.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.long_us(O, H, C),
            (cf.upper_shadow(O, H, C) > 2 * cf.hb(O, C)),
            cf.short_body(O, C),
            cf.no_ls(O, L, C),
        )
    )


def spinning_top_black_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small black candle with shadows longer than the body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.short_black_body(O, C),
            (cf.upper_shadow(O, H, C) > cf.hb(O, C)),
            (cf.lower_shadow(O, L, C) > cf.hb(O, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            np.logical_not(cf.no_us(O, H, C)),
        )
    )


def spinning_top_black_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small black candle with shadows longer than the body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.short_black_body(O, C),
            (cf.upper_shadow(O, H, C) > cf.hb(O, C)),
            (cf.lower_shadow(O, L, C) > cf.hb(O, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            np.logical_not(cf.no_us(O, H, C)),
        )
    )


def spinning_top_black_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small black candle with shadows longer than the body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.short_black_body(O, C),
            (cf.upper_shadow(O, H, C) > cf.hb(O, C)),
            (cf.lower_shadow(O, L, C) > cf.hb(O, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            np.logical_not(cf.no_us(O, H, C)),
        )
    )


def spinning_top_white_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small white candle with shadows longer than the body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.short_white_body(O, C),
            (cf.upper_shadow(O, H, C) > cf.hb(O, C)),
            (cf.lower_shadow(O, L, C) > cf.hb(O, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            np.logical_not(cf.no_us(O, H, C)),
        )
    )


def spinning_top_white_down_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small white candle with shadows longer than the body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.short_white_body(O, C),
            (cf.upper_shadow(O, H, C) > cf.hb(O, C)),
            (cf.lower_shadow(O, L, C) > cf.hb(O, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            np.logical_not(cf.no_us(O, H, C)),
        )
    )


def spinning_top_white_up_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small white candle with shadows longer than the body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.short_white_body(O, C),
            (cf.upper_shadow(O, H, C) > cf.hb(O, C)),
            (cf.lower_shadow(O, L, C) > cf.hb(O, C)),
            np.logical_not(cf.no_ls(O, L, C)),
            np.logical_not(cf.no_us(O, H, C)),
        )
    )


def takuri_line_(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small  candle of either color with no upper shadow, lower shadow at
    least three times the length of the body.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.short_body(O, C),
            cf.no_us(O, H, C),
            (cf.lower_shadow(O, L, C) > 3 * cf.hb(O, C)),
        )
    )


def takuri_line_no_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small  candle of either color with no upper shadow, lower shadow at
    least three times the length of the body.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.short_body(O, C),
            cf.no_us(O, H, C),
            (cf.lower_shadow(O, L, C) > 3 * cf.hb(O, C)),
        )
    )


def takuri_line_opp_trend(candle: np.ndarray, T: np.ndarray) -> bool:
    """Definition: small  candle of either color with no upper shadow, lower shadow at
    least three times the length of the body.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.short_body(O, C),
            cf.no_us(O, H, C),
            (cf.lower_shadow(O, L, C) > 3 * cf.hb(O, C)),
        )
    )
