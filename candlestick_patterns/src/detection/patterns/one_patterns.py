import numpy as np

from detection.patterns.functions import candlestick_functions as cf


def belt_hold_bearish_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: tall black candle in an uptrend with no upper shadow that closes near
    the low.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O, C, percentile),
            cf.no_us(O, H, C, percentile),
            cf.small_ls(O, L, C, percentile),
        )
    )


def belt_hold_bearish_no_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall black candle in an uptrend with no upper shadow that closes near
    the low.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O, C, percentile),
            cf.no_us(O, H, C, percentile),
            cf.small_ls(O, L, C, percentile),
        )
    )


def belt_hold_bearish_opp_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall black candle in an uptrend with no upper shadow that closes near
    the low.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O, C, percentile),
            cf.no_us(O, H, C, percentile),
            cf.small_ls(O, L, C, percentile),
        )
    )


def belt_hold_bullish_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: tall white candle in a downtrend with no lower shadow that closes
    near the high.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O, C, percentile),
            cf.no_ls(O, L, C, percentile),
            cf.small_us(O, H, C, percentile),
        )
    )


def belt_hold_bullish_no_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall white candle in a downtrend with no lower shadow that closes
    near the high.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O, C, percentile),
            cf.no_ls(O, L, C, percentile),
            cf.small_us(O, H, C, percentile),
        )
    )


def belt_hold_bullish_opp_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall white candle in a downtrend with no lower shadow that closes
    near the high.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O, C, percentile),
            cf.no_ls(O, L, C, percentile),
            cf.small_us(O, H, C, percentile),
        )
    )


def candle_black_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.normal_black_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.upper_shadow_length(O, H, C) < cf.body_height(O, C),
            cf.lower_shadow_length(O, L, C) < cf.body_height(O, C),
        )
    )


def candle_black_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.normal_black_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.upper_shadow_length(O, H, C) < cf.body_height(O, C),
            cf.lower_shadow_length(O, L, C) < cf.body_height(O, C),
        )
    )


def candle_black_up_trend(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.normal_black_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.upper_shadow_length(O, H, C) < cf.body_height(O, C),
            cf.lower_shadow_length(O, L, C) < cf.body_height(O, C),
        )
    )


def candle_short_black_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.short_black_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.upper_shadow_length(O, H, C) < cf.body_height(O, C),
            cf.lower_shadow_length(O, L, C) < cf.body_height(O, C),
        )
    )


def candle_short_black_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.short_black_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.upper_shadow_length(O, H, C) < cf.body_height(O, C),
            cf.lower_shadow_length(O, L, C) < cf.body_height(O, C),
        )
    )


def candle_short_black_up_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: normal black candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.short_black_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.upper_shadow_length(O, H, C) < cf.body_height(O, C),
            cf.lower_shadow_length(O, L, C) < cf.body_height(O, C),
        )
    )


def candle_short_white_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.short_white_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.upper_shadow_length(O, H, C) < cf.body_height(O, C),
            cf.lower_shadow_length(O, L, C) < cf.body_height(O, C),
        )
    )


def candle_short_white_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.short_white_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.upper_shadow_length(O, H, C) < cf.body_height(O, C),
            cf.lower_shadow_length(O, L, C) < cf.body_height(O, C),
        )
    )


def candle_short_white_up_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.short_white_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.upper_shadow_length(O, H, C) < cf.body_height(O, C),
            cf.lower_shadow_length(O, L, C) < cf.body_height(O, C),
        )
    )


def candle_white_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.normal_white_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.upper_shadow_length(O, H, C) < cf.body_height(O, C),
            cf.lower_shadow_length(O, L, C) < cf.body_height(O, C),
        )
    )


def candle_white_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.normal_white_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.upper_shadow_length(O, H, C) < cf.body_height(O, C),
            cf.lower_shadow_length(O, L, C) < cf.body_height(O, C),
        )
    )


def candle_white_up_trend(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: normal white candle with shadows that do not exceed the length of the
    body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.normal_white_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.upper_shadow_length(O, H, C) < cf.body_height(O, C),
            cf.lower_shadow_length(O, L, C) < cf.body_height(O, C),
        )
    )


def doji_dragonfly_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: doji with long lower shadow , small upper shadow.

    Trend: either.

    Prediction: reversal if down.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.doji(O, C, percentile),
            cf.small_us(O, H, C, percentile),
            cf.long_ls(O, L, C, percentile),
        )
    )


def doji_dragonfly_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: doji with long lower shadow , small upper shadow.

    Trend: either.

    Prediction: reversal if down.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.doji(O, C, percentile),
            cf.small_us(O, H, C, percentile),
            cf.long_ls(O, L, C, percentile),
        )
    )


def doji_dragonfly_up_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: doji with long lower shadow , small upper shadow.

    Trend: either.

    Prediction: reversal if down.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.doji(O, C, percentile),
            cf.small_us(O, H, C, percentile),
            cf.long_ls(O, L, C, percentile),
        )
    )


def doji_gravestone_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: doji with small lower shadow , long upper shadow.

    Trend: either.

    Prediction: reversal if up.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.doji(O, C, percentile),
            cf.no_ls(O, L, C, percentile),
            cf.long_us(O, H, C, percentile),
        )
    )


def doji_gravestone_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: doji with small lower shadow , long upper shadow.

    Trend: either.

    Prediction: reversal if up.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.doji(O, C, percentile),
            cf.no_ls(O, L, C, percentile),
            cf.long_us(O, H, C, percentile),
        )
    )


def doji_gravestone_up_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: doji with small lower shadow , long upper shadow.

    Trend: either.

    Prediction: reversal if up.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.doji(O, C, percentile),
            cf.no_ls(O, L, C, percentile),
            cf.long_us(O, H, C, percentile),
        )
    )


def doji_long_legged_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: doji with long shadows.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.doji(O, C, percentile),
            cf.long_us(O, H, C, percentile),
            cf.long_ls(O, L, C, percentile),
        )
    )


def doji_long_legged_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: doji with long shadows.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.doji(O, C, percentile),
            cf.long_us(O, H, C, percentile),
            cf.long_ls(O, L, C, percentile),
        )
    )


def doji_long_legged_up_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: doji with long shadows.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.doji(O, C, percentile),
            cf.long_us(O, H, C, percentile),
            cf.long_ls(O, L, C, percentile),
        )
    )


def doji_northern_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: doji in an uptrend.

    Trend: up.

    Prediction: reversal.
    """
    O, C = candle[:, 0], candle[:, 3]
    return np.logical_and.reduce((T == 1, cf.doji(O, C, percentile)))


def doji_northern_no_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: doji in an uptrend.

    Trend: up.

    Prediction: reversal.
    """
    O, C = candle[:, 0], candle[:, 3]
    return cf.doji(O, C, percentile)


def doji_northern_opp_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: doji in an uptrend.

    Trend: up.

    Prediction: reversal.
    """
    O, C = candle[:, 0], candle[:, 3]
    return np.logical_and.reduce((T == -1, cf.doji(O, C, percentile)))


def doji_southern_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: doji in a downtrend.

    Trend: down.

    Prediction: reversal.
    """
    O, C = candle[:, 0], candle[:, 3]
    return np.logical_and.reduce((T == -1, cf.doji(O, C, percentile)))


def doji_southern_no_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: doji in a downtrend.

    Trend: down.

    Prediction: reversal.
    """
    O, C = candle[:, 0], candle[:, 3]
    return cf.doji(O, C, percentile)


def doji_southern_opp_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: doji in a downtrend.

    Trend: down.

    Prediction: reversal.
    """
    O, C = candle[:, 0], candle[:, 3]
    return np.logical_and.reduce((T == 1, cf.doji(O, C, percentile)))


def hammer_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: candle with a small body of either color, lower shadow between 2, 3
    times the length of the body, and a small or no upper shadow.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.short_body(O, C, percentile),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            (2 * cf.body_height(O, C) < cf.lower_shadow_length(O, L, C)),
            (cf.lower_shadow_length(O, L, C) < 3 * cf.body_height(O, C)),
            np.logical_or(
                cf.small_us(O, H, C, percentile), cf.no_us(O, H, C, percentile)
            ),
        )
    )


def hammer_no_trend(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: candle with a small body of either color, lower shadow between 2, 3
    times the length of the body, and a small or no upper shadow.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.short_body(O, C, percentile),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            (2 * cf.body_height(O, C) < cf.lower_shadow_length(O, L, C)),
            (cf.lower_shadow_length(O, L, C) < 3 * cf.body_height(O, C)),
            np.logical_or(
                cf.small_us(O, H, C, percentile), cf.no_us(O, H, C, percentile)
            ),
        )
    )


def hammer_opp_trend(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: candle with a small body of either color, lower shadow between 2, 3
    times the length of the body, and a small or no upper shadow.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.short_body(O, C, percentile),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            (2 * cf.body_height(O, C) < cf.lower_shadow_length(O, L, C)),
            (cf.lower_shadow_length(O, L, C) < 3 * cf.body_height(O, C)),
            np.logical_or(
                cf.small_us(O, H, C, percentile), cf.no_us(O, H, C, percentile)
            ),
        )
    )


def hanging_man_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: small candle of either color with no upper shadow , long lower
    shadow.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.no_us(O, H, C, percentile),
            cf.long_ls(O, L, C, percentile),
            cf.short_body(O, C, percentile),
        )
    )


def hanging_man_no_trend(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: small candle of either color with no upper shadow , long lower
    shadow.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.no_us(O, H, C, percentile),
            cf.long_ls(O, L, C, percentile),
            cf.short_body(O, C, percentile),
        )
    )


def hanging_man_opp_trend(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: small candle of either color with no upper shadow , long lower
    shadow.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.no_us(O, H, C, percentile),
            cf.long_ls(O, L, C, percentile),
            cf.short_body(O, C, percentile),
        )
    )


def high_wave_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: small candle with extremely long shadows.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.exlong_us(O, H, C, percentile),
            cf.exlong_ls(O, L, C, percentile),
            cf.short_body(O, C, percentile),
        )
    )


def high_wave_down_trend(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: small candle with extremely long shadows.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.exlong_us(O, H, C, percentile),
            cf.exlong_ls(O, L, C, percentile),
            cf.short_body(O, C, percentile),
        )
    )


def high_wave_up_trend(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: small candle with extremely long shadows.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.exlong_us(O, H, C, percentile),
            cf.exlong_ls(O, L, C, percentile),
            cf.short_body(O, C, percentile),
        )
    )


def marubozu_black_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: tall black candle without shadows.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O, C, percentile),
            cf.no_us(O, H, C, percentile),
            cf.no_ls(O, L, C, percentile),
        )
    )


def marubozu_black_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall black candle without shadows.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O, C, percentile),
            cf.no_us(O, H, C, percentile),
            cf.no_ls(O, L, C, percentile),
        )
    )


def marubozu_black_up_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall black candle without shadows.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O, C, percentile),
            cf.no_us(O, H, C, percentile),
            cf.no_ls(O, L, C, percentile),
        )
    )


def marubozu_closing_black_(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall black candle without lower shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            cf.no_ls(O, L, C, percentile),
        )
    )


def marubozu_closing_black_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall black candle without lower shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            cf.no_ls(O, L, C, percentile),
        )
    )


def marubozu_closing_black_up_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall black candle without lower shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            cf.no_ls(O, L, C, percentile),
        )
    )


def marubozu_closing_white_(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall white candle without upper shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O, C, percentile),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.no_us(O, H, C, percentile),
        )
    )


def marubozu_closing_white_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall white candle without upper shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O, C, percentile),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.no_us(O, H, C, percentile),
        )
    )


def marubozu_closing_white_up_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall white candle without upper shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O, C, percentile),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.no_us(O, H, C, percentile),
        )
    )


def marubozu_opening_black_(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall black candle without upper shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O, C, percentile),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.no_us(O, H, C, percentile),
        )
    )


def marubozu_opening_black_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall black candle without upper shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O, C, percentile),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.no_us(O, H, C, percentile),
        )
    )


def marubozu_opening_black_up_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall black candle without upper shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O, C, percentile),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            cf.no_us(O, H, C, percentile),
        )
    )


def marubozu_opening_white_(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall white candle without lower shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            cf.no_ls(O, L, C, percentile),
        )
    )


def marubozu_opening_white_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall white candle without lower shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            cf.no_ls(O, L, C, percentile),
        )
    )


def marubozu_opening_white_up_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall white candle without lower shadow.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O, C, percentile),
            np.logical_not(cf.no_us(O, H, C, percentile)),
            cf.no_ls(O, L, C, percentile),
        )
    )


def marubozu_white_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: tall white candle without shadows.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O, C, percentile),
            cf.no_us(O, H, C, percentile),
            cf.no_ls(O, L, C, percentile),
        )
    )


def marubozu_white_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall white candle without shadows.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O, C, percentile),
            cf.no_us(O, H, C, percentile),
            cf.no_ls(O, L, C, percentile),
        )
    )


def marubozu_white_up_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: tall white candle without shadows.

    Trend: either.

    Prediction: continuation.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O, C, percentile),
            cf.no_us(O, H, C, percentile),
            cf.no_ls(O, L, C, percentile),
        )
    )


def rickshaw_man_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: doji with midpoint of the body near the midpoint of the shadows,
    those shadows being exceedingly long.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.doji(O, C, percentile),
            cf.exlong_us(O, H, C, percentile),
            cf.exlong_ls(O, L, C, percentile),
            cf.near(0.5 * (O + C), 0.5 * (H + L), percentile),
        )
    )


def rickshaw_man_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: doji with midpoint of the body near the midpoint of the shadows,
    those shadows being exceedingly long.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.doji(O, C, percentile),
            cf.exlong_us(O, H, C, percentile),
            cf.exlong_ls(O, L, C, percentile),
            cf.near(0.5 * (O + C), 0.5 * (H + L), percentile),
        )
    )


def rickshaw_man_up_trend(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: doji with midpoint of the body near the midpoint of the shadows,
    those shadows being exceedingly long.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.doji(O, C, percentile),
            cf.exlong_us(O, H, C, percentile),
            cf.exlong_ls(O, L, C, percentile),
            cf.near(0.5 * (O + C), 0.5 * (H + L), percentile),
        )
    )


def shooting_star_one_candle_(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: small candle of either color with long upper shadow at least twice
    the height of the body, no lower shadow.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.long_us(O, H, C, percentile),
            (cf.upper_shadow_length(O, H, C) > 2 * cf.body_height(O, C)),
            cf.short_body(O, C, percentile),
            cf.no_ls(O, L, C, percentile),
        )
    )


def shooting_star_one_candle_no_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: small candle of either color with long upper shadow at least twice
    the height of the body, no lower shadow.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.long_us(O, H, C, percentile),
            (cf.upper_shadow_length(O, H, C) > 2 * cf.body_height(O, C)),
            cf.short_body(O, C, percentile),
            cf.no_ls(O, L, C, percentile),
        )
    )


def shooting_star_one_candle_opp_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: small candle of either color with long upper shadow at least twice
    the height of the body, no lower shadow.

    Trend: up.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.long_us(O, H, C, percentile),
            (cf.upper_shadow_length(O, H, C) > 2 * cf.body_height(O, C)),
            cf.short_body(O, C, percentile),
            cf.no_ls(O, L, C, percentile),
        )
    )


def spinning_top_black_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: small black candle with shadows longer than the body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.short_black_body(O, C, percentile),
            (cf.upper_shadow_length(O, H, C) > cf.body_height(O, C)),
            (cf.lower_shadow_length(O, L, C) > cf.body_height(O, C)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            np.logical_not(cf.no_us(O, H, C, percentile)),
        )
    )


def spinning_top_black_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: small black candle with shadows longer than the body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.short_black_body(O, C, percentile),
            (cf.upper_shadow_length(O, H, C) > cf.body_height(O, C)),
            (cf.lower_shadow_length(O, L, C) > cf.body_height(O, C)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            np.logical_not(cf.no_us(O, H, C, percentile)),
        )
    )


def spinning_top_black_up_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: small black candle with shadows longer than the body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.short_black_body(O, C, percentile),
            (cf.upper_shadow_length(O, H, C) > cf.body_height(O, C)),
            (cf.lower_shadow_length(O, L, C) > cf.body_height(O, C)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            np.logical_not(cf.no_us(O, H, C, percentile)),
        )
    )


def spinning_top_white_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: small white candle with shadows longer than the body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.short_white_body(O, C, percentile),
            (cf.upper_shadow_length(O, H, C) > cf.body_height(O, C)),
            (cf.lower_shadow_length(O, L, C) > cf.body_height(O, C)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            np.logical_not(cf.no_us(O, H, C, percentile)),
        )
    )


def spinning_top_white_down_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: small white candle with shadows longer than the body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.short_white_body(O, C, percentile),
            (cf.upper_shadow_length(O, H, C) > cf.body_height(O, C)),
            (cf.lower_shadow_length(O, L, C) > cf.body_height(O, C)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            np.logical_not(cf.no_us(O, H, C, percentile)),
        )
    )


def spinning_top_white_up_trend(
    candle: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: small white candle with shadows longer than the body.

    Trend: either.

    Prediction: either.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.short_white_body(O, C, percentile),
            (cf.upper_shadow_length(O, H, C) > cf.body_height(O, C)),
            (cf.lower_shadow_length(O, L, C) > cf.body_height(O, C)),
            np.logical_not(cf.no_ls(O, L, C, percentile)),
            np.logical_not(cf.no_us(O, H, C, percentile)),
        )
    )


def takuri_line_(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: small  candle of either color with no upper shadow, lower shadow at
    least three times the length of the body.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.short_body(O, C, percentile),
            cf.no_us(O, H, C, percentile),
            (cf.lower_shadow_length(O, L, C) > 3 * cf.body_height(O, C)),
        )
    )


def takuri_line_no_trend(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: small  candle of either color with no upper shadow, lower shadow at
    least three times the length of the body.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            cf.short_body(O, C, percentile),
            cf.no_us(O, H, C, percentile),
            (cf.lower_shadow_length(O, L, C) > 3 * cf.body_height(O, C)),
        )
    )


def takuri_line_opp_trend(candle: np.ndarray, T: np.ndarray, percentile: tuple) -> bool:
    """Definition: small  candle of either color with no upper shadow, lower shadow at
    least three times the length of the body.

    Trend: down.

    Prediction: reversal.
    """
    O, H, L, C = candle[:, 0], candle[:, 1], candle[:, 2], candle[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.short_body(O, C, percentile),
            cf.no_us(O, H, C, percentile),
            (cf.lower_shadow_length(O, L, C) > 3 * cf.body_height(O, C)),
        )
    )
