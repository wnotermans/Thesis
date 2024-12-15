import numpy as np

from detection.patterns.functions import candlestick_functions as cf


def above_the_stomach_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: first a black candle, then a white candle opening and closing at or
    above the midpoint of the previous candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            O_2 >= 0.5 * (O_1 + C_1),
            C_2 >= 0.5 * (O_1 + C_1),
        )
    )


def above_the_stomach_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: first a black candle, then a white candle opening and closing at or
    above the midpoint of the previous candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            O_2 >= 0.5 * (O_1 + C_1),
            C_2 >= 0.5 * (O_1 + C_1),
        )
    )


def above_the_stomach_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: first a black candle, then a white candle opening and closing at or
    above the midpoint of the previous candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            O_2 >= 0.5 * (O_1 + C_1),
            C_2 >= 0.5 * (O_1 + C_1),
        )
    )


def below_the_stomach_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a candle that opens below the midpoint
    of the first candles body and closes at or below that midpoint.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            0.5 * (O_1 + C_1) >= O_2,
            C_2 <= 0.5 * (O_1 + C_1),
        )
    )


def below_the_stomach_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a candle that opens below the midpoint
    of the first candles body and closes at or below that midpoint.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            0.5 * (O_1 + C_1) >= O_2,
            C_2 <= 0.5 * (O_1 + C_1),
        )
    )


def below_the_stomach_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a candle that opens below the midpoint
    of the first candles body and closes at or below that midpoint.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            0.5 * (O_1 + C_1) >= O_2,
            C_2 <= 0.5 * (O_1 + C_1),
        )
    )


def dark_cloud_cover_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a black candle that opens above the
    high of the first candle and closes at or below the midpoint.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            O_2 > H_1,
            C_2 >= 0.5 * (O_1 + C_1),
        )
    )


def dark_cloud_cover_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a black candle that opens above the
    high of the first candle and closes at or below the midpoint.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            O_2 > H_1,
            C_2 >= 0.5 * (O_1 + C_1),
        )
    )


def dark_cloud_cover_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a black candle that opens above the
    high of the first candle and closes at or below the midpoint.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            O_2 > H_1,
            C_2 >= 0.5 * (O_1 + C_1),
        )
    )


def doji_gapping_down_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with a downwards shadow gap.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (T == -1, cf.doji(O_2, C_2), cf.down_shadow_gap(L_1, H_2))
    )


def doji_gapping_down_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with a downwards shadow gap.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce((cf.doji(O_2, C_2), cf.down_shadow_gap(L_1, H_2)))


def doji_gapping_down_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with a downwards shadow gap.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.doji(O_2, C_2), cf.down_shadow_gap(L_1, H_2))
    )


def doji_gapping_up_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with an upwards shadow gap.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.doji(O_2, C_2), cf.up_shadow_gap(H_1, L_2))
    )


def doji_gapping_up_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with an upwards shadow gap.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce((cf.doji(O_2, C_2), cf.up_shadow_gap(H_1, L_2)))


def doji_gapping_up_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: doji with an upwards shadow gap.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (T == -1, cf.doji(O_2, C_2), cf.up_shadow_gap(H_1, L_2))
    )


def doji_star_bearish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: long white candle followed by a doji with an upwards body gap, the
    doji cannot have extremely long shadows and the total height of the shadows cannot
    exceed the body length of the first candle.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.doji(O_2, C_2),
            np.logical_not(cf.exlong_ls(O_2, L_2, C_2)),
            np.logical_not(cf.exlong_us(O_2, L_2, C_2)),
            cf.total_shadow_length(O_2, H_2, L_2, C_2) < cf.body_height(O_1, C_1),
        )
    )


def doji_star_bearish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: long white candle followed by a doji with an upwards body gap, the
    doji cannot have extremely long shadows and the total height of the shadows cannot
    exceed the body length of the first candle.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.doji(O_2, C_2),
            np.logical_not(cf.exlong_ls(O_2, L_2, C_2)),
            np.logical_not(cf.exlong_us(O_2, L_2, C_2)),
            cf.total_shadow_length(O_2, H_2, L_2, C_2) < cf.body_height(O_1, C_1),
        )
    )


def doji_star_bearish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: long white candle followed by a doji with an upwards body gap, the
    doji cannot have extremely long shadows and the total height of the shadows cannot
    exceed the body length of the first candle.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.doji(O_2, C_2),
            np.logical_not(cf.exlong_ls(O_2, L_2, C_2)),
            np.logical_not(cf.exlong_us(O_2, L_2, C_2)),
            cf.total_shadow_length(O_2, H_2, L_2, C_2) < cf.body_height(O_1, C_1),
        )
    )


def doji_star_bullish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: long black candle followed by a doji with a downward body gap, the
    doji cannot have extremely long shadows and the total height of the shadows cannot
    exceed the body length of the first candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.doji(O_2, C_2),
            np.logical_not(cf.exlong_ls(O_2, L_2, C_2)),
            np.logical_not(cf.exlong_us(O_2, L_2, C_2)),
        )
    )


def doji_star_bullish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: long black candle followed by a doji with a downward body gap, the
    doji cannot have extremely long shadows and the total height of the shadows cannot
    exceed the body length of the first candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.doji(O_2, C_2),
            np.logical_not(cf.exlong_ls(O_2, L_2, C_2)),
            np.logical_not(cf.exlong_us(O_2, L_2, C_2)),
        )
    )


def doji_star_bullish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: long black candle followed by a doji with a downward body gap, the
    doji cannot have extremely long shadows and the total height of the shadows cannot
    exceed the body length of the first candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.doji(O_2, C_2),
            np.logical_not(cf.exlong_ls(O_2, L_2, C_2)),
            np.logical_not(cf.exlong_us(O_2, L_2, C_2)),
        )
    )


def engulfing_bearish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle followed by an 'engulfing' black candle (opens above
    the previous close, closes below the previous open).

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.white_body(O_1, C_1), cf.black_body(O_2, C_2), O_1 > C_2, O_2 > C_1)
    )


def engulfing_bearish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle followed by an 'engulfing' black candle (opens above
    the previous close, closes below the previous open).

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (cf.white_body(O_1, C_1), cf.black_body(O_2, C_2), O_1 > C_2, O_2 > C_1)
    )


def engulfing_bearish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle followed by an 'engulfing' black candle (opens above
    the previous close, closes below the previous open).

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            O_1 > C_2,
            O_2 > C_1,
        )
    )


def engulfing_bullish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle followed by an 'engulfing' white candle (opens below
    the previous closes, closes above the previous open).

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            C_1 > O_2,
            C_2 > O_1,
        )
    )


def engulfing_bullish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle followed by an 'engulfing' white candle (opens below
    the previous closes, closes above the previous open).

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (cf.black_body(O_1, C_1), cf.white_body(O_2, C_2), C_1 > O_2, C_2 > O_1)
    )


def engulfing_bullish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle followed by an 'engulfing' white candle (opens below
    the previous closes, closes above the previous open).

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.black_body(O_1, C_1), cf.white_body(O_2, C_2), C_1 > O_2, C_2 > O_1)
    )


def hammer_inverted_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle with a small lower shadow followed by a short
    candle of either color with a long upper shadow and no lower shadow. There is a
    downwards body gap between the two candles.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.small_ls(O_1, L_1, C_1),
            cf.short_body(O_2, C_2),
            cf.long_us(O_2, H_2, C_2),
            cf.no_ls(O_2, L_2, C_2),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
        )
    )


def hammer_inverted_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle with a small lower shadow followed by a short
    candle of either color with a long upper shadow and no lower shadow. There is a
    downwards body gap between the two candles.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.small_ls(O_1, L_1, C_1),
            cf.short_body(O_2, C_2),
            cf.long_us(O_2, H_2, C_2),
            cf.no_ls(O_2, L_2, C_2),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
        )
    )


def hammer_inverted_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle with a small lower shadow followed by a short
    candle of either color with a long upper shadow and no lower shadow. There is a
    downwards body gap between the two candles.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.small_ls(O_1, L_1, C_1),
            cf.short_body(O_2, C_2),
            cf.long_us(O_2, H_2, C_2),
            cf.no_ls(O_2, L_2, C_2),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
        )
    )


def harami_bearish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a short black candle. The open and
    close of the second candle lies within the body of the first candle. The tops and
    bottoms can be equal, but not both at the same time.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            np.logical_or(
                np.logical_and(C_1 >= O_2, C_2 > O_1),
                np.logical_and(C_1 > O_2, C_2 >= O_1),
            ),
        )
    )


def harami_bearish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a short black candle. The open and
    close of the second candle lies within the body of the first candle. The tops and
    bottoms can be equal, but not both at the same time.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            np.logical_or(
                np.logical_and(C_1 >= O_2, C_2 > O_1),
                np.logical_and(C_1 > O_2, C_2 >= O_1),
            ),
        )
    )


def harami_bearish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a short black candle. The open and
    close of the second candle lies within the body of the first candle. The tops and
    bottoms can be equal, but not both at the same time.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            np.logical_or(
                np.logical_and(C_1 >= O_2, C_2 > O_1),
                np.logical_and(C_1 > O_2, C_2 >= O_1),
            ),
        )
    )


def harami_bullish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a short white candle. The open and
    close of the second candle lies within the body of the first candle. The tops and
    bottoms can be equal, but not both at the same time.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.short_white_body(O_2, C_2),
            np.logical_or(
                np.logical_and(O_1 >= C_2, O_2 > C_1),
                np.logical_and(O_1 > C_2, O_2 >= C_1),
            ),
        )
    )


def harami_bullish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a short white candle. The open and
    close of the second candle lies within the body of the first candle. The tops and
    bottoms can be equal, but not both at the same time.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.short_white_body(O_2, C_2),
            np.logical_or(
                np.logical_and(O_1 >= C_2, O_2 > C_1),
                np.logical_and(O_1 > C_2, O_2 >= C_1),
            ),
        )
    )


def harami_bullish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a short white candle. The open and
    close of the second candle lies within the body of the first candle. The tops and
    bottoms can be equal, but not both at the same time.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.short_white_body(O_2, C_2),
            np.logical_or(
                np.logical_and(O_1 >= C_2, O_2 > C_1),
                np.logical_and(O_1 > C_2, O_2 >= C_1),
            ),
        )
    )


def harami_cross_bearish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a doji with shadows inside the
    previous candle.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.tall_white_body(O_1, C_1), cf.doji(O_2, C_2), L_1 < L_2, H_2 < H_1)
    )


def harami_cross_bearish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a doji with shadows inside the
    previous candle.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (cf.tall_white_body(O_1, C_1), cf.doji(O_2, C_2), L_1 < L_2, H_2 < H_1)
    )


def harami_cross_bearish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a doji with shadows inside the
    previous candle.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (T == -1, cf.tall_white_body(O_1, C_1), cf.doji(O_2, C_2), L_1 < L_2, H_2 < H_1)
    )


def harami_cross_bullish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a doji with shadows inside the
    previous candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (T == -1, cf.tall_black_body(O_1, C_1), cf.doji(O_2, C_2), L_1 < L_2, H_2 < H_1)
    )


def harami_cross_bullish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a doji with shadows inside the
    previous candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (cf.tall_black_body(O_1, C_1), cf.doji(O_2, C_2), L_1 < L_2, H_2 < H_1)
    )


def harami_cross_bullish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a doji with shadows inside the
    previous candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.tall_black_body(O_1, C_1), cf.doji(O_2, C_2), L_1 < L_2, H_2 < H_1)
    )


def homing_pigeon_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a short black candle with body inside
    the previous candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            C_1 < C_2,
            O_2 < O_1,
        )
    )


def homing_pigeon_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a short black candle with body inside
    the previous candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            C_1 < C_2,
            O_2 < O_1,
        )
    )


def homing_pigeon_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a short black candle with body inside
    the previous candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            C_1 < C_2,
            O_2 < O_1,
        )
    )


def in_neck_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a short white candle with open below
    the low of the previous candle. The second candle closes inside the firsts body,
    but not by much.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            O_2 < L_1,
            cf.near_up(C_2, C_1),
        )
    )


def in_neck_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a short white candle with open below
    the low of the previous candle. The second candle closes inside the firsts body,
    but not by much.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            O_2 < L_1,
            cf.near_up(C_2, C_1),
        )
    )


def in_neck_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a short white candle with open below
    the low of the previous candle. The second candle closes inside the firsts body,
    but not by much.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            O_2 < L_1,
            cf.near_up(C_2, C_1),
        )
    )


def kicking_bearish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white marubozu followed by a tall black marubozu, with a
    downside shadow gap.

    Trend: none.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.no_us(O_1, H_1, C_1),
            cf.no_ls(O_1, L_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.no_us(O_2, H_2, C_2),
            cf.no_ls(O_2, L_2, C_2),
            cf.down_shadow_gap(L_1, H_2),
        )
    )


def kicking_bearish_down_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white marubozu followed by a tall black marubozu, with a
    downside shadow gap.

    Trend: none.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.no_us(O_1, H_1, C_1),
            cf.no_ls(O_1, L_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.no_us(O_2, H_2, C_2),
            cf.no_ls(O_2, L_2, C_2),
            cf.down_shadow_gap(L_1, H_2),
        )
    )


def kicking_bearish_up_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white marubozu followed by a tall black marubozu, with a
    downside shadow gap.

    Trend: none.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.no_us(O_1, H_1, C_1),
            cf.no_ls(O_1, L_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.no_us(O_2, H_2, C_2),
            cf.no_ls(O_2, L_2, C_2),
            cf.down_shadow_gap(L_1, H_2),
        )
    )


def kicking_bullish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black marubozu followed by a tall white marubozu, with a
    upside body gap.

    Trend: none.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.no_us(O_1, H_1, C_1),
            cf.no_ls(O_1, L_1, C_1),
            cf.tall_white_body(O_2, C_2),
            cf.no_us(O_2, H_2, C_2),
            cf.no_ls(O_2, L_2, C_2),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
        )
    )


def kicking_bullish_down_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black marubozu followed by a tall white marubozu, with a
    upside body gap.

    Trend: none.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.no_us(O_1, H_1, C_1),
            cf.no_ls(O_1, L_1, C_1),
            cf.tall_white_body(O_2, C_2),
            cf.no_us(O_2, H_2, C_2),
            cf.no_ls(O_2, L_2, C_2),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
        )
    )


def kicking_bullish_up_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black marubozu followed by a tall white marubozu, with a
    upside body gap.

    Trend: none.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.no_us(O_1, H_1, C_1),
            cf.no_ls(O_1, L_1, C_1),
            cf.tall_white_body(O_2, C_2),
            cf.no_us(O_2, H_2, C_2),
            cf.no_ls(O_2, L_2, C_2),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
        )
    )


def last_engulfing_bottom_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle followed by a black candle, which opens above the
    prior body and closes below it.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            O_1 > C_2,
            O_2 > C_1,
        )
    )


def last_engulfing_bottom_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle followed by a black candle, which opens above the
    prior body and closes below it.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (cf.white_body(O_1, C_1), cf.black_body(O_2, C_2), O_1 > C_2, O_2 > C_1)
    )


def last_engulfing_bottom_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle followed by a black candle, which opens above the
    prior body and closes below it.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            O_1 > C_2,
            O_2 > C_1,
        )
    )


def last_engulfing_top_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle followed by a white candle, which opens above the
    prior body and closes below it.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            C_1 > O_2,
            C_2 > O_1,
        )
    )


def last_engulfing_top_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle followed by a white candle, which opens above the
    prior body and closes below it.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (cf.black_body(O_1, C_1), cf.white_body(O_2, C_2), C_1 > O_2, C_2 > O_1)
    )


def last_engulfing_top_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle followed by a white candle, which opens above the
    prior body and closes below it.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            C_1 > O_2,
            C_2 > O_1,
        )
    )


def matching_low_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle and black candle with similar closing prices.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (T == -1, cf.tall_black_body(O_1, C_1), cf.black_body(O_2, C_2), C_1 == C_2)
    )


def matching_low_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle and black candle with similar closing prices.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (cf.tall_black_body(O_1, C_1), cf.black_body(O_2, C_2), C_1 == C_2)
    )


def matching_low_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle and black candle with similar closing prices.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (T == 1, cf.tall_black_body(O_1, C_1), cf.black_body(O_2, C_2), C_1 == C_2)
    )


def meeting_lines_bearish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle and tall black candle with closing prices near to
    each other.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.near(C_1, C_2),
        )
    )


def meeting_lines_bearish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle and tall black candle with closing prices near to
    each other.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (cf.tall_white_body(O_1, C_1), cf.tall_black_body(O_2, C_2), cf.near(C_1, C_2))
    )


def meeting_lines_bearish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle and tall black candle with closing prices near to
    each other.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.near(C_1, C_2),
        )
    )


def meeting_lines_bullish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle and tall white candle with closing prices equal to
    each other.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            C_1 == C_2,
        )
    )


def meeting_lines_bullish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle and tall white candle with closing prices equal to
    each other.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (cf.tall_white_body(O_1, C_1), cf.tall_black_body(O_2, C_2), C_1 == C_2)
    )


def meeting_lines_bullish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle and tall white candle with closing prices equal to
    each other.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            C_1 == C_2,
        )
    )


def on_neck_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle and white candle with low prices equal to each
    other and a downside shadow gap.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.down_shadow_gap(L_1, H_2),
            L_1 == L_2,
        )
    )


def on_neck_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle and white candle with low prices equal to each
    other and a downside shadow gap.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.down_shadow_gap(L_1, H_2),
            L_1 == L_2,
        )
    )


def on_neck_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle and white candle with low prices equal to each
    other and a downside shadow gap.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.down_shadow_gap(L_1, H_2),
            L_1 == L_2,
        )
    )


def piercing_pattern_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle and white candle that opens below the previous low and
    closes between the midpoint and the opening price of the first candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            O_2 < L_1,
            0.5 * (O_1 + C_1) < C_2,
            C_2 < O_1,
        )
    )


def piercing_pattern_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle and white candle that opens below the previous low and
    closes between the midpoint and the opening price of the first candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            O_2 < L_1,
            0.5 * (O_1 + C_1) < C_2,
            C_2 < O_1,
        )
    )


def piercing_pattern_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle and white candle that opens below the previous low and
    closes between the midpoint and the opening price of the first candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            O_2 < L_1,
            0.5 * (O_1 + C_1) < C_2,
            C_2 < O_1,
        )
    )


def separating_lines_bearish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle and tall black candle with similar opening prices.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.near(O_1, O_2),
        )
    )


def separating_lines_bearish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle and tall black candle with similar opening prices.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (cf.tall_white_body(O_1, C_1), cf.tall_black_body(O_2, C_2), cf.near(O_1, O_2))
    )


def separating_lines_bearish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle and tall black candle with similar opening prices.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.near(O_1, O_2),
        )
    )


def separating_lines_bullish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle and tall white candle with similar opening prices.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.tall_white_body(O_2, C_2),
            cf.near(O_1, O_2),
        )
    )


def separating_lines_bullish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle and tall white candle with similar opening prices.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (cf.tall_black_body(O_1, C_1), cf.tall_white_body(O_2, C_2), cf.near(O_1, O_2))
    )


def separating_lines_bullish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle and tall white candle with similar opening prices.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.tall_white_body(O_2, C_2),
            cf.near(O_1, O_2),
        )
    )


def shooting_star_two_candle_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle, followed by a short candle of either color with an
    upper shadow at least 3x the length of the previous candles body, with no lower
    shadow. The first candle has to have an upper shadow and there must be an upwards
    body gap.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.white_body(O_1, C_1),
            np.logical_not(cf.no_us(O_1, H_1, C_1)),
            cf.upper_shadow_length(O_2, H_2, C_2) > 3 * cf.body_height(O_1, C_1),
            cf.short_body(O_2, C_2),
            cf.no_ls(O_2, L_2, C_2),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
        )
    )


def shooting_star_two_candle_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle, followed by a short candle of either color with an
    upper shadow at least 3x the length of the previous candles body, with no lower
    shadow. The first candle has to have an upper shadow and there must be an upwards
    body gap.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.white_body(O_1, C_1),
            np.logical_not(cf.no_us(O_1, H_1, C_1)),
            cf.upper_shadow_length(O_2, H_2, C_2) > 3 * cf.body_height(O_1, C_1),
            cf.short_body(O_2, C_2),
            cf.no_ls(O_2, L_2, C_2),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
        )
    )


def shooting_star_two_candle_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle, followed by a short candle of either color with an
    upper shadow at least 3x the length of the previous candles body, with no lower
    shadow. The first candle has to have an upper shadow and there must be an upwards
    body gap.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.white_body(O_1, C_1),
            np.logical_not(cf.no_us(O_1, H_1, C_1)),
            cf.upper_shadow_length(O_2, H_2, C_2) > 3 * cf.body_height(O_1, C_1),
            cf.short_body(O_2, C_2),
            cf.no_ls(O_2, L_2, C_2),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
        )
    )


def thrusting_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle followed by a white candle that opens below the prior
    low and closes near but below the prior midpoint.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            O_2 < L_1,
            cf.near_up(C_2, 0.5 * (O_1 + C_1)),
        )
    )


def thrusting_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle followed by a white candle that opens below the prior
    low and closes near but below the prior midpoint.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            O_2 < L_1,
            cf.near_up(C_2, 0.5 * (O_1 + C_1)),
        )
    )


def thrusting_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle followed by a white candle that opens below the prior
    low and closes near but below the prior midpoint.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            O_2 < L_1,
            cf.near_up(C_2, 0.5 * (O_1 + C_1)),
        )
    )


def tweezers_bottom_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two candles of either color share the same low price.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce((T == -1, L_1 == L_2))


def tweezers_bottom_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two candles of either color share the same low price.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return L_1 == L_2


def tweezers_bottom_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two candles of either color share the same low price.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce((T == 1, L_1 == L_2))


def tweezers_top_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two candles of either color share the same high price.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce((T == 1, H_1 == H_2))


def tweezers_top_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two candles of either color share the same high price.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return H_1 == H_2


def tweezers_top_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two candles of either color share the same high price.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce((T == -1, H_1 == H_2))


def window_falling_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two candles of either color with a downwards shadow gap.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce((T == -1, cf.down_shadow_gap(L_1, H_2)))


def window_falling_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two candles of either color with a downwards shadow gap.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return cf.down_shadow_gap(L_1, H_2)


def window_falling_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two candles of either color with a downwards shadow gap.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce((T == 1, cf.down_shadow_gap(L_1, H_2)))


def window_rising_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two candles of either color with an upwards shadow gap.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce((T == 1, cf.up_shadow_gap(H_1, L_2)))


def window_rising_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two candles of either color with an upwards shadow gap.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return cf.up_shadow_gap(H_1, L_2)


def window_rising_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two candles of either color with an upwards shadow gap.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2 = candles[0], candles[1]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    return np.logical_and.reduce((T == -1, cf.up_shadow_gap(H_1, L_2)))
