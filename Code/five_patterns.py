import candlestick_functions as cf
import numpy as np


def breakaway_bearish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, white candle with an upwards body gap, candle of
    either color that closes higher, white candle that closes higher, tall black candle
    that closes within the gap between #1 and #2.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_4, C_4),
            cf.tall_black_body(O_5, C_5),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            C_3 > C_2,
            C_4 > C_3,
            C_1 < C_5,
            C_5 < O_2,
        )
    )


def breakaway_bearish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, white candle with an upwards body gap, candle of
    either color that closes higher, white candle that closes higher, tall black candle
    that closes within the gap between #1 and #2.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_4, C_4),
            cf.tall_black_body(O_5, C_5),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            C_3 > C_2,
            C_4 > C_3,
            C_1 < C_5,
            C_5 < O_2,
        )
    )


def breakaway_bearish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, white candle with an upwards body gap, candle of
    either color that closes higher, white candle that closes higher, tall black candle
    that closes within the gap between #1 and #2.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_4, C_4),
            cf.tall_black_body(O_5, C_5),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            C_3 > C_2,
            C_4 > C_3,
            C_1 < C_5,
            C_5 < O_2,
        )
    )


def breakaway_bullish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, black candle with a downwards body gap, candle of
    either color that closes lower, black candle that closes lower, tall white candle
    that closes within the gap between #1 and #2.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_4, C_4),
            cf.tall_white_body(O_5, C_5),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            C_3 < C_2,
            C_4 < C_3,
            O_2 < C_5,
            C_5 < C_1,
        )
    )


def breakaway_bullish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, black candle with a downwards body gap, candle of
    either color that closes lower, black candle that closes lower, tall white candle
    that closes within the gap between #1 and #2.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_4, C_4),
            cf.tall_white_body(O_5, C_5),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            C_3 < C_2,
            C_4 < C_3,
            O_2 < C_5,
            C_5 < C_1,
        )
    )


def breakaway_bullish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, black candle with a downwards body gap, candle of
    either color that closes lower, black candle that closes lower, tall white candle
    that closes within the gap between #1 and #2.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_4, C_4),
            cf.tall_white_body(O_5, C_5),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            C_3 < C_2,
            C_4 < C_3,
            O_2 < C_5,
            C_5 < C_1,
        )
    )


def falling_three_methods_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, small white candle, small candle of either color,
    small white candle, tall black candle. #2, #3 and #4 close higher, but the close of
    #2 and #4 are bounded between the high-low range of #1. #5 closes below #1.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.short_white_body(O_2, C_2),
            cf.short_body(O_3, C_3),
            cf.short_white_body(O_4, C_4),
            cf.tall_black_body(O_5, C_5),
            C_2 < C_3,
            C_3 < C_4,
            H_1 > C_4,
            L_1 < C_2,
            C_5 < C_1,
        )
    )


def falling_three_methods_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, small white candle, small candle of either color,
    small white candle, tall black candle. #2, #3 and #4 close higher, but the close of
    #2 and #4 are bounded between the high-low range of #1. #5 closes below #1.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.short_white_body(O_2, C_2),
            cf.short_body(O_3, C_3),
            cf.short_white_body(O_4, C_4),
            cf.tall_black_body(O_5, C_5),
            C_2 < C_3,
            C_3 < C_4,
            H_1 > C_4,
            L_1 < C_2,
            C_5 < C_1,
        )
    )


def falling_three_methods_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, small white candle, small candle of either color,
    small white candle, tall black candle. #2, #3 and #4 close higher, but the close of
    #2 and #4 are bounded between the high-low range of #1. #5 closes below #1.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.short_white_body(O_2, C_2),
            cf.short_body(O_3, C_3),
            cf.short_white_body(O_4, C_4),
            cf.tall_black_body(O_5, C_5),
            C_2 < C_3,
            C_3 < C_4,
            H_1 > C_4,
            L_1 < C_2,
            C_5 < C_1,
        )
    )


def ladder_bottom_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three tall black candles, each opening and closing lower, black
    candle with an upper shadow, white candle with an upwards body gap.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.black_body(O_4, C_4),
            cf.white_body(O_5, C_5),
            np.logical_not(cf.no_us(O_4, H_4, C_4)),
            cf.up_body_gap(O_4, C_4, O_5, C_5),
            O_2 < O_1,
            O_3 < O_2,
            C_2 < C_1,
            C_3 < C_2,
        )
    )


def ladder_bottom_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three tall black candles, each opening and closing lower, black
    candle with an upper shadow, white candle with an upwards body gap.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.black_body(O_4, C_4),
            cf.white_body(O_5, C_5),
            np.logical_not(cf.no_us(O_4, H_4, C_4)),
            cf.up_body_gap(O_4, C_4, O_5, C_5),
            O_2 < O_1,
            O_3 < O_2,
            C_2 < C_1,
            C_3 < C_2,
        )
    )


def ladder_bottom_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three tall black candles, each opening and closing lower, black
    candle with an upper shadow, white candle with an upwards body gap.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.black_body(O_4, C_4),
            cf.white_body(O_5, C_5),
            np.logical_not(cf.no_us(O_4, H_4, C_4)),
            cf.up_body_gap(O_4, C_4, O_5, C_5),
            O_2 < O_1,
            O_3 < O_2,
            C_2 < C_1,
            C_3 < C_2,
        )
    )


def mat_hold_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, small black candle with an upwards body gap, short
    candle of either color and short black candle both closing lower, with close of #4
    staying above the low of #1, white candle that closes above the maximum of the
    previous highs.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            cf.short_body(O_3, C_3),
            cf.short_black_body(O_4, C_4),
            cf.white_body(O_5, C_5),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            C_2 > C_3,
            C_3 > C_4,
            L_1 < C_4,
            C_5 > np.maximum.reduce((H_1, H_2, H_3, H_4)),
        )
    )


def mat_hold_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, small black candle with an upwards body gap, short
    candle of either color and short black candle both closing lower, with close of #4
    staying above the low of #1, white candle that closes above the maximum of the
    previous highs.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            cf.short_body(O_3, C_3),
            cf.short_black_body(O_4, C_4),
            cf.white_body(O_5, C_5),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            C_2 > C_3,
            C_3 > C_4,
            L_1 < C_4,
            C_5 > np.maximum.reduce((H_1, H_2, H_3, H_4)),
        )
    )


def mat_hold_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, small black candle with an upwards body gap, short
    candle of either color and short black candle both closing lower, with close of #4
    staying above the low of #1, white candle that closes above the maximum of the
    previous highs.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            cf.short_body(O_3, C_3),
            cf.short_black_body(O_4, C_4),
            cf.white_body(O_5, C_5),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            C_2 > C_3,
            C_3 > C_4,
            L_1 < C_4,
            C_5 > np.maximum.reduce((H_1, H_2, H_3, H_4)),
        )
    )


def rising_three_methods_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, small black candle, small candle of either color,
    small black candle, tall white candle. #2, #3 and #4 close lower, but the closes of
    #2 and #4 are bounded between the high-low range of #1. #5 closes above the highs of
    #1:#4.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            cf.short_body(O_3, C_3),
            cf.short_black_body(O_4, C_4),
            cf.tall_white_body(O_5, C_5),
            C_2 > C_3,
            C_3 > C_4,
            H_1 > C_4,
            L_1 < C_2,
            C_5 > np.maximum.reduce((H_1, H_2, H_3, H_4)),
        )
    )


def rising_three_methods_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, small black candle, small candle of either color,
    small black candle, tall white candle. #2, #3 and #4 close lower, but the closes of
    #2 and #4 are bounded between the high-low range of #1. #5 closes above the highs of
    #1:#4.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            cf.short_body(O_3, C_3),
            cf.short_black_body(O_4, C_4),
            cf.tall_white_body(O_5, C_5),
            C_2 > C_3,
            C_3 > C_4,
            H_1 > C_4,
            L_1 < C_2,
            C_5 > np.maximum.reduce((H_1, H_2, H_3, H_4)),
        )
    )


def rising_three_methods_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, small black candle, small candle of either color,
    small black candle, tall white candle. #2, #3 and #4 close lower, but the closes of
    #2 and #4 are bounded between the high-low range of #1. #5 closes above the highs of
    #1:#4.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            cf.short_body(O_3, C_3),
            cf.short_black_body(O_4, C_4),
            cf.tall_white_body(O_5, C_5),
            C_2 > C_3,
            C_3 > C_4,
            H_1 > C_4,
            L_1 < C_2,
            C_5 > np.maximum.reduce((H_1, H_2, H_3, H_4)),
        )
    )
