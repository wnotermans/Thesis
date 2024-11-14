import candlestick_functions as cf
import numpy as np


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
            cf.shadow_length(O_2, H_2, L_2, C_2) < cf.hb(O_1, C_1),
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
            cf.shadow_length(O_2, H_2, L_2, C_2) < cf.hb(O_1, C_1),
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
            cf.shadow_length(O_2, H_2, L_2, C_2) < cf.hb(O_1, C_1),
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
