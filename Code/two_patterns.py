import candlestick_functions as cf
import numpy as np


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


def doji_star_bearish_down_trend(candles: np.ndarray, T: np.ndarray) -> bool:
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
