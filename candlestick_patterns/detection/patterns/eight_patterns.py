from detection.patterns.functions import candlestick_functions as cf
import numpy as np


def eight_new_price_lines_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: eight candles of either color reaching a new high.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5, candle_6, candle_7, candle_8 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
        candles[5],
        candles[6],
        candles[7],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    O_6, H_6, L_6, C_6 = candle_6[:, 0], candle_6[:, 1], candle_6[:, 2], candle_6[:, 3]
    O_7, H_7, L_7, C_7 = candle_7[:, 0], candle_7[:, 1], candle_7[:, 2], candle_7[:, 3]
    O_8, H_8, L_8, C_8 = candle_8[:, 0], candle_8[:, 1], candle_8[:, 2], candle_8[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            H_1 < H_2,
            H_2 < H_3,
            H_3 < H_4,
            H_4 < H_5,
            H_5 < H_6,
            H_6 < H_7,
            H_7 < H_8,
        )
    )


def eight_new_price_lines_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: eight candles of either color reaching a new high.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5, candle_6, candle_7, candle_8 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
        candles[5],
        candles[6],
        candles[7],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    O_6, H_6, L_6, C_6 = candle_6[:, 0], candle_6[:, 1], candle_6[:, 2], candle_6[:, 3]
    O_7, H_7, L_7, C_7 = candle_7[:, 0], candle_7[:, 1], candle_7[:, 2], candle_7[:, 3]
    O_8, H_8, L_8, C_8 = candle_8[:, 0], candle_8[:, 1], candle_8[:, 2], candle_8[:, 3]
    return np.logical_and.reduce(
        (
            H_1 < H_2,
            H_2 < H_3,
            H_3 < H_4,
            H_4 < H_5,
            H_5 < H_6,
            H_6 < H_7,
            H_7 < H_8,
        )
    )


def eight_new_price_lines_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: eight candles of either color reaching a new high.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4, candle_5, candle_6, candle_7, candle_8 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
        candles[5],
        candles[6],
        candles[7],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    O_6, H_6, L_6, C_6 = candle_6[:, 0], candle_6[:, 1], candle_6[:, 2], candle_6[:, 3]
    O_7, H_7, L_7, C_7 = candle_7[:, 0], candle_7[:, 1], candle_7[:, 2], candle_7[:, 3]
    O_8, H_8, L_8, C_8 = candle_8[:, 0], candle_8[:, 1], candle_8[:, 2], candle_8[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            H_1 < H_2,
            H_2 < H_3,
            H_3 < H_4,
            H_4 < H_5,
            H_5 < H_6,
            H_6 < H_7,
            H_7 < H_8,
        )
    )
