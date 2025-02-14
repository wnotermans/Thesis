import numpy as np

from detection.patterns.functions import candlestick_functions as cf


def thirteen_new_price_lines_(
    candles: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: thirteen candles of either color reaching a new high.

    Trend: up.

    Prediction: reversal.
    """
    (
        candle_1,
        candle_2,
        candle_3,
        candle_4,
        candle_5,
        candle_6,
        candle_7,
        candle_8,
        candle_9,
        candle_10,
        candle_11,
        candle_12,
        candle_13,
    ) = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
        candles[5],
        candles[6],
        candles[7],
        candles[8],
        candles[9],
        candles[10],
        candles[11],
        candles[12],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    O_6, H_6, L_6, C_6 = candle_6[:, 0], candle_6[:, 1], candle_6[:, 2], candle_6[:, 3]
    O_7, H_7, L_7, C_7 = candle_7[:, 0], candle_7[:, 1], candle_7[:, 2], candle_7[:, 3]
    O_8, H_8, L_8, C_8 = candle_8[:, 0], candle_8[:, 1], candle_8[:, 2], candle_8[:, 3]
    O_9, H_9, L_9, C_9 = candle_9[:, 0], candle_9[:, 1], candle_9[:, 2], candle_9[:, 3]
    O_10, H_10, L_10, C_10 = (
        candle_10[:, 0],
        candle_10[:, 1],
        candle_10[:, 2],
        candle_10[:, 3],
    )
    O_11, H_11, L_11, C_11 = (
        candle_11[:, 0],
        candle_11[:, 1],
        candle_11[:, 2],
        candle_11[:, 3],
    )
    O_12, H_12, L_12, C_12 = (
        candle_12[:, 0],
        candle_12[:, 1],
        candle_12[:, 2],
        candle_12[:, 3],
    )
    O_13, H_13, L_13, C_13 = (
        candle_13[:, 0],
        candle_13[:, 1],
        candle_13[:, 2],
        candle_13[:, 3],
    )
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
            H_8 < H_9,
            H_9 < H_10,
            H_10 < H_11,
            H_11 < H_12,
            H_12 < H_13,
        )
    )


def thirteen_new_price_lines_no_trend(
    candles: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: thirteen candles of either color reaching a new high.

    Trend: up.

    Prediction: reversal.
    """
    (
        candle_1,
        candle_2,
        candle_3,
        candle_4,
        candle_5,
        candle_6,
        candle_7,
        candle_8,
        candle_9,
        candle_10,
        candle_11,
        candle_12,
        candle_13,
    ) = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
        candles[5],
        candles[6],
        candles[7],
        candles[8],
        candles[9],
        candles[10],
        candles[11],
        candles[12],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    O_6, H_6, L_6, C_6 = candle_6[:, 0], candle_6[:, 1], candle_6[:, 2], candle_6[:, 3]
    O_7, H_7, L_7, C_7 = candle_7[:, 0], candle_7[:, 1], candle_7[:, 2], candle_7[:, 3]
    O_8, H_8, L_8, C_8 = candle_8[:, 0], candle_8[:, 1], candle_8[:, 2], candle_8[:, 3]
    O_9, H_9, L_9, C_9 = candle_9[:, 0], candle_9[:, 1], candle_9[:, 2], candle_9[:, 3]
    O_10, H_10, L_10, C_10 = (
        candle_10[:, 0],
        candle_10[:, 1],
        candle_10[:, 2],
        candle_10[:, 3],
    )
    O_11, H_11, L_11, C_11 = (
        candle_11[:, 0],
        candle_11[:, 1],
        candle_11[:, 2],
        candle_11[:, 3],
    )
    O_12, H_12, L_12, C_12 = (
        candle_12[:, 0],
        candle_12[:, 1],
        candle_12[:, 2],
        candle_12[:, 3],
    )
    O_13, H_13, L_13, C_13 = (
        candle_13[:, 0],
        candle_13[:, 1],
        candle_13[:, 2],
        candle_13[:, 3],
    )
    return np.logical_and.reduce(
        (
            H_1 < H_2,
            H_2 < H_3,
            H_3 < H_4,
            H_4 < H_5,
            H_5 < H_6,
            H_6 < H_7,
            H_7 < H_8,
            H_8 < H_9,
            H_9 < H_10,
            H_10 < H_11,
            H_11 < H_12,
            H_12 < H_13,
        )
    )


def thirteen_new_price_lines_opp_trend(
    candles: np.ndarray, T: np.ndarray, percentile: tuple
) -> bool:
    """Definition: thirteen candles of either color reaching a new high.

    Trend: up.

    Prediction: reversal.
    """
    (
        candle_1,
        candle_2,
        candle_3,
        candle_4,
        candle_5,
        candle_6,
        candle_7,
        candle_8,
        candle_9,
        candle_10,
        candle_11,
        candle_12,
        candle_13,
    ) = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
        candles[4],
        candles[5],
        candles[6],
        candles[7],
        candles[8],
        candles[9],
        candles[10],
        candles[11],
        candles[12],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    O_5, H_5, L_5, C_5 = candle_5[:, 0], candle_5[:, 1], candle_5[:, 2], candle_5[:, 3]
    O_6, H_6, L_6, C_6 = candle_6[:, 0], candle_6[:, 1], candle_6[:, 2], candle_6[:, 3]
    O_7, H_7, L_7, C_7 = candle_7[:, 0], candle_7[:, 1], candle_7[:, 2], candle_7[:, 3]
    O_8, H_8, L_8, C_8 = candle_8[:, 0], candle_8[:, 1], candle_8[:, 2], candle_8[:, 3]
    O_9, H_9, L_9, C_9 = candle_9[:, 0], candle_9[:, 1], candle_9[:, 2], candle_9[:, 3]
    O_10, H_10, L_10, C_10 = (
        candle_10[:, 0],
        candle_10[:, 1],
        candle_10[:, 2],
        candle_10[:, 3],
    )
    O_11, H_11, L_11, C_11 = (
        candle_11[:, 0],
        candle_11[:, 1],
        candle_11[:, 2],
        candle_11[:, 3],
    )
    O_12, H_12, L_12, C_12 = (
        candle_12[:, 0],
        candle_12[:, 1],
        candle_12[:, 2],
        candle_12[:, 3],
    )
    O_13, H_13, L_13, C_13 = (
        candle_13[:, 0],
        candle_13[:, 1],
        candle_13[:, 2],
        candle_13[:, 3],
    )
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
            H_8 < H_9,
            H_9 < H_10,
            H_10 < H_11,
            H_11 < H_12,
            H_12 < H_13,
        )
    )
