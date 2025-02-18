import numpy as np


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
    H_1 = candle_1[:, 1]
    H_2 = candle_2[:, 1]
    H_3 = candle_3[:, 1]
    H_4 = candle_4[:, 1]
    H_5 = candle_5[:, 1]
    H_6 = candle_6[:, 1]
    H_7 = candle_7[:, 1]
    H_8 = candle_8[:, 1]
    H_9 = candle_9[:, 1]
    H_10 = candle_10[:, 1]
    H_11 = candle_11[:, 1]
    H_12 = candle_12[:, 1]
    H_13 = candle_13[:, 1]
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
    H_1 = candle_1[:, 1]
    H_2 = candle_2[:, 1]
    H_3 = candle_3[:, 1]
    H_4 = candle_4[:, 1]
    H_5 = candle_5[:, 1]
    H_6 = candle_6[:, 1]
    H_7 = candle_7[:, 1]
    H_8 = candle_8[:, 1]
    H_9 = candle_9[:, 1]
    H_10 = candle_10[:, 1]
    H_11 = candle_11[:, 1]
    H_12 = candle_12[:, 1]
    H_13 = candle_13[:, 1]
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
    H_1 = candle_1[:, 1]
    H_2 = candle_2[:, 1]
    H_3 = candle_3[:, 1]
    H_4 = candle_4[:, 1]
    H_5 = candle_5[:, 1]
    H_6 = candle_6[:, 1]
    H_7 = candle_7[:, 1]
    H_8 = candle_8[:, 1]
    H_9 = candle_9[:, 1]
    H_10 = candle_10[:, 1]
    H_11 = candle_11[:, 1]
    H_12 = candle_12[:, 1]
    H_13 = candle_13[:, 1]
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
