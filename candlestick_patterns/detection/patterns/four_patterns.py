from detection.patterns.functions import candlestick_functions as cf
import numpy as np


def concealing_baby_swallow_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two tall black candles without shadows, black candle with long upper
    shadow and a downwards body gap compared to #1 and #2. The high of #3 is inside the
    body of #2. #4 is a black candle that completely engulfs #3, including shadows.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.black_body(O_4, C_4),
            cf.down_body_gap(O_1, C_1, O_3, C_3),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
            cf.no_us(O_1, H_1, C_1),
            cf.no_us(O_2, H_2, C_2),
            cf.no_ls(O_1, L_1, C_1),
            cf.no_ls(O_2, L_2, C_2),
            cf.long_ls(O_3, L_3, C_3),
            O_2 > H_3,
            H_3 > C_2,
            H_4 > H_3,
            L_3 > L_4,
        )
    )


def concealing_baby_swallow_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two tall black candles without shadows, black candle with long upper
    shadow and a downwards body gap compared to #1 and #2. The high of #3 is inside the
    body of #2. #4 is a black candle that completely engulfs #3, including shadows.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.black_body(O_4, C_4),
            cf.down_body_gap(O_1, C_1, O_3, C_3),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
            cf.no_us(O_1, H_1, C_1),
            cf.no_us(O_2, H_2, C_2),
            cf.no_ls(O_1, L_1, C_1),
            cf.no_ls(O_2, L_2, C_2),
            cf.long_ls(O_3, L_3, C_3),
            O_2 > H_3,
            H_3 > C_2,
            H_4 > H_3,
            L_3 > L_4,
        )
    )


def concealing_baby_swallow_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two tall black candles without shadows, black candle with long upper
    shadow and a downwards body gap compared to #1 and #2. The high of #3 is inside the
    body of #2. #4 is a black candle that completely engulfs #3, including shadows.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3, candle_4 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.black_body(O_4, C_4),
            cf.down_body_gap(O_1, C_1, O_3, C_3),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
            cf.no_us(O_1, H_1, C_1),
            cf.no_us(O_2, H_2, C_2),
            cf.no_ls(O_1, L_1, C_1),
            cf.no_ls(O_2, L_2, C_2),
            cf.long_ls(O_3, L_3, C_3),
            O_2 > H_3,
            H_3 > C_2,
            H_4 > H_3,
            L_3 > L_4,
        )
    )


def three_line_strike_bearish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three black candles, each one closing lower, white candle that opens
    below the close of #3 and closes above the open of #1.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.white_body(O_4, C_4),
            C_1 > C_2,
            C_2 > C_3,
            O_4 < C_3,
            C_4 > O_1,
        )
    )


def three_line_strike_bearish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three black candles, each one closing lower, white candle that opens
    below the close of #3 and closes above the open of #1.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    return np.logical_and.reduce(
        (
            cf.black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.white_body(O_4, C_4),
            C_1 > C_2,
            C_2 > C_3,
            O_4 < C_3,
            C_4 > O_1,
        )
    )


def three_line_strike_bearish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three black candles, each one closing lower, white candle that opens
    below the close of #3 and closes above the open of #1.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.white_body(O_4, C_4),
            C_1 > C_2,
            C_2 > C_3,
            O_4 < C_3,
            C_4 > O_1,
        )
    )


def three_line_strike_bullish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three white candles, each one closing higher, black candle that opens
    above the close of #3 and closes below the open of #1.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.black_body(O_4, C_4),
            C_1 < C_2,
            C_2 < C_3,
            O_4 > C_3,
            C_4 < O_1,
        )
    )


def three_line_strike_bullish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three white candles, each one closing higher, black candle that opens
    above the close of #3 and closes below the open of #1.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    return np.logical_and.reduce(
        (
            cf.white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.black_body(O_4, C_4),
            C_1 < C_2,
            C_2 < C_3,
            O_4 > C_3,
            C_4 < O_1,
        )
    )


def three_line_strike_bullish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three white candles, each one closing higher, black candle that opens
    above the close of #3 and closes below the open of #1.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3, candle_4 = (
        candles[0],
        candles[1],
        candles[2],
        candles[3],
    )
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    O_4, H_4, L_4, C_4 = candle_4[:, 0], candle_4[:, 1], candle_4[:, 2], candle_4[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.black_body(O_4, C_4),
            C_1 < C_2,
            C_2 < C_3,
            O_4 > C_3,
            C_4 < O_1,
        )
    )
