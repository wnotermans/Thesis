import candlestick_functions as cf
import numpy as np


def abandoned_baby_bearish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: first a short/normal/tall white candle, then a doji, finally a
    short/normal/tall black candle. Between the candles there are upwards and downwards
    shadow gaps, respectively.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            np.logical_or.reduce(
                (
                    cf.short_white_body(O_1, C_1),
                    cf.normal_white_body(O_1, C_1),
                    cf.tall_white_body(O_1, C_1),
                )
            ),
            cf.doji(O_2, C_2),
            cf.up_shadow_gap(H_1, L_2),
            cf.down_shadow_gap(L_2, H_3),
            np.logical_or.reduce(
                (
                    cf.short_black_body(O_3, C_3),
                    cf.normal_black_body(O_3, C_3),
                    cf.tall_black_body(O_3, C_3),
                )
            ),
        )
    )


def abandoned_baby_bearish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: first a short/normal/tall white candle, then a doji, finally a
    short/normal/tall black candle. Between the candles there are upwards and downwards
    shadow gaps, respectively.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            np.logical_or.reduce(
                (
                    cf.short_white_body(O_1, C_1),
                    cf.normal_white_body(O_1, C_1),
                    cf.tall_white_body(O_1, C_1),
                )
            ),
            cf.doji(O_2, C_2),
            cf.up_shadow_gap(H_1, L_2),
            cf.down_shadow_gap(L_2, H_3),
            np.logical_or.reduce(
                (
                    cf.short_black_body(O_3, C_3),
                    cf.normal_black_body(O_3, C_3),
                    cf.tall_black_body(O_3, C_3),
                )
            ),
        )
    )


def abandoned_baby_bearish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: first a short/normal/tall white candle, then a doji, finally a
    short/normal/tall black candle. Between the candles there are upwards and downwards
    shadow gaps, respectively.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            np.logical_or.reduce(
                (
                    cf.short_white_body(O_1, C_1),
                    cf.normal_white_body(O_1, C_1),
                    cf.tall_white_body(O_1, C_1),
                )
            ),
            cf.doji(O_2, C_2),
            cf.up_shadow_gap(H_1, L_2),
            cf.down_shadow_gap(L_2, H_3),
            np.logical_or.reduce(
                (
                    cf.short_black_body(O_3, C_3),
                    cf.normal_black_body(O_3, C_3),
                    cf.tall_black_body(O_3, C_3),
                )
            ),
        )
    )


def abandoned_baby_bullish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: first a black candle, then a doji, finally a white candle. Between
    the candles there are downwards and upwards shadow gaps, respectively.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.black_body(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.down_shadow_gap(L_1, H_2),
            cf.up_shadow_gap(H_2, L_3),
            cf.white_body(O_3, C_3),
        )
    )


def abandoned_baby_bullish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: first a black candle, then a doji, finally a white candle. Between
    the candles there are downwards and upwards shadow gaps, respectively.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.black_body(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.down_shadow_gap(L_1, H_2),
            cf.up_shadow_gap(H_2, L_3),
            cf.white_body(O_3, C_3),
        )
    )


def abandoned_baby_bullish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: first a black candle, then a doji, finally a white candle. Between
    the candles there are downwards and upwards shadow gaps, respectively.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.black_body(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.down_shadow_gap(L_1, H_2),
            cf.up_shadow_gap(H_2, L_3),
            cf.white_body(O_3, C_3),
        )
    )


def advance_block_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three white candles, the last two opening within the previous body.
    Shadows of #2 and #3 are larger than their bodies and the shadows of #1.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            O_1 < O_2,
            O_2 < C_1,
            O_2 < O_3,
            O_3 < C_2,
            cf.shadow_length(O_2, H_2, L_2, C_2) > cf.hb(O_2, C_2),
            cf.shadow_length(O_3, H_3, L_3, C_3) > cf.hb(O_3, C_3),
            cf.shadow_length(O_2, H_2, L_2, C_2) > cf.shadow_length(O_1, H_1, L_1, C_1),
            cf.shadow_length(O_3, H_3, L_3, C_3) > cf.shadow_length(O_1, H_1, L_1, C_1),
        )
    )


def advance_block_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three white candles, the last two opening within the previous body.
    Shadows of #2 and #3 are larger than their bodies and the shadows of #1.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            O_1 < O_2,
            O_2 < C_1,
            O_2 < O_3,
            O_3 < C_2,
            cf.shadow_length(O_2, H_2, L_2, C_2) > cf.hb(O_2, C_2),
            cf.shadow_length(O_3, H_3, L_3, C_3) > cf.hb(O_3, C_3),
            cf.shadow_length(O_2, H_2, L_2, C_2) > cf.shadow_length(O_1, H_1, L_1, C_1),
            cf.shadow_length(O_3, H_3, L_3, C_3) > cf.shadow_length(O_1, H_1, L_1, C_1),
        )
    )


def advance_block_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three white candles, the last two opening within the previous body.
    Shadows of #2 and #3 are larger than their bodies and the shadows of #1.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            O_1 < O_2,
            O_2 < C_1,
            O_2 < O_3,
            O_3 < C_2,
            cf.shadow_length(O_2, H_2, L_2, C_2) > cf.hb(O_2, C_2),
            cf.shadow_length(O_3, H_3, L_3, C_3) > cf.hb(O_3, C_3),
            cf.shadow_length(O_2, H_2, L_2, C_2) > cf.shadow_length(O_1, H_1, L_1, C_1),
            cf.shadow_length(O_3, H_3, L_3, C_3) > cf.shadow_length(O_1, H_1, L_1, C_1),
        )
    )


def deliberation_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two tall white candles and a short white candle that opens near the
    second close. Each candle opens and closes higher then the previous open and close.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.tall_white_body(O_2, C_2),
            cf.short_white_body(O_3, C_3),
            cf.near(O_3, C_2),
            O_1 < O_2,
            O_2 < O_3,
            C_1 < C_2,
            C_2 < C_3,
        )
    )


def deliberation_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two tall white candles and a short white candle that opens near the
    second close. Each candle opens and closes higher then the previous open and close.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.tall_white_body(O_2, C_2),
            cf.short_white_body(O_3, C_3),
            cf.near(O_3, C_2),
            O_1 < O_2,
            O_2 < O_3,
            C_1 < C_2,
            C_2 < C_3,
        )
    )


def deliberation_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two tall white candles and a short white candle that opens near the
    second close. Each candle opens and closes higher then the previous open and close.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.tall_white_body(O_2, C_2),
            cf.short_white_body(O_3, C_3),
            cf.near(O_3, C_2),
            O_1 < O_2,
            O_2 < O_3,
            C_1 < C_2,
            C_2 < C_3,
        )
    )


def doji_star_collapsing_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle, doji, black candle, each with a downward shadow gap.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.white_body(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.down_shadow_gap(L_1, H_2),
            cf.down_shadow_gap(L_2, H_3),
        )
    )


def doji_star_collapsing_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle, doji, black candle, each with a downward shadow gap.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.white_body(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.down_shadow_gap(L_1, H_2),
            cf.down_shadow_gap(L_2, H_3),
        )
    )


def doji_star_collapsing_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle, doji, black candle, each with a downward shadow gap.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.white_body(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.down_shadow_gap(L_1, H_2),
            cf.down_shadow_gap(L_2, H_3),
        )
    )


def downside_gap_three_methods_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two tall black candles with a downside shadow gap followed by a
    white that opens is the body of #2 and closes in the body of #1, bridging the
    shadow gap.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.down_shadow_gap(L_1, H_2),
            O_2 > O_3,
            O_3 > C_2,
            O_1 > C_3,
            C_3 > C_1,
        )
    )


def downside_gap_three_methods_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two tall black candles with a downside shadow gap followed by a
    white that opens is the body of #2 and closes in the body of #1, bridging the
    shadow gap.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.down_shadow_gap(L_1, H_2),
            O_2 > O_3,
            O_3 > C_2,
            O_1 > C_3,
            C_3 > C_1,
        )
    )


def downside_gap_three_methods_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two tall black candles with a downside shadow gap followed by a
    white that opens is the body of #2 and closes in the body of #1, bridging the
    shadow gap.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.down_shadow_gap(L_1, H_2),
            O_2 > O_3,
            O_3 > C_2,
            O_1 > C_3,
            C_3 > C_1,
        )
    )


def downside_tasuki_gap_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two black candles with a downside body gap, followed by a white
    candle that opens in the previous body and closes in the body gap between #1 and #2.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            O_2 > O_3,
            O_3 > C_2,
            O_2 < C_3,
            C_3 < C_1,
        )
    )


def downside_tasuki_gap_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two black candles with a downside body gap, followed by a white
    candle that opens in the previous body and closes in the body gap between #1 and #2.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            O_2 > O_3,
            O_3 > C_2,
            O_2 < C_3,
            C_3 < C_1,
        )
    )


def downside_tasuki_gap_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two black candles with a downside body gap, followed by a white
    candle that opens in the previous body and closes in the body gap between #1 and #2.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            O_2 > O_3,
            O_3 > C_2,
            O_2 < C_3,
            C_3 < C_1,
        )
    )


def evening_doji_star_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, doji, and tall black candle with upside and
    downside body gaps respectively. The third candle closes at or below the midpoint of
    the first candle.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
            O_1 < C_3,
            C_3 <= 0.5 * (O_1 + C_1),
        )
    )


def evening_doji_star_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, doji, and tall black candle with upside and
    downside body gaps respectively. The third candle closes at or below the midpoint of
    the first candle.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
            O_1 < C_3,
            C_3 <= 0.5 * (O_1 + C_1),
        )
    )


def evening_doji_star_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, doji, and tall black candle with upside and
    downside body gaps respectively. The third candle closes at or below the midpoint of
    the first candle.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
            O_1 < C_3,
            C_3 <= 0.5 * (O_1 + C_1),
        )
    )


def evening_star_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, small candle of either color, and tall black
    candle with upside and downside body gaps respectively. The third candle closes at
    or below the midpoint of the first candle.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.short_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
            O_1 < C_3,
            C_3 <= 0.5 * (O_1 + C_1),
        )
    )


def evening_star_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, small candle of either color, and tall black
    candle with upside and downside body gaps respectively. The third candle closes at
    or below the midpoint of the first candle.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.short_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
            O_1 < C_3,
            C_3 <= 0.5 * (O_1 + C_1),
        )
    )


def evening_star_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, small candle of either color, and tall black
    candle with upside and downside body gaps respectively. The third candle closes at
    or below the midpoint of the first candle.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.short_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
            O_1 < C_3,
            C_3 <= 0.5 * (O_1 + C_1),
        )
    )


def identical_three_crows_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three tall black candles, the latter two opening near the prior
    closing prices.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.near(C_1, O_2),
            cf.near(C_2, O_3),
        )
    )


def identical_three_crows_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three tall black candles, the latter two opening near the prior
    closing prices.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.near(C_1, O_2),
            cf.near(C_2, O_3),
        )
    )


def identical_three_crows_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three tall black candles, the latter two opening near the prior
    closing prices.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.near(C_1, O_2),
            cf.near(C_2, O_3),
        )
    )


def morning_doji_star_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, doji, and tall white candle with upside and
    downside body gaps respectively.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.tall_white_body(O_3, C_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.up_body_gap(O_2, C_2, O_3, C_3),
        )
    )


def morning_doji_star_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, doji, and tall white candle with upside and
    downside body gaps respectively.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.tall_white_body(O_3, C_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.up_body_gap(O_2, C_2, O_3, C_3),
        )
    )


def morning_doji_star_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, doji, and tall white candle with upside and
    downside body gaps respectively.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.tall_white_body(O_3, C_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.up_body_gap(O_2, C_2, O_3, C_3),
        )
    )


def morning_star_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, small candle of either color, and tall white
    candle with upside and downside body gaps respectively. The third candle closes
    above the midpoint of the first candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.short_body(O_2, C_2),
            cf.tall_white_body(O_3, C_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.up_body_gap(O_2, C_2, O_3, C_3),
            C_3 > 0.5 * (O_1 + C_1),
        )
    )


def morning_star_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, small candle of either color, and tall white
    candle with upside and downside body gaps respectively. The third candle closes
    above the midpoint of the first candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.short_body(O_2, C_2),
            cf.tall_white_body(O_3, C_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.up_body_gap(O_2, C_2, O_3, C_3),
            C_3 > 0.5 * (O_1 + C_1),
        )
    )


def morning_star_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, small candle of either color, and tall white
    candle with upside and downside body gaps respectively. The third candle closes
    above the midpoint of the first candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.short_body(O_2, C_2),
            cf.tall_white_body(O_3, C_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.up_body_gap(O_2, C_2, O_3, C_3),
            C_3 > 0.5 * (O_1 + C_1),
        )
    )


def side_by_side_white_lines_bearish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle and two white candles with a downward body gap between
    #1-#2 and #1-#3. The opening and closing prices of the white candles are similar.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.near(C_2, C_3),
            cf.near(O_2, O_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.down_body_gap(O_1, C_1, O_3, C_3),
        )
    )


def side_by_side_white_lines_bearish_no_trend(
    candles: np.ndarray, T: np.ndarray
) -> bool:
    """Definition: black candle and two white candles with a downward body gap between
    #1-#2 and #1-#3. The opening and closing prices of the white candles are similar.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.near(C_2, C_3),
            cf.near(O_2, O_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.down_body_gap(O_1, C_1, O_3, C_3),
        )
    )


def side_by_side_white_lines_bearish_opp_trend(
    candles: np.ndarray, T: np.ndarray
) -> bool:
    """Definition: black candle and two white candles with a downward body gap between
    #1-#2 and #1-#3. The opening and closing prices of the white candles are similar.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.near(C_2, C_3),
            cf.near(O_2, O_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.down_body_gap(O_1, C_1, O_3, C_3),
        )
    )


def side_by_side_white_lines_bullish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three white candles with an upward body gap between #1-#2 and #1-#3.
    The opening and closing prices of #2/#3 are similar.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.near(C_2, C_3),
            cf.near(O_2, O_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.up_body_gap(O_1, C_1, O_3, C_3),
        )
    )


def side_by_side_white_lines_bullish_no_trend(
    candles: np.ndarray, T: np.ndarray
) -> bool:
    """Definition: three white candles with an upward body gap between #1-#2 and #1-#3.
    The opening and closing prices of #2/#3 are similar.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.near(C_2, C_3),
            cf.near(O_2, O_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.up_body_gap(O_1, C_1, O_3, C_3),
        )
    )


def side_by_side_white_lines_bullish_opp_trend(
    candles: np.ndarray, T: np.ndarray
) -> bool:
    """Definition: three white candles with an upward body gap between #1-#2 and #1-#3.
    The opening and closing prices of #2/#3 are similar.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            cf.near(C_2, C_3),
            cf.near(O_2, O_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.up_body_gap(O_1, C_1, O_3, C_3),
        )
    )


def stick_sandwich_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black-white-black candles, the high of the white candle is above the
    close of the preceding black candle. The closing prices of the black candles are
    similar.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            C_1 < H_2,
            cf.near(C_3, C_1),
        )
    )


def stick_sandwich_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black-white-black candles, the high of the white candle is above the
    close of the preceding black candle. The closing prices of the black candles are
    similar.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            C_1 < H_2,
            cf.near(C_3, C_1),
        )
    )


def stick_sandwich_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black-white-black candles, the high of the white candle is above the
    close of the preceding black candle. The closing prices of the black candles are
    similar.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            C_1 < H_2,
            cf.near(C_3, C_1),
        )
    )


def three_black_crows_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three tall black candles, each one closing at a new low. The last
    two open in the body of the previous candle. All should close at or near the low.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            C_1 > C_2,
            C_2 > C_3,
            C_1 < O_2,
            O_2 < O_1,
            C_2 < O_3,
            O_3 < O_2,
            cf.near(C_1, L_1),
            cf.near(C_2, L_2),
            cf.near(C_3, L_3),
        )
    )


def three_black_crows_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three tall black candles, each one closing at a new low. The last
    two open in the body of the previous candle. All should close at or near the low.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            C_1 > C_2,
            C_2 > C_3,
            C_1 < O_2,
            O_2 < O_1,
            C_2 < O_3,
            O_3 < O_2,
            cf.near(C_1, L_1),
            cf.near(C_2, L_2),
            cf.near(C_3, L_3),
        )
    )


def three_black_crows_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three tall black candles, each one closing at a new low. The last
    two open in the body of the previous candle. All should close at or near the low.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.tall_black_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            C_1 > C_2,
            C_2 > C_3,
            C_1 < O_2,
            O_2 < O_1,
            C_2 < O_3,
            O_3 < O_2,
            cf.near(C_1, L_1),
            cf.near(C_2, L_2),
            cf.near(C_3, L_3),
        )
    )


def three_inside_down_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a small black candle which opens and
    closes within the previous body, cannot equal the bottom and top together, only one.
    The third candle is black and has to close below the first two closes.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            C_1 >= O_2,
            C_2 >= O_1,
            np.logical_or.reduce(
                (
                    np.logical_and(C_1 == O_2, C_2 != O_1),
                    np.logical_and(C_1 != O_2, C_2 == O_1),
                )
            ),
            C_1 > C_3,
            C_2 > C_3,
        )
    )


def three_inside_down_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a small black candle which opens and
    closes within the previous body, cannot equal the bottom and top together, only one.
    The third candle is black and has to close below the first two closes.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            C_1 >= O_2,
            C_2 >= O_1,
            np.logical_or.reduce(
                (
                    np.logical_and(C_1 == O_2, C_2 != O_1),
                    np.logical_and(C_1 != O_2, C_2 == O_1),
                )
            ),
            C_1 > C_3,
            C_2 > C_3,
        )
    )


def three_inside_down_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle followed by a small black candle which opens and
    closes within the previous body, cannot equal the bottom and top together, only one.
    The third candle is black and has to close below the first two closes.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.short_black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            C_1 >= O_2,
            C_2 >= O_1,
            np.logical_or.reduce(
                (
                    np.logical_and(C_1 == O_2, C_2 != O_1),
                    np.logical_and(C_1 != O_2, C_2 == O_1),
                )
            ),
            C_1 > C_3,
            C_2 > C_3,
        )
    )


def three_inside_up_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a small white candle which opens and
    closes within the previous body, cannot equal the bottom and top together, only one.
    The third candle is white and has to close above the first two closes.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.short_white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            C_1 <= O_2,
            C_2 <= O_1,
            np.logical_or.reduce(
                (
                    np.logical_and(C_1 == O_2, C_2 != O_1),
                    np.logical_and(C_1 != O_2, C_2 == O_1),
                )
            ),
            C_1 < C_3,
            C_2 < C_3,
        )
    )


def three_inside_up_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a small white candle which opens and
    closes within the previous body, cannot equal the bottom and top together, only one.
    The third candle is white and has to close above the first two closes.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.short_white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            C_1 <= O_2,
            C_2 <= O_1,
            np.logical_or.reduce(
                (
                    np.logical_and(C_1 == O_2, C_2 != O_1),
                    np.logical_and(C_1 != O_2, C_2 == O_1),
                )
            ),
            C_1 < C_3,
            C_2 < C_3,
        )
    )


def three_inside_up_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle followed by a small white candle which opens and
    closes within the previous body, cannot equal the bottom and top together, only one.
    The third candle is white and has to close above the first two closes.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.short_white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            C_1 <= O_2,
            C_2 <= O_1,
            np.logical_or.reduce(
                (
                    np.logical_and(C_1 == O_2, C_2 != O_1),
                    np.logical_and(C_1 != O_2, C_2 == O_1),
                )
            ),
            C_1 < C_3,
            C_2 < C_3,
        )
    )


def three_outside_down_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle, black candle that opens higher and closes lower,
    black candle that closes below the previous close.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            C_1 < O_2,
            C_2 < O_1,
            C_3 < C_2,
        )
    )


def three_outside_down_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle, black candle that opens higher and closes lower,
    black candle that closes below the previous close.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            C_1 < O_2,
            C_2 < O_1,
            C_3 < C_2,
        )
    )


def three_outside_down_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: white candle, black candle that opens higher and closes lower,
    black candle that closes below the previous close.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            C_1 < O_2,
            C_2 < O_1,
            C_3 < C_2,
        )
    )


def three_outside_up_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle, white candle that opens lower and closes higher,
    white candle that closes above the previous close.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            C_1 > O_2,
            C_2 > O_1,
            C_3 > C_2,
        )
    )


def three_outside_up_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle, white candle that opens lower and closes higher,
    white candle that closes above the previous close.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            C_1 > O_2,
            C_2 > O_1,
            C_3 > C_2,
        )
    )


def three_outside_up_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: black candle, white candle that opens lower and closes higher,
    white candle that closes above the previous close.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.black_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.white_body(O_3, C_3),
            C_1 > O_2,
            C_2 > O_1,
            C_3 > C_2,
        )
    )


def three_stars_in_the_south_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle with long lower shadow, black candle with a higher
    low and a smaller body length, tall black candle without shadows with a lower high
    and higher low compared to the previous candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.long_ls(O_1, L_1, C_1),
            L_1 < L_2,
            cf.hb(O_1, C_1) > cf.hb(O_2, C_2),
            cf.no_us(O_3, H_3, C_3),
            cf.no_ls(O_3, L_3, C_3),
            H_2 > H_3,
            L_2 < L_3,
        )
    )


def three_stars_in_the_south_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle with long lower shadow, black candle with a higher
    low and a smaller body length, tall black candle without shadows with a lower high
    and higher low compared to the previous candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.long_ls(O_1, L_1, C_1),
            L_1 < L_2,
            cf.hb(O_1, C_1) > cf.hb(O_2, C_2),
            cf.no_us(O_3, H_3, C_3),
            cf.no_ls(O_3, L_3, C_3),
            H_2 > H_3,
            L_2 < L_3,
        )
    )


def three_stars_in_the_south_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle with long lower shadow, black candle with a higher
    low and a smaller body length, tall black candle without shadows with a lower high
    and higher low compared to the previous candle.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.tall_black_body(O_3, C_3),
            cf.long_ls(O_1, L_1, C_1),
            L_1 < L_2,
            cf.hb(O_1, C_1) > cf.hb(O_2, C_2),
            cf.no_us(O_3, H_3, C_3),
            cf.no_ls(O_3, L_3, C_3),
            H_2 > H_3,
            L_2 < L_3,
        )
    )


def three_white_soldiers_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three tall white candles, each reaching a new high. #2 and #3 open in
    the previous body. Closing prices should be near the high for all three candles.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.tall_white_body(O_2, C_2),
            cf.tall_white_body(O_3, C_3),
            H_1 < H_2,
            H_2 < H_3,
            O_1 < O_2,
            O_2 < C_1,
            O_2 < O_3,
            O_3 < C_2,
            cf.near(C_1, H_1),
            cf.near(C_2, H_2),
            cf.near(C_3, H_3),
        )
    )


def three_white_soldiers_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three tall white candles, each reaching a new high. #2 and #3 open in
    the previous body. Closing prices should be near the high for all three candles.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.tall_white_body(O_2, C_2),
            cf.tall_white_body(O_3, C_3),
            H_1 < H_2,
            H_2 < H_3,
            O_1 < O_2,
            O_2 < C_1,
            O_2 < O_3,
            O_3 < C_2,
            cf.near(C_1, H_1),
            cf.near(C_2, H_2),
            cf.near(C_3, H_3),
        )
    )


def three_white_soldiers_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three tall white candles, each reaching a new high. #2 and #3 open in
    the previous body. Closing prices should be near the high for all three candles.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.tall_white_body(O_2, C_2),
            cf.tall_white_body(O_3, C_3),
            H_1 < H_2,
            H_2 < H_3,
            O_1 < O_2,
            O_2 < C_1,
            O_2 < O_3,
            O_3 < C_2,
            cf.near(C_1, H_1),
            cf.near(C_2, H_2),
            cf.near(C_3, H_3),
        )
    )


def tri_star_bearish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three dojis, with an upward and a downward body gap between them.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.doji(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.doji(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
        )
    )


def tri_star_bearish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three dojis, with an upward and a downward body gap between them.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.doji(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.doji(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
        )
    )


def tri_star_bearish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three dojis, with an upward and a downward body gap between them.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.doji(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.doji(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
        )
    )


def tri_star_bullish_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three dojis, with a downward and an upward body gap between them.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.doji(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.doji(O_3, C_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.up_body_gap(O_2, C_2, O_3, C_3),
        )
    )


def tri_star_bullish_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three dojis, with a downward and an upward body gap between them.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.doji(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.doji(O_3, C_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.up_body_gap(O_2, C_2, O_3, C_3),
        )
    )


def tri_star_bullish_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: three dojis, with a downward and an upward body gap between them.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.doji(O_1, C_1),
            cf.doji(O_2, C_2),
            cf.doji(O_3, C_3),
            cf.down_body_gap(O_1, C_1, O_2, C_2),
            cf.up_body_gap(O_2, C_2, O_3, C_3),
        )
    )


def two_black_gapping_candles_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two black candles, such that #1 has a downward shadow gap and #2 has
    a lower high than #1.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.down_shadow_gap(L_1, H_2),
            H_2 > H_3,
        )
    )


def two_black_gapping_candles_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two black candles, such that #1 has a downward shadow gap and #2 has
    a lower high than #1.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.down_shadow_gap(L_1, H_2),
            H_2 > H_3,
        )
    )


def two_black_gapping_candles_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two black candles, such that #1 has a downward shadow gap and #2 has
    a lower high than #1.

    Trend: down.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.down_shadow_gap(L_1, H_2),
            H_2 > H_3,
        )
    )


def two_crows_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, two black candles. Upwards body gap between #1 and
    #3. #3 opens in the previous body and closes in the body of #1.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            C_2 < O_3,
            O_3 < O_2,
            O_1 < C_3,
            C_3 < C_1,
        )
    )


def two_crows_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, two black candles. Upwards body gap between #1 and
    #3. #3 opens in the previous body and closes in the body of #1.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            C_2 < O_3,
            O_3 < O_2,
            O_1 < C_3,
            C_3 < C_1,
        )
    )


def two_crows_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle, two black candles. Upwards body gap between #1 and
    #3. #3 opens in the previous body and closes in the body of #1.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            C_2 < O_3,
            O_3 < O_2,
            O_1 < C_3,
            C_3 < C_1,
        )
    )


def unique_three_river_bottom_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, black candle inside the previous body with a long
    lower shadow below the prior low. Short white candle with a downwards body gap.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.short_white_body(O_3, C_3),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
            O_1 > O_2,
            C_1 < C_2,
            cf.long_ls(O_2, L_2, C_2),
            L_1 > L_2,
        )
    )


def unique_three_river_bottom_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, black candle inside the previous body with a long
    lower shadow below the prior low. Short white candle with a downwards body gap.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.short_white_body(O_3, C_3),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
            O_1 > O_2,
            C_1 < C_2,
            cf.long_ls(O_2, L_2, C_2),
            L_1 > L_2,
        )
    )


def unique_three_river_bottom_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, black candle inside the previous body with a long
    lower shadow below the prior low. Short white candle with a downwards body gap.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.short_white_body(O_3, C_3),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
            O_1 > O_2,
            C_1 < C_2,
            cf.long_ls(O_2, L_2, C_2),
            L_1 > L_2,
        )
    )


def unique_three_river_bottom_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall black candle, black candle inside the previous body with a long
    lower shadow below the prior low. Short white candle with a downwards body gap.

    Trend: down.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_black_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.short_white_body(O_3, C_3),
            cf.down_body_gap(O_2, C_2, O_3, C_3),
            O_1 > O_2,
            C_1 < C_2,
            cf.long_ls(O_2, L_2, C_2),
            L_1 > L_2,
        )
    )


def upside_gap_three_methods_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two tall white candles with an upwards body gap.
    A black candle that bridges the gap.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.tall_white_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.up_shadow_gap(H_1, L_2),
            C_3 < C_1,
            O_2 < O_3,
        )
    )


def upside_gap_three_methods_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two tall white candles with an upwards body gap.
    A black candle that bridges the gap.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.tall_white_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.up_shadow_gap(H_1, L_2),
            C_3 < C_1,
            O_2 < O_3,
        )
    )


def upside_gap_three_methods_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two tall white candles with an upwards body gap.
    A black candle that bridges the gap.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.tall_white_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.up_shadow_gap(H_1, L_2),
            C_3 < C_1,
            O_2 < O_3,
        )
    )


def upside_gap_two_crows_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle , black candle with an upwards body gap, black
    candle that engulfs the previous candle. Close of #3 remains above close of #1.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.tall_white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            O_3 > O_2,
            C_2 > C_3,
            C_3 > C_1,
        )
    )


def upside_gap_two_crows_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle , black candle with an upwards body gap, black
    candle that engulfs the previous candle. Close of #3 remains above close of #1.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.tall_white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            O_3 > O_2,
            C_2 > C_3,
            C_3 > C_1,
        )
    )


def upside_gap_two_crows_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: tall white candle , black candle with an upwards body gap, black
    candle that engulfs the previous candle. Close of #3 remains above close of #1.

    Trend: up.

    Prediction: reversal.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.tall_white_body(O_1, C_1),
            cf.black_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.up_body_gap(O_1, C_1, O_2, C_2),
            O_3 > O_2,
            C_2 > C_3,
            C_3 > C_1,
        )
    )


def upside_tasuki_gap_(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two white candles with an upwards shadow gap, black candle opening in
    the prior body and closing in the gap. Close of #3 is above the high of #1 but below
    the low of #2.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == 1,
            cf.white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.up_shadow_gap(H_1, L_2),
            O_2 < O_3,
            O_3 < C_2,
            C_3 > H_1,
            C_3 < L_2,
        )
    )


def upside_tasuki_gap_no_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two white candles with an upwards shadow gap, black candle opening in
    the prior body and closing in the gap. Close of #3 is above the high of #1 but below
    the low of #2.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            cf.white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.up_shadow_gap(H_1, L_2),
            O_2 < O_3,
            O_3 < C_2,
            C_3 > H_1,
            C_3 < L_2,
        )
    )


def upside_tasuki_gap_opp_trend(candles: np.ndarray, T: np.ndarray) -> bool:
    """Definition: two white candles with an upwards shadow gap, black candle opening in
    the prior body and closing in the gap. Close of #3 is above the high of #1 but below
    the low of #2.

    Trend: up.

    Prediction: continuation.
    """
    candle_1, candle_2, candle_3 = candles[0], candles[1], candles[2]
    O_1, H_1, L_1, C_1 = candle_1[:, 0], candle_1[:, 1], candle_1[:, 2], candle_1[:, 3]
    O_2, H_2, L_2, C_2 = candle_2[:, 0], candle_2[:, 1], candle_2[:, 2], candle_2[:, 3]
    O_3, H_3, L_3, C_3 = candle_3[:, 0], candle_3[:, 1], candle_3[:, 2], candle_3[:, 3]
    return np.logical_and.reduce(
        (
            T == -1,
            cf.white_body(O_1, C_1),
            cf.white_body(O_2, C_2),
            cf.black_body(O_3, C_3),
            cf.up_shadow_gap(H_1, L_2),
            O_2 < O_3,
            O_3 < C_2,
            C_3 > H_1,
            C_3 < L_2,
        )
    )
