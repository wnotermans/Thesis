import os
import time

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from word2number import w2n

from indicators import backtest
from shared import constants

BUY_NAMES = {
    "belt hold bullish",
    "doji southern",
    "hammer",
    "takuri line",
    "above the stomach",
    "doji star bullish",
    "engulfing bullish",
    "hammer inverted",
    "harami bullish",
    "harami cross bullish",
    "homing pigeon",
    "last engulfing bottom",
    "matching low",
    "meeting lines bullish",
    "piercing pattern",
    "tweezers bottom",
    "abandoned baby bullish",
    "morning doji star",
    "morning star",
    "stick sandwich",
    "three inside up",
    "three outside up",
    "three stars in the south",
    "three white soldiers",
    "tri star bullish",
    "unique three river bottom",
    "concealing baby swallow",
    "breakaway bullish",
    "ladder bottom",
}
SELL_NAMES = {
    "belt hold bearish",
    "doji northern",
    "hanging man",
    "shooting star one candle",
    "below the stomach",
    "dark cloud cover",
    "doji star bearish",
    "engulfing bearish",
    "harami bearish",
    "harami cross bearish",
    "last engulfing top",
    "meeting lines bearish",
    "shooting star two candle",
    "tweezers top",
    "abandoned baby bearish",
    "advance block",
    "deliberation",
    "doji star collapsing",
    "evening doji star",
    "evening star",
    "identical three crows",
    "three black crows",
    "three inside down",
    "three outside down",
    "tri star bearish",
    "two crows",
    "upside gap two crows",
    "breakaway bearish",
    "eight new price lines",
    "ten new price lines",
    "twelve new price lines",
    "thirteen new price lines",
}
HOLD_NAMES = {
    "marubozu black",
    "marubozu closing black",
    "marubozu closing white",
    "marubozu opening black",
    "marubozu opening white",
    "marubozu white",
    "doji gapping down",
    "doji gapping up",
    "in neck",
    "on neck",
    "separating lines bearish",
    "separating lines bullish",
    "thrusting",
    "window falling",
    "window rising",
    "downside gap three methods",
    "downside tasuki gap",
    "side by side white lines bearish",
    "side by side white lines bullish",
    "two black gapping candles",
    "upside gap three methods",
    "upside tasuki gap",
    "three line strike bearish",
    "three line strike bullish",
    "falling three methods",
    "mat hold",
    "rising three methods",
    "long black day",
    "long white day",
}

COLUMN_HEADERS = [
    "Pattern",
    "Number of candlesticks",
    "Number detected",
    "Signal type",
    "Win rate",
    "Binomial test sell",
    "Binomial test buy",
    "Significance",
]


def make_summary_table(*, run_name: str) -> None:
    """
    Aggregates data into a summary table. The following data is included:
    - Pattern name
    - Number of candlesticks
    - Number of patterns detected
    - Type of pattern signal (buy/sell/any/hold)
    - Win rate
    - binomial tests: "greater", "less"
    - Significance (``***``/``**``/``*``)

    Parameters
    ----------
    filename : str
        Filename of the output ``.csv`` file.

    Returns
    -------
    None
        ``filename.csv`` to disk.
    """
    dataframe_rows = []

    for number_str in constants.PATTERN_NUMBERS_AS_STRING:
        for pattern in os.listdir(f"data/runs/{run_name}/evaluation/{number_str}"):
            if not pattern.endswith("evaluation.csv"):
                continue

            ser = pd.Series()

            win_rate, down_test, up_test, number_detected = pd.read_csv(
                f"data/runs/{run_name}/evaluation/{number_str}/{pattern}",
                header=None,
            ).iloc[0]

            pattern_no_ext = pattern.removesuffix("evaluation.csv")
            ser["Pattern"], ser["Number of candlesticks"], ser["Number detected"] = (
                f"{pattern_no_ext.replace('_', ' ').strip()}",
                w2n.word_to_num(number_str),
                number_detected,
            )

            down_test = down_test if down_test != "/" else 1.0
            up_test = up_test if up_test != "/" else 1.0
            ser["Absolute win rate"] = win_rate
            ser["p value"] = min(down_test, up_test)

            if down_test < up_test:
                ser["Signal type"] = "Sell"
            elif up_test < down_test:
                ser["Signal type"] = "Buy"
            else:
                ser["Signal type"] = ""

            if ser["Absolute win rate"] != "/":
                ser["Adjusted z-score"] = (
                    (2 * win_rate - 1)
                    * np.sqrt(number_detected)
                    * min(np.log(number_detected), np.log(5000))
                )
            else:
                ser["Adjusted z-score"] = 0
            dataframe_rows.append(ser)
    data = pd.DataFrame(dataframe_rows)
    data["p value"] = false_discovery_control(data["p value"])
    data["Significance"] = ""
    data.loc[data["p value"] < constants.ONE_STAR_SIGNIFICANCE, "Significance"] = "*"
    data.loc[data["p value"] < constants.TWO_STAR_SIGNIFICANCE, "Significance"] = "**"
    data.loc[data["p value"] < constants.THREE_STAR_SIGNIFICANCE, "Significance"] = (
        "***"
    )
    data.to_csv(f"data/runs/{run_name}/summary.csv", index=False)


def make_meta_summary(*, run_name: str) -> None:
    """
    Make meta summary of best buy/sell patterns, most profitable pattern, number
    detected...

    Parameters
    ----------
    run_name : str
        The run name.
    """
    data = pd.read_csv(f"data/runs/{run_name}/summary.csv")
    data_exclude_low_count = data[data["Number detected"] > 0]
    significant_buy_signals = (
        data_exclude_low_count.loc[
            data_exclude_low_count["Signal type"] == "Buy", "p value"
        ]
        < constants.ONE_STAR_SIGNIFICANCE
    ).sum()
    significant_sell_signals = (
        data_exclude_low_count.loc[
            data_exclude_low_count["Signal type"] == "Sell", "p value"
        ]
        < constants.ONE_STAR_SIGNIFICANCE
    ).sum()
    low_count_signals = (data["Number detected"] == 0).sum()
    non_significant_signals = (
        constants.TOTAL_NUMBER_OF_PATTERNS
        - significant_buy_signals
        - significant_sell_signals
        - low_count_signals
    )
    if (buy_signals := data.loc[data["Signal type"] == "Buy", "p value"]).empty:
        best_buy_pattern, best_buy_pvalue, best_buy_win_rate, best_buy_z_score = (
            "/",
            np.nan,
            np.nan,
            np.nan,
        )
    else:
        best_buy_index = buy_signals.idxmin()
        best_buy_pattern, best_buy_pvalue, best_buy_win_rate, best_buy_z_score = (
            data.iloc[best_buy_index][
                ["Pattern", "p value", "Absolute win rate", "Adjusted z-score"]
            ].to_numpy()
        )
    if (sell_signals := data.loc[data["Signal type"] == "Sell", "p value"]).empty:
        best_sell_pattern, best_sell_pvalue, best_sell_win_rate, best_sell_z_score = (
            "/",
            np.nan,
            np.nan,
            np.nan,
        )
    else:
        best_sell_index = sell_signals.idxmin()
        best_sell_pattern, best_sell_pvalue, best_sell_win_rate, best_sell_z_score = (
            data.iloc[best_sell_index][
                ["Pattern", "p value", "Absolute win rate", "Adjusted z-score"]
            ].to_numpy()
        )
    total_patterns_detected = (data["Number detected"]).sum()

    meta = pd.Series()
    (
        meta["Significant buy signals"],
        meta["Significant sell signals"],
        meta["Non significant signals"],
    ) = significant_buy_signals, significant_sell_signals, non_significant_signals
    (
        meta["Best buy pattern"],
        meta["Best buy p value"],
        meta["Best buy win rate"],
        meta["Best buy adjusted z-score"],
    ) = (best_buy_pattern, best_buy_pvalue, best_buy_win_rate, best_buy_z_score)
    (
        meta["Best sell pattern"],
        meta["Best sell p value"],
        meta["Best sell win rate"],
        meta["Best sell adjusted z-score"],
    ) = (best_sell_pattern, best_sell_pvalue, best_sell_win_rate, best_sell_z_score)
    meta["Average adjusted z-score"] = data.loc[
        (data["Adjusted z-score"] != 0) & (~data["Significance"].isna()),
        "Adjusted z-score",
    ].mean()
    meta["Total number detected"] = int(total_patterns_detected)
    meta.to_csv(f"data/runs/{run_name}/meta_summary.csv", header=False)


def make_summaries(*, run_name: str) -> None:
    """
    Make the summary tables and aggregate the indicators.

    Parameters
    ----------
    run_name : str
        The run name.
    """
    print("Making summary tables", end="\r")
    t = time.perf_counter()
    make_summary_table(run_name=run_name)
    make_meta_summary(run_name=run_name)
    backtest.aggregate_indicators(run_name=run_name)
    print(
        f"Making summary tables done in {time.perf_counter() - t:3.2f}s",
        end="\n\n",
    )
