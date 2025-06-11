import os

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from word2number import w2n

from shared import constants


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

            obs_win_rate, null_win_rate, p_value, number_detected = pd.read_csv(
                f"data/runs/{run_name}/evaluation/{number_str}/{pattern}",
                header=None,
            ).iloc[0]

            pattern_no_ext = pattern.removesuffix("evaluation.csv")
            ser["Pattern"], ser["Number of candlesticks"], ser["Number detected"] = (
                f"{pattern_no_ext.replace('_', ' ').strip()}",
                w2n.word_to_num(number_str),
                number_detected,
            )

            if any(x in str(obs_win_rate) for x in ["+", "-"]):
                plus_minus = obs_win_rate[-1]
                obs_win_rate = float(obs_win_rate[:-1])
            else:
                plus_minus = ""
            if any(x in str(null_win_rate) for x in ["+", "-"]):
                plus_minus = null_win_rate[-1]
                null_win_rate = float(null_win_rate[:-1])

            ser["Observed win rate"] = obs_win_rate
            ser["Null win rate"] = null_win_rate
            ser["p value"] = p_value

            if plus_minus == "-":
                ser["Signal type"] = "Sell"
            elif plus_minus == "+":
                ser["Signal type"] = "Buy"
            else:
                ser["Signal type"] = ""

            if ser["Observed win rate"] != "/" and ser["Null win rate"] not in [
                "/",
                0,
                1,
            ]:
                ser["Adjusted z-score"] = (
                    (obs_win_rate - null_win_rate)
                    / (null_win_rate * (1 - null_win_rate))
                    * np.sqrt(number_detected)
                    * min(np.log(number_detected), np.log(5000))
                )
            else:
                ser["Adjusted z-score"] = 0
            dataframe_rows.append(ser)
    data = pd.DataFrame(dataframe_rows)
    data["p value"] = false_discovery_control(data["p value"], method="by")
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
                ["Pattern", "p value", "Observed win rate", "Adjusted z-score"]
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
                ["Pattern", "p value", "Observed win rate", "Adjusted z-score"]
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
    ) = best_buy_pattern, best_buy_pvalue, best_buy_win_rate, best_buy_z_score
    (
        meta["Best sell pattern"],
        meta["Best sell p value"],
        meta["Best sell win rate"],
        meta["Best sell adjusted z-score"],
    ) = best_sell_pattern, best_sell_pvalue, best_sell_win_rate, best_sell_z_score
    meta["Average adjusted z-score"] = data.loc[
        (data["Adjusted z-score"] != 0) & (~data["Significance"].isna()),
        "Adjusted z-score",
    ].mean()
    meta["Total number detected"] = int(total_patterns_detected)
    data["Observed win rate"] = data["Observed win rate"].replace("/", None)
    data["Null win rate"] = data["Null win rate"].replace("/", None)
    meta["Mean observed win rate"] = (
        mean_obs := (
            data.loc[
                ~data["Observed win rate"].isna(),
                "Observed win rate",
            ]
            .astype(float)
            .mean()
        )
    )
    meta["Mean null win rate"] = (
        mean_null := (
            data.loc[
                ~data["Null win rate"].isna(),
                "Null win rate",
            ]
            .astype(float)
            .mean()
        )
    )
    meta["Excess return"] = mean_obs - mean_null
    meta.to_csv(f"data/runs/{run_name}/meta_summary.csv", header=False)


def make_summaries(*, run_name: str) -> None:
    """
    Make the summary tables and aggregate the indicators.

    Parameters
    ----------
    run_name : str
        The run name.
    """
    make_summary_table(run_name=run_name)
    make_meta_summary(run_name=run_name)
