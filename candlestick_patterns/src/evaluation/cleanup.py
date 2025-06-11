import csv
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from shared import constants

pd.set_option("future.no_silent_downcasting", True)  # noqa: FBT003


def clean(run_name: str) -> None:
    for number_str in constants.PATTERN_NUMBERS_AS_STRING:
        for pattern in os.listdir(f"data/runs/{run_name}/evaluation/{number_str}"):
            if not pattern.endswith("evaluation.csv"):
                continue
            data = pd.read_csv(
                f"data/runs/{run_name}/evaluation/{number_str}/{pattern}",
                header=None,
            )
            data.columns = ["obs win rate", "null win rate", "number detected"]
            plus_minus = data["obs win rate"].str[-1].value_counts().idxmax()
            plus_minus = plus_minus if plus_minus != "/" else ""
            plus_minus_null = data["obs win rate"].str[-1].value_counts().idxmax()
            plus_minus_null = plus_minus_null if plus_minus_null != "/" else ""
            data["obs win rate"] = data["obs win rate"].str[:-1]
            data["null win rate"] = data["obs win rate"].str[:-1]
            data = data.replace(["/", ""], np.nan)

            p_value = 0
            if data["obs win rate"].isna().all():
                obs_win_rate = "/"
                p_value = 1
                total = 0
            else:
                obs_win_rate = (
                    data.loc[~data["obs win rate"].isna(), "obs win rate"]
                    .astype(float)
                    .mean()
                )
                total = data["number detected"].sum()
                obs_number = int(obs_win_rate * total)
                obs_win_rate = str(obs_win_rate) + plus_minus
            if data["null win rate"].isna().all():
                null_win_rate = "/"
            else:
                null_win_rate = (
                    data.loc[~data["null win rate"].isna(), "null win rate"]
                    .astype(float)
                    .mean()
                )
            if not p_value and null_win_rate != "/":
                p_value = binomtest(
                    obs_number, total, null_win_rate, alternative="greater"
                ).pvalue
            null_win_rate = str(null_win_rate) + plus_minus_null
            with open(
                f"data/runs/{run_name}/evaluation/{number_str}/{pattern}",
                "w",
            ) as csvfile:
                csv.writer(csvfile).writerow(
                    [obs_win_rate, null_win_rate, p_value, total]
                )
