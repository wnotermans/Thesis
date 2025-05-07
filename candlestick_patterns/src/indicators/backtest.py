import os

import pandas as pd

from shared import constants


def aggregate_indicators(*, run_name: str) -> None:
    dataframe_rows = []
    for number_str in constants.PATTERN_NUMBERS_AS_STRING:
        base_path = f"data/runs/{run_name}"
        for pattern in os.listdir(f"{base_path}/evaluation/{number_str}"):
            if not pattern.endswith("indicators.csv"):
                continue
            dataframe_rows.append(
                pd.read_csv(
                    f"{base_path}/evaluation/{number_str}/{pattern}",
                    header=None,
                )
                .iloc[0]
                .to_list()
            )
    (
        pd.DataFrame(dataframe_rows, columns=constants.INDICATOR_COLUMNS)
        .mean()
        .round(2)
        .to_csv(f"{base_path}/backtest.csv", header=False)
    )
