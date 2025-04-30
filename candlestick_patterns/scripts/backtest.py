import os

import pandas as pd

from shared import constants


def backtest() -> None:
    """Perform backtest of technical indicators."""
    base_path = "src/data/runs"
    dataframe_rows = []
    for folder in os.listdir(base_path):
        dataframe_rows.append(
            pd.read_csv(f"{base_path}/{folder}/backtest.csv", header=None)
            .iloc[1:, 1]
            .to_list()
        )
    print(pd.DataFrame(dataframe_rows, columns=constants.INDICATOR_COLUMNS).mean())


if __name__ == "__main__":
    backtest()
