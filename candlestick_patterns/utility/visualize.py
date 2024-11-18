import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
import mplfinance as mpf
import matplotlib.pyplot as plt

df = pd.read_parquet(f"../Data/ESCC.parquet")
df["datetime"] = pd.to_datetime(df["datetime"])
datetime_index = df["datetime"]
df = df.set_index("datetime")  # set datetime as index for mplfinance
df["idx"] = np.arange(len(df)) + 1

for number, n in list(
    zip(
        [
            "one",
            "two",
            "three",
            "four",
            "five",
            "eight",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
        ],
        [1, 2, 3, 4, 5, 8, 10, 11, 12, 13],
    )
):
    for pattern in os.listdir(f"../Data/Patterns/{number}"):
        df["pattern"] = (
            pq.read_table(f"../Data/Patterns/{number}/{pattern}")
            .to_pandas()
            .set_index(datetime_index)
        )
        subset = df[df["pattern"] == True]
        if len(subset) == 0:
            continue
        idx = subset.sample()["idx"].values[0] - n - 5
        fig, axlist = mpf.plot(
            df.iloc[idx : idx + n + 5],
            type="candle",
            returnfig=True,
        )
        axlist[0].set_title(f"{pattern.removesuffix(".parquet").replace("_"," ")}")
        fig.savefig(f"../Data/plots/{pattern.removesuffix(".parquet")}.png")
        plt.close(fig)
