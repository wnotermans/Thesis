import os

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

pd.options.mode.copy_on_write = True

data = pd.read_parquet("src/data/raw/SPY.parquet")
data["datetime"] = pd.to_datetime(data["datetime"])
data = data.set_index("datetime")
unique_dates = sorted(set(data.index.date))
time_idx = pd.concat(
    [
        pd.date_range(
            start=f"{date} 09:30:00", end=f"{date} 16:00:00", freq="min"
        ).to_series()
        for date in unique_dates
    ]
)
data = data.reindex(
    time_idx
).round(  # filters to US market time and adds NaNs for missing data
    {"open": 3, "high": 3, "low": 3, "close": 3, "volume": 0}
)

data = (
    data.resample("5min", label="right")
    .agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    .dropna()
)  # .agg() fills in missing times with NaNs

reference_set = data[data.index < "2007-01-01"]
main_set = data[data.index >= "2007-01-01"]
main_set["idx"] = np.arange(len(main_set)) + 1

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
        strict=False,
    )
):
    for pattern in os.listdir(
        f"src/data/runs/30c90d00_SPY_5min_SMA_monotonic/detection/{number}"
    ):
        main_set["pattern"] = (
            pq.read_table(
                f"src/data/runs/30c90d00_SPY_5min_SMA_monotonic/detection/{number}/{pattern}"
            )
            .to_pandas()
            .set_index(main_set.index)
        )
        subset = main_set[main_set["pattern"]]
        if len(subset) == 0:
            continue
        idx = subset.sample()["idx"].to_numpy()[0] - n
        cm = 1 / 2.54
        fig, axlist = mpf.plot(
            main_set.iloc[idx : idx + n],
            type="candle",
            returnfig=True,
            figsize=(14.8 * cm, 10.5 * cm),
            axisoff=True,
        )
        plt.tight_layout()
        fig.savefig(f"plots/{pattern.removesuffix('.parquet')}.pdf")
        plt.close(fig)
