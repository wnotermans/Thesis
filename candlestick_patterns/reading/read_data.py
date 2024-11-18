import pandas as pd
from trend import trend_calculation
from aggregation import aggregate
import time


def run(filename, agg_interval=1):
    print("Reading and handling data", end="\r")
    t = time.time()
    df = pd.read_parquet(f"../Data/{filename}.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")  # set datetime as index for mplfinance
    print(f"Reading and handling data done in {round(time.time()-t,2):<3.2f}s")

    df = aggregate.aggregate(df, agg_interval)

    print("Calculating moving average", end="\r")
    t = time.time()
    df["5_MA"] = df["close"].rolling(5).mean()
    print(f"Calculating moving average done in {round(time.time()-t,2):<3.2f}s")

    print("Calculating trend", end="\r")
    t = time.time()
    # 7 consecutive increases/decreases in moving average for trend to be defined
    df["trend"] = (
        df["5_MA"]
        .rolling(7)
        .apply(
            trend_calculation.trend,
            raw=True,
            engine="numba",
        )
    )
    del df["5_MA"]
    print(f"Calculating trend done in {round(time.time()-t,2):<3.2f}s", end="\n\n")
    return df
