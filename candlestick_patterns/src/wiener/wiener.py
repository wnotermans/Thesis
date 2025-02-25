import numpy as np
import pandas as pd


def wiener(N: int, mu: float = 0) -> list:
    """
    Wiener process of size N.

    Generates a Wiener process of size N. The process starts at zero, and then at each
    step a normally distributed disturbance is added to it. The optional parameter mu
    can be passed to create a drift. Standard deviation is always 1.

    Parameters
    ----------
    N : int
        Size of the desired output
    mu : float
        Drift parameter. Defaults to zero meaning no drift.

    Returns
    -------
    list
        The Wiener process.
    """
    return np.cumsum(np.insert(np.random.normal(mu, 1, N - 1), 0, 0))


np.random.seed(1)

time_idx = pd.concat(
    [
        pd.date_range(
            start=f"{date} 09:30:00", end=f"{date} 15:59:00", freq="s"
        ).to_series()
        for date in pd.date_range(start="2001-01-01", end="2020-12-31")
    ]
)

ser = pd.Series(wiener(len(time_idx)), time_idx)
# ser = pd.Series(wiener(len(time_idx), 1 / (6.5 * 60 * 60)), time_idx)  # drift
df = ser.resample("1min", label="right").ohlc()
df["datetime"] = df.index
df["volume"] = 0
df.to_parquet("../data/Wiener.parquet", compression="lz4")
