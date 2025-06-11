import numpy as np
import pandas as pd


def geometric_brownian_motion(
    N: int, S0: float = 1000, mu: float = 0, sigma: float = 1
) -> np.ndarray:
    """
    Wiener process of size N.

    Generates a Wiener process of size N. The process starts at zero, and then at each
    step a normally distributed disturbance is added to it. The optional parameter mu
    can be passed to create a drift. Standard deviation is always 1.

    Parameters
    ----------
    N : int
        How many data points to generate.
    S0 : float, optional, default 100
        The initial value.
    mu : float, optional, default 0
        Drift parameter. Defaults to zero meaning no drift.
    sigma : float, optional, default 1
        Volatility of the process.

    Returns
    -------
    np.ndarray
        The Wiener process.
    """
    rng = np.random.default_rng(0)
    dt = 1 / (365 * 23400)
    T = N * dt
    t = np.linspace(0, T, N)
    W = rng.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    return S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)


if __name__ == "__main__":
    time_idx = pd.concat(
        [
            pd.date_range(
                start=f"{date} 09:30:00", end=f"{date} 15:59:59", freq="s"
            ).to_series()
            for date in pd.date_range(start="2001-01-01", end="2020-12-31")
        ]
    )

    M = 8.764849e-07
    S = 6.466583e-04
    GBM_series = pd.Series(geometric_brownian_motion(len(time_idx), sigma=S), time_idx)
    # GBM_series = pd.Series(
    #     geometric_brownian_motion(len(time_idx), mu=M, sigma=S), time_idx
    # )  # drift
    GBM_df = GBM_series.resample("1min", label="right").ohlc()
    GBM_df["datetime"] = GBM_df.index
    GBM_df["volume"] = 0
    GBM_df.to_parquet("src/data/Wiener.parquet", compression="brotli")
