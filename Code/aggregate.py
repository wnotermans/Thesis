import pandas as pd


def aggregate(df: pd.DataFrame, interval: int = 5) -> pd.DataFrame:
    """Aggregate data in df over `interval` minutes, defaults to 5.
    Builds up a list sequentually and then outputs to df.
    Output df has columns [`datetime`,`open`,`high`,`low`,`close`,`volume`]

    TODO: parallelize the list generation.
    """

    if interval == 1:
        return df

    out = []

    remainder = len(df) % interval
    if remainder == 0:
        idx_range = range(0, len(df), interval)
    else:
        idx_range = range(0, len(df) - interval, interval)

    for idx in idx_range:  # this for loop can be parallelized.
        # .iloc[] everywhere, otherwise indexing only works for dfs starting at 0.

        # handles time overflow into a new day
        if df["date"].iloc[idx] == df["date"].iloc[idx + interval - 1]:
            Datetime = (
                df["date"].iloc[idx]
                + ": "
                + df["time"].iloc[idx]
                + " - "
                + df["time"].iloc[idx + interval - 1]
            )
        else:
            Datetime = (
                df["date"].iloc[idx]
                + ": "
                + df["time"].iloc[idx]
                + " - "
                + df["date"].iloc[idx + interval - 1]
                + ": "
                + df["time"].iloc[idx + interval - 1]
            )

        Open = df["open"].iloc[idx]
        High = max(df["high"].iloc[idx : idx + interval])  # slice includes the -1.
        Low = min(df["low"].iloc[idx : idx + interval])
        Close = df["close"].iloc[idx + interval - 1]
        Volume = sum(df["volume"].iloc[idx : idx + interval])
        out.append([Datetime, Open, High, Low, Close, Volume])

    if remainder == 1:
        out.append(df[["datetime", "open", "high", "low", "close", "volume"]].iloc[-1])
    elif remainder > 1:
        if df["date"].iloc[-remainder] == df["date"].iloc[-1]:
            Datetime = (
                df["date"].iloc[-remainder]
                + ": "
                + df["time"].iloc[-remainder]
                + " - "
                + df["time"].iloc[-1]
            )
        else:
            Datetime = (
                df["date"].iloc[-remainder]
                + ": "
                + df["time"].iloc[-remainder]
                + " - "
                + df["date"].iloc[-1]
                + ": "
                + df["time"].iloc[-1]
            )

        Open = df["open"].iloc[-remainder]
        High = max(df["high"].iloc[-remainder:])
        Low = min(df["low"].iloc[-remainder:])
        Close = df["close"].iloc[-1]
        Volume = sum(df["volume"].iloc[-remainder:])
        out.append([Datetime, Open, High, Low, Close, Volume])

    return pd.DataFrame(
        out,
        columns=["datetime", "open", "high", "low", "close", "volume"],
    )
