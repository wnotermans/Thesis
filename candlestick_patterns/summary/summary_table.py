import os
import time

import pyarrow.parquet as pq
from prettytable import PrettyTable
from word2number import w2n


def make_summary(filename: str) -> None:
    """
    Aggregates data into a summary table. The following data is included:
     - Pattern name
     - Number of candlesticks
     - Number of patterns detected
     - Type of pattern signal (buy/sell/any/hold)
     - Buy evaluation (win %, average profit), sell evaluation is the opposite
     - t-tests: "greater", "less", "two-sided"

    Parameters
    ----------
    filename : str
        Filename of the output `txt` file.

    Returns
    -------
    None
        `filename`.txt to disk.
    """
    t = time.perf_counter()
    print("Making summary table", end="\r")

    table = PrettyTable()
    field_names = [
        "Pattern",
        "Number of candlesticks",
        "Number detected",
        "Signal type",
    ]
    field_names.extend(["Buy evaluation", "Binomial test >", "Binomial test <"])
    table.field_names = field_names
    table.align["Pattern"] = "l"
    table.align["Number detected"] = "r"

    for number in [
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
    ]:
        for div, pattern in enumerate(os.listdir(f"data/patterns/{number}")):
            row = [
                f"{pattern.removesuffix(".parquet").replace("_"," ")}",
                w2n.word_to_num(number),
                pq.read_table(f"data/patterns/{number}/{pattern}")
                .to_pandas()
                .sum()
                .values[0],
            ]
            name = (
                pattern.removesuffix(".parquet")
                .replace("_", " ")
                .removesuffix(" no trend")
                .removesuffix(" opp trend")
                .removesuffix(" up trend")
                .removesuffix(" down trend")
                .rstrip()
            )
            if name in [
                "belt hold bullish",
                "doji southern",
                "hammer",
                "takuri line",
                "above the stomach",
                "doji star bullish",
                "engulfing bullish",
                "hammer inverted",
                "harami bullish",
                "harami cross bullish",
                "homing pigeon",
                "last engulfing bottom",
                "matching low",
                "meeting lines bullish",
                "piercing pattern",
                "tweezers bottom",
                "abandoned baby bullish",
                "morning doji star",
                "morning star",
                "stick sandwich",
                "three inside up",
                "three outside up",
                "three stars in the south",
                "three white soldiers",
                "tri star bullish",
                "unique three river bottom",
                "concealing baby swallow",
                "breakaway bullish",
                "ladder bottom",
            ]:
                row.append("Buy")
            elif name in [
                "belt hold bearish",
                "doji northern",
                "hanging man",
                "shooting star one candle",
                "below the stomach",
                "dark cloud cover",
                "doji star bearish",
                "engulfing bearish",
                "harami bearish",
                "harami cross bearish",
                "last engulfing top",
                "meeting lines bearish",
                "shooting star two candle",
                "tweezers top",
                "abandoned baby bearish",
                "advance block",
                "deliberation",
                "doji star collapsing",
                "evening doji star",
                "evening star",
                "identical three crows",
                "three black crows",
                "three inside down",
                "three outside down",
                "tri star bearish",
                "two crows",
                "upside gap two crows",
                "breakaway bearish",
                "eight new price lines",
                "ten new price lines",
                "twelve new price lines",
                "thirteen new price lines",
            ]:
                row.append("Sell")
            elif name in [
                "marubozu black",
                "marubozu closing black",
                "marubozu closing white",
                "marubozu opening black",
                "marubozu opening white",
                "marubozu white",
                "doji gapping down",
                "doji gapping up",
                "in neck",
                "on neck",
                "separating lines bearish",
                "separating lines bullish",
                "thrusting",
                "window falling",
                "window rising",
                "downside gap three methods",
                "downside tasuki gap",
                "side by side white lines bearish",
                "side by side white lines bullish",
                "two black gapping candles",
                "upside gap three methods",
                "upside tasuki gap",
                "three line strike bearish",
                "three line strike bullish",
                "falling three methods",
                "mat hold",
                "rising three methods",
                "long black day",
                "long white day",
            ]:
                row.append("Hold")
            else:
                row.append("Any")
            row.extend(
                [
                    x
                    for x in pq.read_table(f"data/evaluation/{number}/{pattern}")
                    .to_pandas()[["evaluation", "uptest", "downtest"]]
                    .iloc[0]
                    .values
                ]
            )
            table.add_row(
                row,
                divider=((div + 1) % 3 == 0),
            )

    with open(filename, "w") as f:
        f.write(str(table))

    print(
        f"Making summary table done in {round(time.perf_counter()-t,2):>3.2f}s",
        end="\n\n",
    )
