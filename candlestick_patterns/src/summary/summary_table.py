import os
import time

import pandas as pd
import pyarrow.parquet as pq
from word2number import w2n

SIGNIFICANCE_VALUE = 0.05
LOW_COUNT = 100
NUM_PATTERNS = 315
COLS = [
    "Pattern",
    "Number of candlesticks",
    "Number detected",
    "Signal type",
    "Buy evaluation",
    "Binomial test >",
    "Binomial test <",
]
PATTERN_NUMBERS = [
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
]
BUY_NAMES = [
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
]
SELL_NAMES = [
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
]
HOLD_NAMES = [
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
]


def make_summary(filename: str) -> None:
    """
    Aggregates data into a summary table. The following data is included:
    - Pattern name
    - Number of candlesticks
    - Number of patterns detected
    - Type of pattern signal (buy/sell/any/hold)
    - Buy evaluation (win %, average profit), sell evaluation is the opposite
    - binomial tests: "greater", "less"

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

    table = pd.DataFrame(columns=COLS)

    for number_str in PATTERN_NUMBERS:
        for pattern in os.listdir(f"data/patterns/{number_str}"):
            row = [
                f"{pattern.removesuffix('.parquet').replace('_', ' ').strip()}",
                w2n.word_to_num(number_str),
                int(
                    pq.read_table(f"data/patterns/{number_str}/{pattern}")
                    .to_pandas()
                    .sum()
                    .to_numpy()[0]
                ),
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
            if name in BUY_NAMES:
                row.append("Buy")
            elif name in SELL_NAMES:
                row.append("Sell")
            elif name in HOLD_NAMES:
                row.append("Hold")
            else:
                row.append("Any")
            row.extend(
                [
                    x.strip("[']")
                    for x in pq.read_table(
                        f"data/evaluation/{number_str}/{pattern}"
                    ).to_pandas()["evaluation"]
                ]
            )
            row.extend(
                [
                    f"{float(x)}" if x != "/" else 2
                    for x in pq.read_table(f"data/evaluation/{number_str}/{pattern}")
                    .to_pandas()[["up_test", "down_test"]]
                    .iloc[0]
                    .to_numpy()
                ]
            )
            table = pd.concat([table, pd.DataFrame([row], columns=COLS)])
    table = table.reset_index(drop=True)
    significant_buy_signals = (
        table["Binomial test >"].astype(float) < SIGNIFICANCE_VALUE
    ).sum()
    significant_sell_signals = (
        table["Binomial test <"].astype(float) < SIGNIFICANCE_VALUE
    ).sum()
    low_count_signals = (table["Number detected"].astype(int) <= LOW_COUNT).sum()
    non_significant_signals = (
        NUM_PATTERNS
        - significant_buy_signals
        - significant_sell_signals
        - low_count_signals
    )
    best_indices = table[["Binomial test >", "Binomial test <"]].astype(float).idxmin()
    best_buy_pattern, best_sell_pattern = table.iloc[best_indices]["Pattern"].to_numpy()
    best_buy_pvalue = float(table.iloc[best_indices.iloc[0]]["Binomial test >"])
    best_sell_pvalue = float(table.iloc[best_indices.iloc[1]]["Binomial test <"])
    best_buy_win_rate, best_sell_win_rate = (
        table.iloc[best_indices.iloc[0]]["Buy evaluation"],
        table.iloc[best_indices.iloc[1]]["Buy evaluation"],
    )
    total_patterns_detected = (table["Number detected"].astype(int)).sum()
    table = pd.concat([table, pd.DataFrame([[""] * 7], columns=COLS)])
    table = pd.concat(
        [
            table,
            pd.DataFrame(
                [
                    [
                        f"{significant_buy_signals = :d}",
                        f"{significant_sell_signals = :d}",
                        f"{non_significant_signals = :d}",
                        f"{best_buy_pattern = }, {best_buy_pvalue = :.5g}, "
                        f"{best_buy_win_rate = }",
                        f"{best_sell_pattern = }, {best_sell_pvalue = :.5g}, "
                        f"{best_sell_win_rate = }",
                        f"{total_patterns_detected = :d}",
                        "",
                    ]
                ],
                columns=COLS,
            ),
        ]
    )

    table.to_csv(f"data/summaries/{filename}.csv", index=False)

    print(
        f"Making summary table done in {round(time.perf_counter() - t, 2):>3.2f}s",
        end="\n\n",
    )
