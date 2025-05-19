import os

import pandas as pd

COLS = {
    "Significant buy signals": int,
    "Significant sell signals": int,
    "Non significant signals": int,
    "Best buy pattern": str,
    "Best buy p value": float,
    "Best buy win rate": float,
    "Best buy adjusted z-score": float,
    "Best sell pattern": str,
    "Best sell p value": float,
    "Best sell win rate": float,
    "Best sell adjusted z-score": float,
    "Average adjusted z-score": float,
    "Total number detected": int,
}


def bests_and_means(include: list[str] = None, exclude: list[str] = None) -> None:
    """
    Print several best and mean statistics.

    Parameters
    ----------
    filter_string : str, optional, default ""
        String to filter by. If this string is not in parameters.txt, it is excluded
        from the results.
    """
    if include is None:
        include = [""]
    if exclude is None:
        exclude = []
    rows, indices = [], []
    for run_name in os.listdir("src/data/runs"):
        with open(f"src/data/runs/{run_name}/parameters.txt") as f:
            index = f.read()
        if not all(filter_string in index for filter_string in include):
            continue
        if any(filter_string in index for filter_string in exclude):
            continue
        indices.append(index)
        rows.append(
            pd.read_csv(f"src/data/runs/{run_name}/meta_summary.csv", header=None)
            .iloc[:, 1]
            .to_list()
        )
    data = pd.DataFrame(rows, index=indices, columns=COLS.keys()).astype(COLS)
    if n := len(data):
        print(f"Selected {n} runs that {include = } and {exclude = }.")
    else:
        return

    data.loc[data["Best buy win rate"] == "/", "Best buy win rate"] = 50
    data.loc[data["Best sell win rate"] == "/", "Best sell win rate"] = 50

    print(f"Significant buy signals: {data['Significant buy signals'].mean():.2f}")
    print(f"Significant sell signals: {data['Significant sell signals'].mean():.2f}")
    print(f"Non significant signals: {data['Non significant signals'].mean():.2f}")
    print(f"Mean number detected: {data['Total number detected'].mean().astype(int)}")

    best_buy = data["Best buy pattern"].mode().iloc[0]
    best_buy_idx = data[data["Best buy pattern"] == best_buy].index
    best_buy_mean_pvalue = data["Best buy p value"].loc[best_buy_idx].mean()
    best_buy_mean_win_rate = data["Best buy win rate"].loc[best_buy_idx].mean()
    best_sell = data["Best sell pattern"].mode().iloc[0]
    best_sell_idx = data[data["Best sell pattern"] == best_sell].index
    best_sell_mean_pvalue = data["Best sell p value"].loc[best_sell_idx].mean()
    best_sell_mean_win_rate = data["Best sell win rate"].loc[best_sell_idx].mean()
    print(
        "Most occurring best buy pattern with mean p-value and win rate: "
        f"{best_buy}, "
        f"{best_buy_mean_pvalue:.4g}, "
        f"{best_buy_mean_win_rate:.2%}",
    )
    print(
        "Most occurring best sell pattern with mean p-value and win rate: "
        f"{best_sell}, "
        f"{best_sell_mean_pvalue:.4g}, "
        f"{best_sell_mean_win_rate:.2%}",
    )

    most_profitable = data["Most profitable pattern"].mode().iloc[0]
    most_profitable_idx = data[data["Most profitable pattern"] == most_profitable].index
    most_profitable_mean = (
        data["Highest adjusted z-score"].loc[most_profitable_idx].mean()
    )
    print(
        "Most occurring most profitable pattern with mean z-score: "
        f"{most_profitable}, "
        f"{most_profitable_mean:.2f}"
    )

    print(f"Average adjusted z-score: {data['Average adjusted z-score'].mean():.2f}")


if __name__ == "__main__":
    bests_and_means()
