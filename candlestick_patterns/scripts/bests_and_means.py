import os

import pandas as pd

COLS = {
    "Significant buy signals": int,
    "Significant sell signals": int,
    "Non significant signals": int,
    "Best buy pattern": str,
    "Best buy p value": float,
    "Best buy win rate": str,
    "Best sell pattern": str,
    "Best sell p value": float,
    "Best sell win rate": str,
    "Most profitable pattern": str,
    "Most profitable score": float,
    "Average profitability score": float,
    "Total number detected": int,
}


def bests_and_means(filter_string: str = "") -> None:
    rows, indices = [], []
    for run_name in os.listdir("src/data/runs"):
        with open(f"src/data/runs/{run_name}/parameters.txt") as f:
            index = f.read()
        if filter_string not in index:
            continue
        indices.append(index)
        rows.append(
            pd.read_csv(f"src/data/runs/{run_name}/meta_summary.csv", header=None)
            .iloc[:, 1]
            .to_list()
        )
    data = pd.DataFrame(rows, index=indices, columns=COLS.keys()).astype(COLS)

    data[["Best buy win rate", "Best sell win rate"]] = data[
        ["Best buy win rate", "Best sell win rate"]
    ].apply(lambda x: x.str.slice(stop=5).astype(float))

    print(f"Mean significant buy signals: {data['Significant buy signals'].mean():.2f}")
    print(
        f"Mean significant sell signals: {data['Significant sell signals'].mean():.2f}"
    )
    print(f"Mean non significant signals: {data['Non significant signals'].mean():.2f}")

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
        f"{best_buy_mean_pvalue:.4f}, "
        f"{best_buy_mean_win_rate:.2f}%",
    )
    print(
        "Most occurring best sell pattern with mean p-value and win rate: "
        f"{best_sell}, "
        f"{best_sell_mean_pvalue:.4f}, "
        f"{best_sell_mean_win_rate:.2f}%",
    )

    most_profitable = data["Most profitable pattern"].mode().iloc[0]
    most_profitable_idx = data[data["Most profitable pattern"] == most_profitable].index
    most_profitable_mean = data["Most profitable score"].loc[most_profitable_idx].mean()
    print(
        "Most occurring most profitable pattern with mean profitability score: "
        f"{most_profitable}, "
        f"{most_profitable_mean:.2f}"
    )

    print(data["Average profitability score"].describe())
    print(data["Total number detected"].describe())


if __name__ == "__main__":
    bests_and_means()
