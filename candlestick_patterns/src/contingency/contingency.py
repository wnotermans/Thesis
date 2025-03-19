import os

import pandas as pd


def get_buy_sell(file_path: str) -> tuple[int, int] | tuple[None, None]:
    last_line = pd.read_csv(file_path).iloc[-1]
    return (
        (
            int("".join(filter(str.isnumeric, last_line.iloc[0]))),
            int("".join(filter(str.isnumeric, last_line.iloc[1]))),
        )
        if str(last_line)
        else (
            None,
            None,
        )
    )


def list_csv_files(directory: str) -> list[str] | list:
    """
    List all csv files in `directory`. Recurses into subdirectories as well.

    Parameters
    ----------
    directory : str
        Name of the directory to search.

    Returns
    -------
    list[str] | list
        List of the filepaths of the csvs.
    """
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def make_contingency_table(directory: str) -> pd.DataFrame:
    csv_files = list_csv_files(directory)

    # TODO filtering by name here, regex?

    contingency_records = []

    for filename in csv_files:
        buy, sell = get_buy_sell(filename)
        contingency_records.append(
            {
                "Filename": filename.removeprefix("../data/summaries\\"),
                "# sign. buy signals": buy,
                "# sign. sell signals": sell,
            }
        )
    contingency_table = pd.DataFrame.from_records(contingency_records)
    return contingency_table.set_index("Filename")


directory = "../data/summaries"
contingency_table = make_contingency_table(directory)
print(contingency_table)
