import csv
import os

import pandas as pd


def get_last_line_values(file_path: str) -> tuple[int, int]:
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        last_line = None
        for row in reader:
            last_line = row
    return (
        (
            "".join(filter(str.isnumeric, last_line[0])),
            "".join(filter(str.isnumeric, last_line[1])),
        )
        if last_line
        else (
            None,
            None,
        )
    )


def find_csv_files(directory: str) -> str:
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def update_contingency_table(directory: str) -> pd.DataFrame:
    contingency_table = pd.DataFrame(columns=["File", "Value 1", "Value 2"])

    csv_files = find_csv_files(directory)
    # TODO filtering by name here, regex?

    for file in csv_files:
        value1, value2 = get_last_line_values(file)
        contingency_table = pd.concat(
            [
                contingency_table,
                pd.DataFrame(
                    {"File": file, "Value 1": value1, "Value 2": value2}, index=[0]
                ),
            ],
            ignore_index=True,
        )

    return contingency_table


directory = "../data/summaries"
contingency_table = update_contingency_table(directory)
print(contingency_table)
