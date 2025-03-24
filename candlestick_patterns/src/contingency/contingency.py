import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ONE_PARAMETER_DIFFERENT = 2
THREE_STAR = 0.001
TWO_STAR = 0.01
ONE_STAR = 0.05


def print_all_tables(directory: str) -> None:
    """
    Print all (2x2) contingency tables.

    A table is constructed and printed when exactly one parameter is different.

    Parameters
    ----------
    directory : str
        Directory containing summaries. Recurses into subdirectories.
    """
    csv_filepaths = list_csv_filepaths(directory)
    parameter_list = [
        pd.read_csv(filepath).iloc[-1].iloc[-1].split("_") for filepath in csv_filepaths
    ]
    for i in range(1, len(parameter_list)):
        for j in range(i):
            symmetric_difference = set(parameter_list[i]) ^ set(parameter_list[j])
            if len(symmetric_difference) == ONE_PARAMETER_DIFFERENT:
                print_contingency_table(
                    csv_filepaths, parameter_list, symmetric_difference, i, j
                )


def print_contingency_table(
    csv_filepaths: list[str],
    parameter_list: list[list[str]],
    symmetric_difference: set[str],
    i: int,
    j: int,
) -> None:
    """
    First performs the necessary computations, then prints a specific 2x2 contingency
    table.

    Parameters
    ----------
    csv_filepaths : list[str]
        Filepaths of the csvs to analyze.
    parameter_list : list[list[str]]
        List of parameter lists.
    symmetric_difference : set[str]
        Symmetric difference of the parameters.
    i : int
        Index of first filepath/parameter list.
    j : int
        Index of second filepath/parameter list.
    """
    intersection_elements = [
        x for x in parameter_list[i] if x in set(parameter_list[j])
    ]
    difference_str = " vs. ".join(map(str, list(symmetric_difference)))
    signif1, non_signif1 = get_signif_non_signif(csv_filepaths[i])
    signif2, non_signif2 = get_signif_non_signif(csv_filepaths[j])
    contingency_table = np.array(
        [
            [signif1, signif2],
            [non_signif1, non_signif2],
        ]
    )
    box_width = max([len(x) for x in [intersection_elements, difference_str]])
    print(f"+{'Common'.center(box_width, '-')}+")
    for line in intersection_elements:
        print(f"|{line:^{box_width}}|")
    print(f"+{'Difference'.center(box_width, '-')}+")
    print(f"|{difference_str:^{box_width}}|")
    print(f"+{'Cont. table'.center(box_width, '-')}+")
    for line in contingency_table:
        print(f"|{' '.join(map(str, line)):^{box_width}}|")
    print(f"+{'p-value'.center(box_width, '-')}+")
    p_value = fisher_exact(contingency_table).pvalue
    if p_value <= THREE_STAR:
        p_value_str = f"{p_value:.3g} ***"
    elif p_value <= TWO_STAR:
        p_value_str = f"{p_value:.3g} **"
    elif p_value <= ONE_STAR:
        p_value_str = f"{p_value:.3g} *"
    else:
        p_value_str = f"{p_value:.3g}"
    print(f"|{p_value_str:^{box_width}}|")
    print(f"+{'-' * box_width}+")


def get_signif_non_signif(file_path: str) -> tuple[int, int]:
    """
    Gets the number of significant respectively non-significant patterns from a csv.

    Parameters
    ----------
    file_path : str
        Filepath to the csv.

    Returns
    -------
    tuple[int, int]
        Number of significant respectively non-significant patterns.
    """
    last_line = pd.read_csv(file_path).iloc[-1]
    return (
        int("".join(filter(str.isnumeric, last_line.iloc[0])))
        + int("".join(filter(str.isnumeric, last_line.iloc[1]))),
        int("".join(filter(str.isnumeric, last_line.iloc[2]))),
    )


def list_csv_filepaths(directory: str) -> list[str]:
    """
    List all csv filepaths in `directory`. Recurses into subdirectories as well.

    Parameters
    ----------
    directory : str
        Name of the directory to search.

    Returns
    -------
    list[str]
        List of the filepaths of the csvs.
    """
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


directory = "../data/summaries"
contingency_table = print_all_tables(directory)
