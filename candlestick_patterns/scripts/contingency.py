import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from shared import constants

ONE_PARAMETER_DIFFERENT = 2


def print_tables(filter_string: str = "") -> None:
    """
    Print all (2x2) contingency tables.

    A table is constructed and printed when exactly one parameter is different.

    Parameters
    ----------
    filter_str : str
        String to filter by, if the string is not present in the symmetric difference,
        the corresponding contingency table is skipped.
    """
    folders = os.listdir(f"{DIRECTORY}")
    parameter_list = []
    for folder in folders:
        with open(f"{DIRECTORY}/{folder}/parameters.txt") as file:
            parameter_list.append(file.readline().split("#"))
    for i in range(1, len(parameter_list)):
        for j in range(i):
            symmetric_difference = set(parameter_list[i]) ^ set(parameter_list[j])
            if len(symmetric_difference) <= ONE_PARAMETER_DIFFERENT:
                if not any(f"{filter_string}" in s for s in symmetric_difference):
                    continue
                print_contingency_table(
                    folders, parameter_list, symmetric_difference, i, j
                )


def print_contingency_table(
    folder: list[str],
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
    folder : list[str]
        Folder containing the run data.
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
    signif1, non_signif1 = get_signif_non_signif(folder[i])
    signif2, non_signif2 = get_signif_non_signif(folder[j])
    contingency_table = np.array(
        [
            [signif1, signif2],
            [non_signif1, non_signif2],
        ]
    )
    box_width = max([len(x) for x in [*intersection_elements, difference_str]])
    print(f"+{'Common':-^{box_width}}+")
    for line in intersection_elements:
        print(f"|{line:^{box_width}}|")
    print(f"+{'Difference':-^{box_width}}+")
    print(f"|{difference_str:^{box_width}}|")
    print(f"+{'Contingency table':-^{box_width}}+")
    for line in contingency_table:
        print(f"|{' '.join(map(str, line)):^{box_width}}|")
    print(f"+{'p-value'.center(box_width, '-')}+")
    p_value = fisher_exact(contingency_table).pvalue
    if p_value <= constants.THREE_STAR_SIGNIFICANCE:
        p_value_str = f"{p_value:.3g} ***"
    elif p_value <= constants.TWO_STAR_SIGNIFICANCE:
        p_value_str = f"{p_value:.3g} **"
    elif p_value <= constants.ONE_STAR_SIGNIFICANCE:
        p_value_str = f"{p_value:.3g} *"
    else:
        p_value_str = f"{p_value:.3g}"
    print(f"|{p_value_str:^{box_width}}|")
    print(f"+{'-' * box_width}+")


def get_signif_non_signif(folder: str) -> tuple[int, int]:
    """
    Gets the number of significant respectively non-significant patterns from a csv.

    Parameters
    ----------
    folder : str
        Folder containing the summary csv.

    Returns
    -------
    tuple[int, int]
        Number of significant respectively non-significant patterns.
    """
    meta = pd.read_csv(
        f"{DIRECTORY}/{folder}/meta_summary.csv", header=None, index_col=0
    ).squeeze()
    return (
        int(meta["Significant buy signals"]) + int(meta["Significant sell signals"]),
        int(meta["Non significant signals"]),
    )


DIRECTORY = "src/data/runs"
if __name__ == "__main__":
    print_tables()
