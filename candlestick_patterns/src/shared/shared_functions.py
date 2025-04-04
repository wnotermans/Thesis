def print_status_bar(pattern_name: str, i: int, total_patterns: int) -> None:
    """
    Prints out the status bar (function name, progress bar and count), overwriting the
    previous status bar.

    Parameters
    ----------
    pattern_name : str
        Name of the pattern being evaluated.
    i : int
        Current iteration number.
    total_patterns : int
        Total number of patterns.
    """
    left_line = (51 * (i + 1) // total_patterns) * "-"
    right_line = (51 - len(left_line)) * "-"
    progress_bar = f"|{left_line}>>{right_line}|"
    status_bar = (
        f"Evaluating {pattern_name:<52}"
        + progress_bar
        + f"({i + 1:>3}/{total_patterns})"
    )
    print(status_bar, end="\r")


def box_print(parameters: dict) -> None:
    box_width = max([len(f"{key}={value}") for key, value in parameters.items()])
    print(f"+{'Parameters':-^{box_width}}+".center(127))
    for key, value in parameters.items():
        line = "=".join([key, str(value)])
        print(f"|{line:^{box_width}}|".center(127))
    print(f"+{'-' * box_width}+".center(127), end="\n\n")
