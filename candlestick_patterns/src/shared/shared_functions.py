def print_status_bar(pattern_name: str, i: int, total_patterns: int) -> None:
    """
    Prints out the status bar (function name, progress bar and count).

    Parameters
    ----------
    pattern_name : str
        Name of the pattern being evaluated.
    i : int
        Current iteration number.
    total_patterns : int
        Total number of patterns.

    Returns
    -------
    None
        Prints a line that overwrites the previous status bar.
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
