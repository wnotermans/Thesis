def print_status_bar(message: str, i: int, total_patterns: int) -> None:
    """
    Prints out the status bar (function name, progress bar and count), overwriting the
    previous status bar.

    Parameters
    ----------
    message : str
        Message to be printed.
    i : int
        Current iteration number. Automatically adds +1 for convenience with
        ``enumerate()``.
    total_patterns : int
        Total number of patterns.
    """
    left_line = (51 * (i + 1) // total_patterns) * "-"
    right_line = (51 - len(left_line)) * "-"
    progress_bar = f"|{left_line}>>{right_line}|"
    status_bar = f"{message:<62}" + progress_bar + f"({i + 1:>3}/{total_patterns})"
    print(status_bar, end="\r")


def box_print(input_dict: dict) -> None:
    """
    Print out a dictionary's key:value pairs in a nice box

    Parameters
    ----------
    input_dict : dict
        A dictionary
    """
    box_width = max([len(f"{key}={value}") for key, value in input_dict.items()])
    print(f"+{'Parameters':-^{box_width}}+".center(127))
    for key, value in input_dict.items():
        line = "=".join([key, str(value)])
        print(f"|{line:^{box_width}}|".center(127))
    print(f"+{'-' * box_width}+".center(127), end="\n\n")


def set_kwarg_defaults(
    input_kwargs: dict, *, kwargs_to_set: dict | list, default_dict: dict
) -> dict:
    """
    Set default kwargs.

    Parameters
    ----------
    input_kwargs : dict
        The kwargs specified by the user. Left untouched.
    kwargs_to_set : dict | list
        Dict or list whose keys/values will be given a default value when no
        user-defined one is given.
    default_dict : dict
        The dict containing the default key:value pairs.

    Returns
    -------
    dict
        Dict of kwargs with default key:value pairs added.
    """
    out = {**input_kwargs}
    for key in kwargs_to_set:
        out.setdefault(key, default_dict.get(key, {}))
    return out
