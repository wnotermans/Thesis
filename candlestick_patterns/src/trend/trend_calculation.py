import numba


@numba.jit
def monotonic(C: list) -> int:
    """
    Trend calculation based on strict in/decreases.

    Parameters
    ----------
    C : list
        List of closes.

    Returns
    -------
    int
        1 if the list is strictly increasing, -1 if strictly decreasing, 0 otherwise.
    """
    increasing, decreasing = False, False

    for i in range(len(C)):
        if C[i + 1] > C[i]:
            increasing = True
        elif C[i + 1] < C[i]:
            decreasing = True
        else:
            return 0

        if increasing and decreasing:
            return 0

    if increasing:
        return 1
    elif decreasing:
        return -1
