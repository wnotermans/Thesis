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
    for i in range(len(C)):
        if C[i] >= C[i + 1]:
            break
        if i == len(C) - 1:
            return 1
    for i in range(len(C)):
        if C[i] <= C[i + 1]:
            break
        if i == len(C) - 1:
            return -1
    return 0
