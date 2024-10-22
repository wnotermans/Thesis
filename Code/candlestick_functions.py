"""Functions related to candlestick parameters, such as height of the body, shadows...
"""

import numpy as np
import numba


@numba.jit
def hb(O: float, C: float) -> float:
    """Inputs: open and close.
    Outputs: the height of the real body.
    NOTE:No normalization
    """
    return abs(O - C)


@numba.jit
def top_body(O: float, C: float) -> float:
    """Inputs: open and close.
    Outputs: the top of the body.
    """
    return max(O, C)


@numba.jit
def bottom_body(O: float, C: float) -> float:
    """Inputs: open and close.
    Outputs: the bottom of the body.
    """
    return min(O, C)


@numba.jit
def upper_shadow(O: float, H: float, C: float) -> float:
    """Inputs: open, high and close.
    Outputs: the length of the upper shadow.
    """
    return H - top_body(O, C)


@numba.jit
def lower_shadow(O: float, L: float, C: float) -> float:
    """Inputs: open, low and close.
    Outputs: the length of the lower shadow.
    """
    return bottom_body(O, C) - L


@numba.jit
def shadow_length(O: float, H: float, L: float, C: float) -> float:
    """Inputs: open, high, low and close.
    Outputs: the total length of the shadows.
    """
    return upper_shadow(O, H, C) + lower_shadow(O, L, C)


@numba.jit
def black_body(O: float, C: float) -> bool:
    """Inputs: open and close.
    Outputs: True if open > close.
    """
    return O > C


@numba.jit
def white_body(O: float, C: float) -> bool:
    """Inputs: open and close.
    Outputs: True if open < close.
    """
    return O < C


@numba.jit
def trend(C: list) -> int:
    if (np.sort(C) == C).all():
        return 1
    elif (-np.sort(-C) == C).all():
        return -1
    else:
        return 0
