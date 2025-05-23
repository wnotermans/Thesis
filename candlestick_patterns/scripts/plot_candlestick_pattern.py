from datetime import datetime, timedelta

import matplotlib.axes
import mplfinance as mpf
import pandas as pd


def parse_candlestick(
    open_price: float,
    body_size: float,
    upper_shadow: float,
    lower_shadow: float,
    color: str,
) -> tuple[float, float, float, float] | tuple[None, None, None, None]:
    """
    Parse candlestick data into OHLC format.

    Parameters
    ----------
    open_price : float
        Open price.
    body_size : float
        Length of the body.
    upper_shadow : float
        Length of the upper shadow.
    lower_shadow : float
        Length of the lower shadow.
    color : str
        Color of the candle.

    Returns
    -------
    tuple[float]
        Tuple with OHLC data.

    Raises
    ------
    ValueError
        If an unknown color is passed.
    """
    if color.lower() == "white":
        close_price = open_price + body_size
    elif color.lower() == "black":
        close_price = open_price - body_size
    else:
        raise ValueError("Color must be 'black' or 'white'")

    high = max(open_price, close_price) + upper_shadow
    low = min(open_price, close_price) - lower_shadow

    return open_price, high, low, close_price


def add_trend_indicator(ax: matplotlib.axes.Axes, trend: str) -> None:
    """
    Adds trend indicator bars to the bottom or top left of the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Figure ax.
    trend : str
        * ``up`` for up trend.
        * ``down`` for down trend.
    """
    if trend not in ["up", "down"]:
        return

    bar_height = 0.1
    spacing = 0.1
    x_base = 0.15

    if trend == "up":
        y_base = 0.05
        c = 0.075
    else:
        y_base = 0.85
        c = -0.075

    for i in range(3):
        x = x_base + 0.5 * i * spacing
        ax.plot(
            [x, x],
            [y_base + c * i, y_base + bar_height + c * i],
            color="black",
            transform=ax.transAxes,
            linewidth=2,
        )


def add_arrow_indicator(ax: matplotlib.axes.Axes, arrow: str) -> None:
    """
    Adds an arrow to the middle right of the plot, pointing up or down right.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Figure ax.
    arrow : str
        * ``up`` for up right.
        * ``down`` for down right.
    """
    if arrow not in ["up", "down"]:
        return
    xy = (0.95, 0.6) if arrow == "up" else (0.95, 0.4)
    xytext = (0.85, 0.5)
    arrowprops = {"arrowstyle": "->", "lw": 2}
    ax.annotate(
        "",
        xy=xy,
        xytext=xytext,
        arrowprops=arrowprops,
        xycoords="figure fraction",
        transform=ax.transAxes,
    )


def plot_candlestick_pattern() -> None:
    """Make and plot a candlestick pattern."""
    name = "Stick sandwich"
    trend = "down"  # "up", "down"
    arrow = "up"
    candlestick_data = [
        {
            "color": "white",
            "body_size": 3,
            "upper_shadow": 1,
            "lower_shadow": 1,
            "open_price": 100,
        },
        {
            "color": "black",
            "body_size": 0,
            "upper_shadow": 0.5,
            "lower_shadow": 0.5,
            "open_price": 106,
        },
        {
            "color": "black",
            "body_size": 3,
            "upper_shadow": 1,
            "lower_shadow": 1,
            "open_price": 104,
        },
    ]
    candles = [parse_candlestick(**data) for data in candlestick_data]
    if trend in ["up", "down"]:
        candles.insert(0, (None, None, None, None))

    start_time = datetime.now()
    dates = [start_time + timedelta(minutes=5 * i) for i in range(len(candles))]

    data = pd.DataFrame(
        candles, columns=["Open", "High", "Low", "Close"], index=pd.DatetimeIndex(dates)
    )

    cm = 1 / 2.54
    fig, axlist = mpf.plot(
        data,
        type="candle",
        returnfig=True,
        figsize=(14.8 * cm, 10.5 * cm),
        axisoff=True,
    )
    axlist[0].set_title(f"{name}", x=0.5, fontdict={"fontweight": "bold"})

    add_trend_indicator(axlist[0], trend=trend)
    add_arrow_indicator(axlist[0], arrow=arrow)

    fig.savefig(f"{name}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    plot_candlestick_pattern()
