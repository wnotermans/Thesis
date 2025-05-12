import mplfinance as mpf
import yfinance as yf

ticker = yf.Ticker("BRK-A")
data = ticker.history(start="2025-01-01", end="2025-02-01")
cm = 1 / 2.54
fig, ax = mpf.plot(
    data,
    type="candle",
    title="Berkshire Hathaway stock price",
    returnfig=True,
    figsize=(14.8 * cm, 10.5 * cm),
)
fig.savefig("BRK.A_candle.pdf", bbox_inches="tight")
