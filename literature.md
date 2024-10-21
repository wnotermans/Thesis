# Performance of Candlestick Analysis on Intraday Futures Data (2005)
- intraday (5-minute) data, Duitse futures
- negatief resultaat, zelfs zonder transactiekosten performen de candles niet beter dan random transacties
- kleine en grote candles gedifinieerd a.d.h.v. percententielen
- testen eerst de patronen zelf, dan combinatie met moving average, momentum, relative strength index, moving average convergence/divergence
- benchmark: randomized buy signals, 30 minuten holding time, ook voor de transacties na detectie van een patroon, alternatieve benchmark: 0 earnings, vanwege de stochasticiteit van prijzen op zo'n korte intervallen
- enkel patroon: slechts 2/19 patronen zijn significant goede deals, slechts 2/19 hebben significante earnings
- moving average: 1/19 goede deal, 1/19 significante earnings
- gelijkaardig voor andere combinaties met indicators

# A formal approach to candlestick pattern classification in financial time series
- formele definities van 103 candlestick patronen van de Encyclopedia of candlesticks, a.d.h.v. logica en functies
- candlesticks op dagbasis
- recent onderzoek naar candlestick patronen a.d.h.v. machine learning, waarvoor "computerdefinities" noodzakelijk zijn
- trend a.d.h.v. 4 toenames/afnames in 5-moving average van close
- geen specificatie van normalisatie candle body
- doji: maximaal 0.3% verschil (ook voor shadows)
- small: tussen 0.3% en 1%
- normal: tussen 1% en 2.5%
- long: tussen 2.5% en 5%
- extreme: meer dan 5%
- definities overnemen en de extra candles op een gelijkaardige manier definiëren
- experiment: normalisatie in neighborhood
- classificatie a.d.h.v. machine learning, goede resultaten voor random forest op synthetische data
- focus puur op classificatie, geen analyse van winst/verlies

# Profitable candlestick trading strategies - The evidence from a new perspective
- 60% van traders gebruiken technical analysis
- geen consensus rond candlestick patronen in de literatuur
- daily data, 6 patronen, 5-moving average over de closing prices voor trend
- buy = open na patroon, sell = open na omgekeerd patroon --> variabele holding period
- houden rekening met ex-right en ex-dividend dates
- transactiekosten
- bullish patronen significante returns en winning rates, bearish enkel winning rates
- extra test op bear, bull en oscillating market
- extra out-of-sample test met bootstrap
- positieve resultaten voor de bullish patronen, zelfs zonder filters

# Bullish and Bearish Engulfing Japanese Candlestick patterns: A statistical analysis on the S&P 500 index
- small: kijken t.o.v. tweede candle (max 75/50/25%)
- bestuderen de patronen zowel in de juiste trend als zonder naar de trend te kijken
- daily candles, geen normalisatie voor body
- trend: n-moving average over de close, 70% van n moet stijgend/dalend zijn (n=3/5/7/10)
- evaluatie a.d.h.v. close, open, low/high criterium
- ook analyse van de failures
- positief resultaat: statistisch significant bij alle criteriums buiten close
- trend niet significant
- definitie small (25/50/75%) niet significant
- criterium wel significant
- failures van patroon kleiner dan over alle candles
- analyse van inter-arrival time
- lijkt skew exponential/Poisson, maar ze slagen er niet in een significante verdeling te fitten
- vergelijking tussen verschillende decennia, misschien interessant om te vergelijken tussen data van voor/na 1991, toen candlestick patronen "ontdekt" werden in het westen

# The Application of Japanese Candlestick Trading Strategies in Taiwan
- daily candles
- normalisatie: delen door open voor witte candle, delen door close bij zwart
- percentielen (0-20-80-100)
- trend: 5-moving average, 7 opeenvolgende dalingen/stijgingen voor trend
- evaluatie: buy open na detectie, hold 1-10 dagen, iedere dag return berekenen
- sommige patronen (bijna) nergens significant, sommige elke dag
- vooral positief resultaat bij bullish
- stop loss: -5,-7,-10%, -5% beste
- stop loss verbetert de resultaten