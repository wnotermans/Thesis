# 1: Detectie van patronen, baseline resultaat
Detectie van patronen gebaseerd op initiële definities. Definities liggen niet (100%) vast, kleine wijzigingen zouden de (eventuele) voorspellende kracht kunnen impacteren. Zeker bij marges (5-10-15%?) zou het wel interessant zijn om hiernaar te kijken.

Aandachtspunten om naar te kijken:
- definities van doji's, kleine en grote candles -> gebaseerd op percentielen? welke?
- normalisatie van candles: lengte real body delen door iets anders -> niet, delen door open, close, volume, of baseren op tijd?
- definitie van trend -> (exponential) moving average, Bollinger bands, andere technical indicators, combinaties?
- meten van performance -> vergelijken met random buy signalen, constante hold period / vergelijken met variabele hold period

Op deze manier wil ik komen tot een baseline resultaat (mogelijks/waarschijnlijk(?) negatief?). Aangezien ik veel meer patronen wil bekijken dan andere literatuur zou het wel eens kunnen dat er een aantal patronen tussen zitten met een bepaalde mate van voorspellende kracht.

# 2: Patronen uitbreiden met andere signalen / filters

Aangezien candlestick patronen an sich niet al te veel voorspellende kracht lijken te hebben, worden ze vaak gecombineerd met andere technical indicators. In deze fase wil ik de candlestick patronen dus stapsgewijs combineren met technical indicators om zo tot een beter resultaat te komen

# 3: Machine learning

In een laatste fase zou het gebruik van machine learning interessant kunnen zijn. Zeker het definiëren van trends lijkt hiervoor geschikt. Aangezien ik niet al te veel ervaring heb met implementatie en machine learning ook niet de focus is van deze thesis, wil ik dit tot het laatst bewaren.
