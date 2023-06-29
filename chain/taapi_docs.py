RSI_API_DOCS="""Relative Strength Index (RSI)
Base URL: https://api.taapi.io/rsi

The relative strength index (RSI) is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset. The RSI is displayed as an oscillator (a line graph that moves between two extremes) and can have a reading from 0 to 100.
The RSI provides signals that tell investors to buy when the security or currency is oversold and to sell when it is overbought. 

API parameters:
secret: The secret which is emailed to you when you request an API key. 
Note: The secret is: {taapi_key} 
exchange: The exchange you want to calculate the indicator from: gateio or one of our supported exchanges. For other crypto / stock exchanges, please refer to our Client or Manual integration methods.
symbol: Symbol names are always uppercase, with the coin separated by a forward slash and the market: COIN/MARKET. For example: BTC/USDT Bitcoin to Tether, or LTC/BTC Litecoin to Bitcoin...
interval: Interval or time frame: We support the following time frames: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d, 1w. So if you're interested in values on hourly candles, use interval=1h, for daily values use interval=1d, etc.

More examples:
Let's say you want to know the rsi value on the last closed candle on the 30m timeframe. You are not interest in the real-time value, so you use the backtrack=1 optional parameter to go back 1 candle in history to the last closed candle.
```
[GET] https://api.taapi.io/rsi?secret={taapi_key}&exchange=binance&symbol=BTC/USDT&interval=30m&backtrack=1
```

Get rsi values on each of the past X candles in one call:
Let's say you want to know what the rsi daily value was each day for the previous 10 days. You can get this returned by our API easily and efficiently in one call using the backtracks=10 parameter:
```
[GET] https://api.taapi.io/rsi?secret={taapi_key}&exchange=binance&symbol=BTC/USDT&interval=1d&backtracks=10
```
"""

CCI_API_DOCS="""Commodity Channel Index
Base URL: https://api.taapi.io/cci

Developed by Donald Lambert, the Commodity Channel Index​ (CCI) is a momentum-based oscillator used to help determine when an asset is reaching overbought or oversold conditions.

API parameters:
secret: The secret which is emailed to you when you request an API key. 
Note: The secret is: {taapi_key} 
exchange: The exchange you want to calculate the indicator from: gateio or one of our supported exchanges. For other crypto / stock exchanges, please refer to our Client or Manual integration methods.
symbol: Symbol names are always uppercase, with the coin separated by a forward slash and the market: COIN/MARKET. For example: BTC/USDT Bitcoin to Tether, or LTC/BTC Litecoin to Bitcoin...
interval: Interval or time frame: We support the following time frames: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d, 1w. So if you're interested in values on hourly candles, use interval=1h, for daily values use interval=1d, etc.

More examples:
Let's say you want to know the cci value on the last closed candle on the 30m timeframe. You are not interest in the real-time value, so you use the backtrack=1 optional parameter to go back 1 candle in history to the last closed candle.
```
[GET] https://api.taapi.io/cci?secret=MY_SECRET&exchange=binance&symbol=BTC/USDT&interval=30m&backtrack=1
```

Get cci values on each of the past X candles in one call
Let's say you want to know what the cci daily value was each day for the previous 10 days. You can get this returned by our API easily and efficiently in one call using the backtracks=10 parameter:
```
[GET] https://api.taapi.io/cci?secret=MY_SECRET&exchange=binance&symbol=BTC/USDT&interval=1d&backtracks=10
```
"""