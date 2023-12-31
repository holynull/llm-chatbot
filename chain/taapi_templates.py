GENERATE_RSI_QUESTION="""The user's input is to ask for suggestions on the purchase or sale of a certain cryptocurrency. The RSI (Relative Strength Index) will provide the market signal of purchase or sale. Please change the user's input into a question about what the RSI of the cryptocurrency involved in the user's input is. If you don't know how to change, please answer that you don't know.
User's input:
{input}
Question:
"""
RSI_CONCLUSION="""RSI:
The relative strength index (RSI) is a technical indicator used in the analysis of financial markets. It is intended to chart the current and historical strength or weakness of a stock or market based on the closing prices of a recent trading period. The indicator should not be confused with relative strength.
The RSI is classified as a momentum oscillator, measuring the velocity and magnitude of price movements. Momentum is the rate of the rise or fall in price. The relative strength RS is given as the ratio of higher closes to lower closes. Concretely, one computes two averages of absolute values of closing price changes, i.e. two sums involving the sizes of candles in a candle chart. The RSI computes momentum as the ratio of higher closes to overall closes: stocks which have had more or stronger positive changes have a higher RSI than stocks which have had more or stronger negative changes.
The RSI is most typically used on a 14-day timeframe, measured on a scale from 0 to 100, with high and low levels marked at 70 and 30, respectively. Short or longer timeframes are used for alternately shorter or longer outlooks. High and low levels—80 and 20, or 90 and 10—occur less frequently but indicate stronger momentum.
The relative strength index was developed by J. Welles Wilder and published in a 1978 book, New Concepts in Technical Trading Systems, and in Commodities magazine (now Modern Trader magazine) in the June 1978 issue.[1] It has become one of the most popular oscillator indices.
The RSI provides signals that tell investors to buy when the security or currency is oversold and to sell when it is overbought. 
RSI with recommended parameters and its day-to-day optimization was tested and compared with other strategies in Marek and Šedivá (2017). The testing was randomised in time and companies (e.g., Apple, Exxon Mobil, IBM, Microsoft) and showed that RSI can still produce good results; however, in longer time it is usually overcome by the simple buy-and-hold strategy. 

RSI Interpretation:
Basic configuration: The RSI is presented on a graph above or below the price chart. The indicator has an upper line, typically at 70, a lower line at 30, and a dashed mid-line at 50. Wilder recommended a smoothing period of 14 (see exponential smoothing, i.e. α = 1/14 or N = 14).
Principles: Wilder posited that when price moves up very rapidly, at some point it is considered overbought. Likewise, when price falls very rapidly, at some point it is considered oversold. In either case, Wilder deemed a reaction or reversal imminent.The level of the RSI is a measure of the stock's recent trading strength. The slope of the RSI is directly proportional to the velocity of a change in the trend. The distance traveled by the RSI is proportional to the magnitude of the move.Wilder believed that tops and bottoms are indicated when RSI goes above 70 or drops below 30. Traditionally, RSI readings greater than the 70 level are considered to be in overbought territory, and RSI readings lower than the 30 level are considered to be in oversold territory. In between the 30 and 70 level is considered neutral, with the 50 level a sign of no trend. 
Divergence: Wilder further believed that divergence between RSI and price action is a very strong indication that a market turning point is imminent. Bearish divergence occurs when price makes a new high but the RSI makes a lower high, thus failing to confirm. Bullish divergence occurs when price makes a new low but RSI makes a higher low.
Overbought and oversold conditions: Wilder thought that "failure swings" above 50 and below 50 on the RSI are strong indications of market reversals.[6] For example, assume the RSI hits 76, pulls back to 72, then rises to 77. If it falls below 72, Wilder would consider this a "failure swing" above 70.Finally, Wilder wrote that chart formations and areas of support and resistance could sometimes be more easily seen on the RSI chart as opposed to the price chart. The center line for the relative strength index is 50, which is often seen as both the support and resistance line for the indicator. If the relative strength index is below 50, it generally means that the stock's losses are greater than the gains. When the relative strength index is above 50, it generally means that the gains are greater than the losses.
Uptrends and downtrends: n addition to Wilder's original theories of RSI interpretation, Andrew Cardwell has developed several new interpretations of RSI to help determine and confirm trend. First, Cardwell noticed that uptrends generally traded between RSI 40 and 80, while downtrends usually traded between RSI 60 and 20. Cardwell observed when securities change from uptrend to downtrend and vice versa, the RSI will undergo a "range shift." Next, Cardwell noted that bearish divergence: 1) only occurs in uptrends, and 2) mostly only leads to a brief correction instead of a reversal in trend. Therefore, bearish divergence is a sign confirming an uptrend. Similarly, bullish divergence is a sign confirming a downtrend.
Reversals: Finally, Cardwell discovered the existence of positive and negative reversals in the RSI. Reversals are the opposite of divergence. For example, a positive reversal occurs when an uptrend price correction results in a higher low compared to the last price correction, while RSI results in a lower low compared to the prior correction. A negative reversal happens when a downtrend rally results in a lower high compared to the last downtrend rally, but RSI makes a higher high compared to the prior rally. In other words, despite stronger momentum as seen by the higher high or lower low in the RSI, price could not make a higher high or lower low. This is evidence the main trend is about to resume. Cardwell noted that positive reversals only happen in uptrends while negative reversals only occur in downtrends, and therefore their existence confirms the trend.

Content:
{data}

The analysis conclusion of market trend in the content is as follows:"""

GENERATE_CCI_QUESTION="""The user's input is to ask for suggestions on the purchase or sale of a certain cryptocurrency. The CCI (Commodity Channel Index) will provide the market signal of purchase or sale. Please change the user's input into a question about what the CCI of the cryptocurrency involved in the user's input is. If you don't know how to change, please answer that you don't know.
User's input:
{input}
Question:
"""
CCI_CONCLUSION="""CCI:
The commodity channel index (CCI) is an oscillator originally introduced by Donald Lambert in 1980.
Since its introduction, the indicator has grown in popularity and is now a very common tool for traders in identifying cyclical trends not only in commodities but also equities and currencies. The CCI can be adjusted to the timeframe of the market traded on by changing the averaging period.

Calculation:
CCI measures a security’s variation from the statistical mean.
For scaling purposes, Lambert set the constant at 0.015 to ensure that approximately 70 to 80 percent of CCI values would fall between −100 and +100. The CCI fluctuates above and below zero. The percentage of CCI values that fall between +100 and −100 will depend on the number of periods used. A shorter CCI will be more volatile with a smaller percentage of values between +100 and −100. Conversely, the more periods used to calculate the CCI, the higher the percentage of values between +100 and −100.

Interpretation:
Traders and investors use the commodity channel index to help identify price reversals, price extremes and trend strength. As with most indicators, the CCI should be used in conjunction with other aspects of technical analysis. CCI fits into the momentum category of oscillators. In addition to momentum, volume indicators and the price chart may also influence a technical assessment. It is often used for detecting divergences from price trends as an overbought/oversold indicator, and to draw patterns on it and trade according to those patterns. In this respect, it is similar to bollinger bands, but is presented as an indicator rather than as overbought/oversold levels.
The CCI typically oscillates above and below a zero line. Normal oscillations will occur within the range of +100 and −100. Readings above +100 imply an overbought condition, while readings below −100 imply an oversold condition. As with other overbought/oversold indicators, this means that there is a large probability that the price will correct to more representative levels.
The CCI has seen substantial growth in popularity amongst technical investors; today's traders often use the indicator to determine cyclical trends in not only commodities, but also equities and currencies.
The CCI, when used in conjunction with other oscillators, can be a valuable tool to identify potential peaks and valleys in the asset's price, and thus provide investors with reasonable evidence to estimate changes in the direction of price movement of the asset.[3]
Lambert's trading guidelines for the CCI focused on movements above +100 and below −100 to generate buy and sell signals. Because about 70 to 80 percent of the CCI values are between +100 and −100, a buy or sell signal will be in force only 20 to 30 percent of the time. When the CCI moves above +100, a security is considered to be entering into a strong uptrend and a buy signal is given. The position should be closed when the CCI moves back below +100. When the CCI moves below −100, the security is considered to be in a strong downtrend and a sell signal is given. The position should be closed when the CCI moves back above −100.
Since Lambert's original guidelines, traders have also found the CCI valuable for identifying reversals. The CCI is a versatile indicator capable of producing a wide array of buy and sell signals.
- CCI can be used to identify overbought and oversold levels. A security would be deemed oversold when the CCI dips below −100 and overbought when it exceeds +100. From oversold levels, a buy signal might be given when the CCI moves back above −100. From overbought levels, a sell signal might be given when the CCI moved back below +100.
- As with most oscillators, divergences can also be applied to increase the robustness of signals. A positive divergence below −100 would increase the robustness of a signal based on a move back above −100. A negative divergence above +100 would increase the robustness of a signal based on a move back below +100.
- Trend line breaks can be used to generate signals. Trend lines can be drawn connecting the peaks and troughs. From oversold levels, an advance above −100 and trend line breakout could be considered bullish. From overbought levels, a decline below +100 and a trend line break could be considered bearish.

Content:
{data}

The analysis conclusion of market trend in the content is as follows:"""

GENERATE_DMI_QUESTION="""The user's input is to ask for suggestions on the purchase or sale of a certain cryptocurrency. The DMI (Directional Movement Index) will provide the market signal of purchase or sale. Please change the user's input into a question about what the DMI of the cryptocurrency involved in the user's input is. If you don't know how to change, please answer that you don't know.
User's input:
{input}
Question:
"""
DMI_CONCLUSION="""DMI:
Average directional movement index
The average directional movement index (ADX) was developed in 1978 by J. Welles Wilder as an indicator of trend strength in a series of prices of a financial instrument.[1] ADX has become a widely used indicator for technical analysts, and is provided as a standard in collections of indicators offered by various trading platforms.

Calculation:
The ADX is a combination of two other indicators developed by Wilder, the positive directional indicator (abbreviated +DI) and negative directional indicator (-DI).[2] The ADX combines them and smooths the result with a smoothed moving average.

To calculate +DI and -DI, one needs price data consisting of high, low, and closing prices each period (typically each day). One first calculates the directional movement (+DM and -DM):

UpMove = today's high − yesterday's high
DownMove = yesterday's low − today's low
if UpMove > DownMove and UpMove > 0, then +DM = UpMove, else +DM = 0
if DownMove > UpMove and DownMove > 0, then -DM = DownMove, else -DM = 0
After selecting the number of periods (Wilder used 14 days originally), +DI and -DI are:

+DI = 100 times the smoothed moving average of (+DM) divided by average true range
-DI = 100 times the smoothed moving average of (-DM) divided by average true range
The smoothed moving average is calculated over the number of periods selected, and the average true range is a smoothed average of the true ranges. Then:

ADX = 100 times the smoothed moving average of the absolute value of (+DI − -DI) divided by (+DI + -DI)
Variations of this calculation typically involve using different types of moving averages, such as an exponential moving average, a weighted moving average or an adaptive moving average.[3]

Interpretation:
The ADX does not indicate trend direction or momentum, only trend strength.[4] It is a lagging indicator; that is, a trend must have established itself before the ADX will generate a signal that a trend is under way. ADX will range between 0 and 100. Generally, ADX readings below 20 indicate trend weakness, and readings above 40 indicate trend strength. An extremely strong trend is indicated by readings above 50. Alternative interpretations have also been proposed and accepted among technical analysts. For example it has been shown how ADX is a reliable coincident indicator of classical chart pattern development, whereby ADX readings below 20 occur just prior to pattern breakouts.[5] The value of the ADX is proportional to the slope of the trend. The slope of the ADX line is proportional to the acceleration of the price movement (changing trend slope). If the trend is a constant slope then the ADX value tends to flatten out.[6]

Timing:
Various market timing methods have been devised using ADX. One of these methods is discussed by Alexander Elder in his book Trading for a Living. One of the best buy signals is when ADX turns up when below both Directional Lines and +DI is above -DI. You would sell when ADX turns back down.[7]

Content:
{data}

The analysis conclusion of market trend in the content is as follows:"""

GENERATE_MACD_QUESTION="""The user's input is to ask for suggestions on the purchase or sale of a certain cryptocurrency. The MACD (Moving Average Convergence Divergence) will provide the market signal of purchase or sale. Please change the user's input into a question about what the MACD of the cryptocurrency involved in the user's input is. If you don't know how to change, please answer that you don't know.
User's input:
{input}
Question:
"""
MACD_CONCLUSION="""MACD:
Moving Average Convergence/Divergence
This article is about stock price analysis. For other uses, see MACD (disambiguation).
Example of historical stock price data (top half) with the typical presentation of a MACD(12,26,9) indicator (bottom half). The blue line is the MACD series proper, the difference between the 12-day and 26-day EMAs of the price. The red line is the average or signal series, a 9-day EMA of the MACD series. The bar graph shows the divergence series, the difference of those two lines.
MACD, short for moving average convergence/divergence, is a trading indicator used in technical analysis of securities prices, created by Gerald Appel in the late 1970s.[1] It is designed to reveal changes in the strength, direction, momentum, and duration of a trend in a stock's price.
The MACD indicator[2] (or "oscillator") is a collection of three time series calculated from historical price data, most often the closing price. These three series are: the MACD series proper, the "signal" or "average" series, and the "divergence" series which is the difference between the two. The MACD series is the difference between a "fast" (short period) exponential moving average (EMA), and a "slow" (longer period) EMA of the price series. The average series is an EMA of the MACD series itself.
The MACD indicator thus depends on three time parameters, namely the time constants of the three EMAs. The notation "MACD(a,b,c)" usually denotes the indicator where the MACD series is the difference of EMAs with characteristic times a and b, and the average series is an EMA of the MACD series with characteristic time c. These parameters are usually measured in days. The most commonly used values are 12, 26, and 9 days, that is, MACD(12,26,9). As true with most of the technical indicators, MACD also finds its period settings from the old days when technical analysis used to be mainly based on the daily charts. The reason was the lack of the modern trading platforms which show the changing prices every moment. As the working week used to be 6-days, the period settings of (12, 26, 9) represent 2 weeks, 1 month and one and a half week. Now when the trading weeks have only 5 days, possibilities of changing the period settings cannot be overruled. However, it is always better to stick to the period settings which are used by the majority of traders as the buying and selling decisions based on the standard settings further push the prices in that direction.
Although the MACD and average series are discrete values in nature, but they are customarily displayed as continuous lines in a plot whose horizontal axis is time, whereas the divergence is shown as a bar chart (often called a histogram).
MACD indicator showing vertical lines (histogram)
A fast EMA responds more quickly than a slow EMA to recent changes in a stock's price. By comparing EMAs of different periods, the MACD series can indicate changes in the trend of a stock. It is claimed that the divergence series can reveal subtle shifts in the stock's trend.
Since the MACD is based on moving averages, it is a lagging indicator. As a future metric of price trends, the MACD is less useful for stocks that are not trending (trading in a range) or are trading with unpredictable price action. Hence the trends will already be completed or almost done by the time MACD shows the trend.

Terminology:
Over the years, elements of the MACD have become known by multiple and often over-loaded terms. The common definitions of particularly overloaded terms are:
Divergence:
1. As the D in MACD, "divergence" refers to the two underlying moving averages drifting apart, while "convergence" refers to the two underlying moving averages coming towards each other.
2. Gerald Appel referred to a "divergence" as the situation where the MACD line does not conform to the price movement, e.g. a price low is not accompanied by a low of the MACD.[3]
3. Thomas Asprey dubbed the difference between the MACD and its signal line the "divergence" series.
In practice, definition number 2 above is often preferred.
Histogram:
1. Gerald Appel referred to bar graph plots of the basic MACD time series as "histogram". In Appel's Histogram the height of the bar corresponds to the MACD value for a particular point in time.
2. The difference between the MACD and its Signal line is often plotted as a bar chart and called a "histogram".
In practice, definition number 2 above is often preferred.

Mathematical interpretation:
In signal processing terms, the MACD series is a filtered measure of the derivative of the input (price) series with respect to time. (The derivative is called "velocity" in technical stock analysis.) MACD estimates the derivative as if it were calculated and then filtered by the two low-pass filters in tandem, multiplied by a "gain" equal to the difference in their time constants. It also can be seen to approximate the derivative as if it were calculated and then filtered by a single low pass exponential filter (EMA) with time constant equal to the sum of time constants of the two filters, multiplied by the same gain.[6] So, for the standard MACD filter time constants of 12 and 26 days, the MACD derivative estimate is filtered approximately by the equivalent of a low-pass EMA filter of 38 days. The time derivative estimate (per day) is the MACD value divided by 14.
The average series is also a derivative estimate, with an additional low-pass filter in tandem for further smoothing (and additional lag). The difference between the MACD series and the average series (the divergence series) represents a measure of the second derivative of price with respect to time ("acceleration" in technical stock analysis). This estimate has the additional lag of the signal filter and an additional gain factor equal to the signal filter constant.
Classification:
The MACD can be classified as an absolute price oscillator (APO), because it deals with the actual prices of moving averages rather than percentage changes. A percentage price oscillator (PPO), on the other hand, computes the difference between two moving averages of price divided by the longer moving average value.
While an APO will show greater levels for higher priced securities and smaller levels for lower priced securities, a PPO calculates changes relative to price. Subsequently, a PPO is preferred when: comparing oscillator values between different securities, especially those with substantially different prices; or comparing oscillator values for the same security at significantly different times, especially a security whose value has changed greatly.
Another member of the price oscillator family is the detrended price oscillator (DPO), which ignores long term trends while emphasizing short term patterns.
Trading interpretation:
Exponential moving averages highlight recent changes in a stock's price. By comparing EMAs of different lengths, the MACD series gauges changes in the trend of a stock. The difference between the MACD series and its average is claimed to reveal subtle shifts in the strength and direction of a stock's trend. It may be necessary to correlate the signals with the MACD to indicators like RSI power.
Some traders attribute special significance to the MACD line crossing the signal line, or the MACD line crossing the zero axis. Significance is also attributed to disagreements between the MACD line or the difference line and the stock price (specifically, higher highs or lower lows on the price series that are not matched in the indicator series).
- Signal-line crossover
A "signal-line crossover" occurs when the MACD and average lines cross; that is, when the divergence (the bar graph) changes sign. The standard interpretation of such an event is a recommendation to buy if the MACD line crosses up through the average line (a "bullish" crossover), or to sell if it crosses down through the average line (a "bearish" crossover).[7] These events are taken as indications that the trend in the stock is about to accelerate in the direction of the crossover.
- Zero crossover
A "zero crossover" event occurs when the MACD series changes sign, that is, the MACD line crosses the horizontal zero axis. This happens when there is no difference between the fast and slow EMAs of the price series. A change from positive to negative MACD is interpreted as "bearish", and from negative to positive as "bullish". Zero crossovers provide evidence of a change in the direction of a trend but less confirmation of its momentum than a signal line crossover.
- Divergence:
A "positive divergence" or "bullish divergence" occurs when the price makes a new low but the MACD does not confirm with a new low of its own. A "negative divergence" or "bearish divergence" occurs when the price makes a new high but the MACD does not confirm with a new high of its own.[8] A divergence with respect to price may occur on the MACD line and/or the MACD Histogram.[9]
- Timing
The MACD is only as useful as the context in which it is applied. An analyst might apply the MACD to a weekly scale before looking at a daily scale, in order to avoid making short term trades against the direction of the intermediate trend.[10] Analysts will also vary the parameters of the MACD to track trends of varying duration. One popular short-term set-up, for example, is the (5,35,5).
- False signals
Like any forecasting algorithm, the MACD can generate false signals. A false positive, for example, would be a bullish crossover followed by a sudden decline in a stock. A false negative would be a situation where there is bearish crossover, yet the stock accelerated suddenly upwards.
A prudent strategy may be to apply a filter to signal line crossovers to ensure that they have held up. An example of a price filter would be to buy if the MACD line breaks above the signal line and then remains above it for three days. As with any filtering strategy, this reduces the probability of false signals but increases the frequency of missed profit.
Analysts use a variety of approaches to filter out false signals and confirm true ones.
A MACD crossover of the signal line indicates that the direction of the acceleration is changing. The MACD line crossing zero suggests that the average velocity is changing direction.

Content:
{data}

The analysis conclusion of market trend in the content is as follows:"""

GENERATE_PSAR_QUESTION="""The user's input is to ask for suggestions on the purchase or sale of a certain cryptocurrency. The PSAR (Parabolic SAR) will provide the market signal of purchase or sale. Please change the user's input into a question about what the PSAR of the cryptocurrency involved in the user's input is. If you don't know how to change, please answer that you don't know.
User's input:
{input}
Question:
"""
PSAR_CONCLUSION="""PSAR:
Parabolic SAR
In stock and securities market technical analysis, parabolic SAR (parabolic stop and reverse) is a method devised by J. Welles Wilder Jr., to find potential reversals in the market price direction of traded goods such as securities or currency exchanges such as forex.[1] It is a trend-following (lagging) indicator and may be used to set a trailing stop loss or determine entry or exit points based on prices tending to stay within a parabolic curve during a strong trend.
Similar to option theory's concept of time decay, the concept draws on the idea that "time is the enemy". Thus, unless a security can continue to generate more profits over time, it should be liquidated. The indicator generally works only in trending markets, and creates "whipsaws" during ranging or, sideways phases. Therefore, Wilder recommends first establishing the direction or change in direction of the trend through the use of parabolic SAR, and then using a different indicator such as the Average Directional Index to determine the strength of the trend.
A parabola below the price is generally bullish, while a parabola above is generally bearish. A parabola below the price may be used as support, whereas a parabola above the price may represent resistance.[2]

== Construction ==

The parabolic SAR is calculated almost independently for each trend in the price. When the price is in an uptrend, the SAR emerges below the price and converges upwards towards it. Similarly, on a downtrend, the SAR emerges above the price and converges downwards.
At each step within a trend, the SAR is calculated one period in advance. That is, tomorrow's SAR value is built using data available today. The general formula used for this is:

<math>{{SAR}}_{{n+1}} = {{SAR}}_n + \alpha ( EP - {{SAR}}_n )</math>,

where ''SAR<sub>n</sub>'' and ''SAR<sub>n+1</sub>'' represent the current period and the next period's SAR values, respectively.

''EP'' (the extreme point) is a record kept during each trend that represents the highest value reached by the price during the current uptrend – or lowest value during a downtrend. During each period, if a new maximum (or minimum) is observed, the EP is updated with that value.

The ''α'' value represents the acceleration factor. Usually, this is set initially to a value of 0.02, but can be chosen by the trader. This factor is increased by 0.02 each time a new EP is recorded, which means that every time a new EP is observed, it will make the acceleration factor go up. The rate will then quicken to a point where the SAR converges towards the price. To prevent it from getting too large, a maximum value for the acceleration factor is normally set to 0.20. The traders can set these numbers depending on their trading style and the instruments being traded. Generally, it is preferable in stocks trading to set the acceleration factor to 0.01, so that it is not too sensitive to local decreases. For commodity or currency trading, the preferred value is 0.02.

The SAR is calculated in this manner for each new period. However, two special cases will modify the SAR value:
* If the next period's SAR value is inside (or beyond) the current period or the previous period's price range, the SAR must be set to the closest price bound. For example, if in an upward trend, the new SAR value is calculated and if it results to be more than today's or yesterday's lowest price, it must be set equal to that lower boundary.
* If the next period's SAR value is inside (or beyond) the next period's price range, a new trend direction is then signaled. The SAR must then switch sides.

Upon a trend switch, the first SAR value for this new trend is set to the last EP recorded on the prior trend, EP is then reset accordingly to this period's maximum, and the acceleration factor is reset to its initial value of 0.02.

Statistical Results
The parabolic SAR showed results at a 95% confidence level in a study of 17 years of data.[3]

Content:
{data}

The analysis conclusion of market trend in the content is as follows:"""

GENERATE_STOCHRSI_QUESTION="""The user's input is to ask for suggestions on the purchase or sale of a certain cryptocurrency. The STOCHRSI (Stochastic Relative Strength Index) will provide the market signal of purchase or sale. Please change the user's input into a question about what the STOCHRSI of the cryptocurrency involved in the user's input is. If you don't know how to change, please answer that you don't know.
User's input:
{input}
Question:
"""

GENERATE_CMF_QUESTION="""The user's input is to ask for suggestions on the purchase or sale of a certain cryptocurrency. The CMF (Chaikin Money Flow) will provide the market signal of purchase or sale. Please change the user's input into a question about what the CMF of the cryptocurrency involved in the user's input is. If you don't know how to change, please answer that you don't know.
User's input:
{input}
Question:
"""