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

The analysis conclusion of buying and selling signals in the content is as follows:"""

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

The analysis conclusion of buying and selling signals in the content is as follows:"""
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

The analysis conclusion of buying and selling signals in the content is as follows:"""