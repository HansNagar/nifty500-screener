# indicators.py
import pandas as pd

def supertrend(df, period=7, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    atr = (df['High'].rolling(period).max() - df['Low'].rolling(period).min()).rolling(period).mean()

    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    direction = [True] * len(df)
    for i in range(1, len(df)):
        if df['Close'].iat[i] > upperband.iat[i-1]:
            direction[i] = True
        elif df['Close'].iat[i] < lowerband.iat[i-1]:
            direction[i] = False
        else:
            direction[i] = direction[i-1]
            if direction[i] and lowerband.iat[i] < lowerband.iat[i-1]:
                lowerband.iat[i] = lowerband.iat[i-1]
            if not direction[i] and upperband.iat[i] > upperband.iat[i-1]:
                upperband.iat[i] = upperband.iat[i-1]

    df['Supertrend'] = direction
    df['ST_Upper']  = upperband
    df['ST_Lower']  = lowerband
    return df
