import yfinance as yf
import pandas as pd
import numpy as np

# Download 1-minute intraday data for GRASIM.NS (last 7 days max)
df = yf.download(tickers='GRASIM.NS', period='7d', interval='1m')

# Reset index to get datetime in a column
df = df.reset_index()

# Use 'Close' as price and 'Volume' as volume proxy
df = df[['Datetime', 'Close', 'Volume']]
df.columns = ['timestamp', 'price', 'volume']

# Compute tick direction using tick rule (+1 for uptick, -1 for downtick)
def tick_rule(prices):
    directions = [0]
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            directions.append(1)
        elif prices[i] < prices[i-1]:
            directions.append(-1)
        else:
            directions.append(directions[-1])
    return directions

df['tick_direction'] = tick_rule(df['price'].values)

# Calculate signed volume for imbalance
df['signed_volume'] = df['tick_direction'] * df['volume']

# Parameters for dynamic threshold
ewma_span = 10  # shorter due to less granular data
threshold_multiplier = 5

# EWMA of absolute signed volume
ewma_abs = df['signed_volume'].abs().ewm(span=ewma_span, adjust=False).mean().values

imbalance_bars = []
current_bar = []
current_bar_cum_imbalance = 0

for i in range(len(df)):
    signed_vol = df.at[i, 'signed_volume']
    current_bar.append(df.iloc[i])
    current_bar_cum_imbalance += signed_vol
    threshold = threshold_multiplier * ewma_abs[i]
    if abs(current_bar_cum_imbalance) > threshold:
        bar_df = pd.DataFrame(current_bar)
        imbalance_bars.append({
            'start_time': bar_df['timestamp'].iloc[0],
            'end_time': bar_df['timestamp'].iloc[-1],
            'open': bar_df['price'].iloc[0],
            'close': bar_df['price'].iloc[-1],
            'high': bar_df['price'].max(),
            'low': bar_df['price'].min(),
            'volume': bar_df['volume'].sum(),
            'cum_imbalance': current_bar_cum_imbalance
        })
        current_bar = []
        current_bar_cum_imbalance = 0

imbalance_bars_df = pd.DataFrame(imbalance_bars)
print(imbalance_bars_df.head())

imbalance_bars_df.to_csv('grasim_tick_imbalance_bars.csv', index=False)
