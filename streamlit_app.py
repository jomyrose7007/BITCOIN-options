import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime
import plotly.graph_objects as go
import requests
from sklearn.metrics import f1_score, matthews_corrcoef

# Define constants
TICKER = 'BTC-USD'
TIMEZONE = pytz.timezone('America/New_York')
LOG_FILE = 'signals_log.csv'

# Function to convert datetime to EST
def to_est(dt):
    return dt.tz_convert(TIMEZONE) if dt.tzinfo else TIMEZONE.localize(dt)

# Fetch live data from Yahoo Finance
def fetch_data(ticker):
    try:
        data = yf.download(ticker, period='1d', interval='1m')
        if data.index.tzinfo is None:
            data.index = data.index.tz_localize(pytz.utc).tz_convert(TIMEZONE)
        else:
            data.index = data.index.tz_convert(TIMEZONE)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

data = fetch_data(TICKER)

# Check if data is available
if data.empty:
    st.stop()

# Calculate technical indicators
def calculate_indicators(data):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
    data['STOCH'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
    data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    data['ROC'] = ta.momentum.ROCIndicator(data['Close']).roc()
    data['WILLIAMSR'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
    return data

data = calculate_indicators(data)
data.dropna(inplace=True)  # Drop rows with NaN values

# Calculate Fibonacci retracement levels
def fibonacci_retracement(high, low):
    diff = high - low
    return [high - diff * ratio for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]]

high = data['High'].max()
low = data['Low'].min()
fib_levels = fibonacci_retracement(high, low)

# Detect Doji candlestick patterns
def detect_doji(data):
    threshold = 0.001
    data['Doji'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low']) < threshold
    return data

data = detect_doji(data)

# Calculate support and resistance levels
def calculate_support_resistance(data, window=5):
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

data = calculate_support_resistance(data)

# Plot support and resistance levels
def plot_support_resistance(data, fib_levels):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Support'], name='Support', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Resistance'], name='Resistance', line=dict(dash='dash')))
    for level in fib_levels:
        fig.add_trace(go.Scatter(x=[data.index.min(), data.index.max()], y=[level, level], mode='lines', name=f'Fibonacci Level {level:.4f}', line=dict(dash='dot')))
    fig.update_layout(title='Support and Resistance Levels', xaxis_title='Time', yaxis_title='Price')
    return fig

st.plotly_chart(plot_support_resistance(data, fib_levels))

# Generate summary of technical indicators
def technical_indicators_summary(data):
    return {
        'RSI': data['RSI'].iloc[-1],
        'MACD': data['MACD'].iloc[-1] - data['MACD_Signal'].iloc[-1],
        'STOCH': data['STOCH'].iloc[-1],
        'ADX': data['ADX'].iloc[-1],
        'CCI': data['CCI'].iloc[-1],
        'ROC': data['ROC'].iloc[-1],
        'WILLIAMSR': data['WILLIAMSR'].iloc[-1]
    }

indicators = technical_indicators_summary(data)

# Generate summary of moving averages
def moving_averages_summary(data):
    ma = {
        'MA5': data['Close'].rolling(window=5).mean().iloc[-1],
        'MA10': data['Close'].rolling(window=10).mean().iloc[-1],
        'MA20': data['Close'].rolling(window=20).mean().iloc[-1],
        'MA50': data['Close'].rolling(window=50).mean().iloc[-1],
        'MA100': data['Close'].rolling(window=100).mean().iloc[-1],
        'MA200': data['Close'].rolling(window=200).mean().iloc[-1]
    }
    return ma

moving_averages = moving_averages_summary(data)

# Generate buy/sell signals based on indicators and moving averages
def generate_signals(indicators, moving_averages, data):
    signals = {}
    signals['timestamp'] = to_est(data.index[-1]).strftime('%Y-%m-%d %I:%M:%S %p')

    # RSI Signal
    if indicators['RSI'] < 30:
        signals['RSI'] = 'Buy'
    elif indicators['RSI'] > 70:
        signals['RSI'] = 'Sell'
    else:
        signals['RSI'] = 'Neutral'

    # MACD Signal
    if indicators['MACD'] > 0:
        signals['MACD'] = 'Buy'
    else:
        signals['MACD'] = 'Sell'

    # ADX Signal
    if indicators['ADX'] > 25:
        signals['ADX'] = 'Buy'
    else:
        signals['ADX'] = 'Neutral'

    # CCI Signal
    if indicators['CCI'] > 100:
        signals['CCI'] = 'Buy'
    elif indicators['CCI'] < -100:
        signals['CCI'] = 'Sell'
    else:
        signals['CCI'] = 'Neutral'

    # Moving Averages Signal
    signals['MA'] = 'Buy' if moving_averages['MA5'] > moving_averages['MA10'] else 'Sell'

    return signals

signals = generate_signals(indicators, moving_averages, data)

# Calculate signal accuracy
def calculate_signal_accuracy(logs, signals):
    if logs.empty:
        return 'N/A'
    y_true = logs.iloc[-1][1:]  # Assuming logs contain columns for actual signals
    y_pred = pd.Series(signals).reindex(y_true.index, fill_value='Neutral')
    return {
        'F1 Score': f1_score(y_true, y_pred, average='weighted'),
        'Matthews Correlation Coefficient': matthews_corrcoef(y_true, y_pred)
    }

# Log signals
try:
    logs = pd.read_csv(LOG_FILE)
except FileNotFoundError:
    logs = pd.DataFrame(columns=['timestamp', 'RSI', 'MACD', 'ADX', 'CCI', 'MA'])

new_log = pd.DataFrame([signals])
logs = pd.concat([logs, new_log], ignore_index=True)
logs.to_csv(LOG_FILE, index=False)

# Fetch Fear and Greed Index
def fetch_fear_and_greed_index():
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        latest_data = data['data'][0]
        return latest_data['value'], latest_data['value_classification']
    except Exception as e:
        st.error(f"Error fetching Fear and Greed Index: {e}")
        return 'N/A', 'N/A'

fear_and_greed_value, fear_and_greed_classification = fetch_fear_and_greed_index()

# Generate a perpetual options decision
def generate_perpetual_options_decision(indicators, moving_averages, fib_levels, current_price):
    decision = 'Neutral'
    resistance_levels = [fib_levels[3], fib_levels[4], high]
    
    # Check if current price is near any resistance level
    if any([current_price >= level for level in resistance_levels]):
        decision = 'Go Short'
    else:
        buy_signals = [value for value in generate_signals(indicators, moving_averages, data).values() if value == 'Buy']
        sell_signals = [value for value in generate_signals(indicators, moving_averages, data).values() if value == 'Sell']
        
        if len(buy_signals) > len(sell_signals):
            decision = 'Go Long'
        elif len(sell_signals) > len(buy_signals):
            decision = 'Go Short'
    
    return decision

current_price = data['Close'].iloc[-1]
perpetual_options_decision = generate_perpetual_options_decision(indicators, moving_averages, fib_levels, current_price)

# Determine entry point
def determine_entry_point(signals):
    entry_point = 'N/A'
    if signals['RSI'] == 'Buy' and signals['MACD'] == 'Buy' and signals['ADX'] == 'Buy':
        entry_point = 'Buy Now'
    elif signals['RSI'] == 'Sell' and signals['MACD'] == 'Sell' and signals['ADX'] == 'Sell':
        entry_point = 'Sell Now'
    return entry_point

entry_point = determine_entry_point(signals)

# Display results
st.write(f"### Technical Indicators Summary")
st.write(indicators)
st.write(f"### Moving Averages Summary")
st.write(moving_averages)
st.write(f"### Buy/Sell Signals")
st.write(signals)
st.write(f"### Signal Accuracy")
st.write(calculate_signal_accuracy(logs, signals))
st.write(f"### Fear and Greed Index")
st.write(f"Value: {fear_and_greed_value}")
st.write(f"Classification: {fear_and_greed_classification}")
st.write(f"### Perpetual Options Decision")
st.write(perpetual_options_decision)
st.write(f"### Entry Point")
st.write(entry_point)
