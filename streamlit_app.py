import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime
import plotly.graph_objects as go
import requests

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

# Function to convert datetime to EST
def to_est(dt):
    if dt.tzinfo is None:
        dt = est.localize(dt)
    return dt.astimezone(est)

# Function to fetch live data from Yahoo Finance
@st.cache_data(ttl=30)
def fetch_data(ticker):
    try:
        data = yf.download(ticker, period='1d', interval='1m')
        if data.empty:
            st.error("No data fetched from Yahoo Finance.")
            return None

        if data.index.tzinfo is None:
            data.index = data.index.tz_localize(pytz.utc).tz_convert(est)
        else:
            data.index = data.index.tz_convert(est)

        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to calculate technical indicators
def calculate_indicators(data):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
    data['STOCH'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
    data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    data['ROC'] = ta.momentum.ROCIndicator(data['Close']).roc()
    data['WILLIAMSR'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
    data.dropna(inplace=True)
    return data

# Function to calculate support and resistance levels
def calculate_support_resistance(data, window=5):
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

# Function to detect Doji candlestick patterns
def detect_doji(data, threshold=0.1):
    data['Doji'] = np.where(
        (data['Close'] - data['Open']).abs() / (data['High'] - data['Low']) < threshold,
        'Yes',
        'No'
    )
    return data

# Function to generate summary of technical indicators
def technical_indicators_summary(data):
    indicators = {
        'RSI': data['RSI'].iloc[-1],
        'MACD': data['MACD'].iloc[-1] - data['MACD_Signal'].iloc[-1],
        'STOCH': data['STOCH'].iloc[-1],
        'ADX': data['ADX'].iloc[-1],
        'CCI': data['CCI'].iloc[-1],
        'ROC': data['ROC'].iloc[-1],
        'WILLIAMSR': data['WILLIAMSR'].iloc[-1]
    }
    return indicators

# Function to generate summary of moving averages
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

# Function to generate buy/sell signals based on indicators and moving averages
def generate_signals(indicators, moving_averages, data):
    signals = {}
    last_timestamp = to_est(data.index[-1]).strftime('%Y-%m-%d %I:%M:%S %p')
    signals['timestamp'] = last_timestamp
    
    # RSI Signal
    if indicators['RSI'] < 30:
        signals['RSI'] = 'Buy'
    elif indicators['RSI'] > 70:
        signals['RSI'] = 'Sell'
    else:
        signals['RSI'] = 'Neutral'
    
    # MACD Signal
    signals['MACD'] = 'Buy' if indicators['MACD'] > 0 else 'Sell'
    
    # ADX Signal
    signals['ADX'] = 'Buy' if indicators['ADX'] > 25 else 'Neutral'
    
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

# Function to log signals
def log_signals(signals):
    log_file = 'signals_log.csv'
    try:
        logs = pd.read_csv(log_file)
    except FileNotFoundError:
        logs = pd.DataFrame(columns=['timestamp', 'RSI', 'MACD', 'ADX', 'CCI', 'MA'])

    new_log = pd.DataFrame([signals])
    logs = pd.concat([new_log, logs], ignore_index=True)
    logs.to_csv(log_file, index=False)

# Function to fetch Fear and Greed Index
def fetch_fear_and_greed_index():
    url = "https://api.alternative.me/fng/"
    response = requests.get(url)
    data = response.json()
    latest_data = data['data'][0]
    return latest_data['value'], latest_data['value_classification']

# Function to generate a perpetual options decision
def generate_perpetual_options_decision(indicators, moving_averages, data):
    signals = generate_signals(indicators, moving_averages, data)
    
    # Decision logic
    buy_signals = [value for key, value in signals.items() if value == 'Buy']
    sell_signals = [value for key, value in signals.items() if value == 'Sell']
    
    if len(buy_signals) > len(sell_signals):
        decision = 'Go Long'
        take_profit_pct = 0.02
        stop_loss_pct = 0.01
    elif len(sell_signals) > len(buy_signals):
        decision = 'Go Short'
        take_profit_pct = -0.02  # Note the negative sign
        stop_loss_pct = 0.01
    else:
        decision = 'Neutral'
        take_profit_pct = 0
        stop_loss_pct = 0

    entry_point = data['Close'].iloc[-1]
    take_profit_level = entry_point * (1 + take_profit_pct)
    stop_loss_level = entry_point * (1 - stop_loss_pct)  # Corrected calculation

    return decision, entry_point, take_profit_level, stop_loss_level

# Function to calculate signal accuracy
def calculate_signal_accuracy(logs, signals):
    if logs.empty:
        return 'N/A'
    last_signal = logs.iloc[-1]
    accurate_signals = sum([last_signal.get(key) == signals.get(key) for key in signals])
    total_signals = len(signals)
    accuracy = (accurate_signals / total_signals) * 100
    return f"{accuracy:.2f}%"

# Main function to update data
def update_data():
    data = fetch_data(ticker)
    if data is None:
        st.stop()

    data = calculate_indicators(data)
    data = calculate_support_resistance(data)
    data = detect_doji(data)

    # Add chart to display support and resistance levels
    st.title('Bitcoin Technical Analysis and Signal Summary')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Support'], name='Support', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Resistance'], name='Resistance', line=dict(dash='dash')))
    fig.update_layout(title='Support and Resistance Levels', xaxis_title='Time', yaxis_title='Price')
    st.plotly_chart(fig)

    # Generate summaries
    indicators = technical_indicators_summary(data)
    ma = moving_averages_summary(data)
    signals = generate_signals(indicators, ma, data)
    log_signals(signals)
    
    # Add Fear and Greed Index
    fng_value, fng_class = fetch_fear_and_greed_index()
    st.write(f"Fear and Greed Index: {fng_value} ({fng_class})")

    # Generate decision for perpetual options
    decision, entry_point, take_profit, stop_loss = generate_perpetual_options_decision(indicators, ma, data)
    accuracy = calculate_signal_accuracy(pd.read_csv('signals_log.csv'), signals)

    st.write(f"Decision: {decision}")
    st.write(f"Entry Point: {entry_point}")
    st.write(f"Take Profit Level: {take_profit}")
    st.write(f"Stop Loss Level: {stop_loss}")
    st.write(f"Signal Accuracy: {accuracy}")

# Run the app
if __name__ == "__main__":
    update_data()
