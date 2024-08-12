import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime
import plotly.graph_objects as go

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

# Function to generate weighted signals
def generate_weighted_signals(indicators, moving_averages):
    weights = {
        'RSI': 0.2,
        'MACD': 0.3,
        'STOCH': 0.1,
        'ADX': 0.1,
        'CCI': 0.1,
        'ROC': 0.1,
        'WILLIAMSR': 0.1
    }
    signals = {}
    for indicator, value in indicators.items():
        if value > 0:
            signals[indicator] = 'Buy'
        elif value < 0:
            signals[indicator] = 'Sell'
        else:
            signals[indicator] = 'Neutral'

    weighted_score = sum([weights[indicator] for indicator, signal in signals.items() if signal == 'Buy']) - \
                     sum([weights[indicator] for indicator, signal in signals.items() if signal == 'Sell'])
    
    # Include moving averages in the signal if needed
    for ma in moving_averages:
        signals[ma] = 'Neutral'

    return signals, weighted_score

# Function to log signals with additional details
def log_signals(signals, decision, entry_point_long, entry_point_short, take_profit, stop_loss, weighted_score):
    log_file = 'signals_log.csv'
    try:
        logs = pd.read_csv(log_file)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        logs = pd.DataFrame(columns=['timestamp', 'RSI', 'MACD', 'ADX', 'CCI', 'MA', 'Entry Point Long', 'Entry Point Short', 'Take Profit', 'Stop Loss', 'Decision', 'Weighted Score'])
    
    # Add new log
    new_log = pd.DataFrame([{
        'timestamp': datetime.now(est).strftime('%Y-%m-%d %H:%M:%S'),
        'RSI': signals.get('RSI', 'N/A'),
        'MACD': signals.get('MACD', 'N/A'),
        'ADX': signals.get('ADX', 'N/A'),
        'CCI': signals.get('CCI', 'N/A'),
        'MA': signals.get('MA', 'N/A'),  # Adjusted for potential MA missing keys
        'Entry Point Long': entry_point_long,
        'Entry Point Short': entry_point_short,
        'Take Profit': take_profit,
        'Stop Loss': stop_loss,
        'Decision': decision,
        'Weighted Score': weighted_score
    }])
    logs = pd.concat([new_log, logs], ignore_index=True)
    logs.to_csv(log_file, index=False)

# Function to generate a perpetual options decision
def generate_perpetual_options_decision(indicators, moving_averages, data, account_balance):
    signals, weighted_score = generate_weighted_signals(indicators, moving_averages)
    
    if not isinstance(signals, dict):
        st.error("Error: Signals is not a dictionary.")
        return 'Error', 0, 0, 0, 0

    buy_signals = [value for key, value in signals.items() if value == 'Buy']
    sell_signals = [value for key, value in signals.items() if value == 'Sell']
    
    if len(buy_signals) > len(sell_signals):
        decision = 'Go Long'
    elif len(sell_signals) > len(buy_signals):
        decision = 'Go Short'
    else:
        decision = 'Neutral'
    
    # Modified take profit and stop loss calculations
    take_profit_pct = 0.02 if decision == 'Go Long' else -0.02
    stop_loss_pct = -0.01 if decision == 'Go Long' else 0.01
    
    entry_point_long = data['Close'].iloc[-1] * 1.001  # Entry point for long trade, 0.1% above current price
    entry_point_short = data['Close'].iloc[-1] * 0.999  # Entry point for short trade, 0.1% below current price
    
    take_profit = data['Close'].iloc[-1] * (1 + take_profit_pct)
    stop_loss = data['Close'].iloc[-1] * (1 + stop_loss_pct)
    
    log_signals(signals, decision, entry_point_long, entry_point_short, take_profit, stop_loss, weighted_score)
    
    return decision, entry_point_long, entry_point_short, take_profit, stop_loss

# Main app logic
def main():
    st.title("Bitcoin Trading Signals")

    data = fetch_data(ticker)
    if data is not None:
        data = calculate_indicators(data)
        data = calculate_support_resistance(data)
        data = detect_doji(data)

        indicators = technical_indicators_summary(data)
        moving_averages = moving_averages_summary(data)

        account_balance = st.number_input("Enter your account balance", min_value=0.0, value=1000.0, step=10.0)

        decision, entry_point_long, entry_point_short, take_profit, stop_loss = generate_perpetual_options_decision(indicators, moving_averages, data, account_balance)

        st.subheader("Technical Indicators Summary")
        st.write(indicators)

        st.subheader("Moving Averages Summary")
        st.write(moving_averages)

        st.subheader("Decision")
        st.write(f"Decision: {decision}")
        st.write(f"Entry Point Long: {entry_point_long:.2f}")
        st.write(f"Entry Point Short: {entry_point_short:.2f}")
        st.write(f"Take Profit: {take_profit:.2f}")
        st.write(f"Stop Loss: {stop_loss:.2f}")

        st.subheader("Live Bitcoin Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Candlestick'))
        fig.update_layout(title='Live Bitcoin Price Chart',
                          xaxis_title='Date',
                          yaxis_title='Price (USD)')
        st.plotly_chart(fig)

    if st.button('Refresh'):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
