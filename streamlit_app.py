import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import pytz
from datetime import datetime
import streamlit.components.v1 as components
from yahoo_fin import news as yf_news
import requests
from bs4 import BeautifulSoup

nltk.download('vader_lexicon')

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

# Function to convert datetime to EST
def to_est(dt):
    return dt.tz_convert(est) if dt.tzinfo else est.localize(dt)

# Fetch live data from Yahoo Finance
data = yf.download(ticker, period='1d', interval='1m')

# Convert index to EST if it's not already timezone-aware
if data.index.tzinfo is None:
    data.index = data.index.tz_localize(pytz.utc).tz_convert(est)
else:
    data.index = data.index.tz_convert(est)

# Calculate technical indicators
data['RSI'] = data['Close'].rolling(window=14).apply(lambda x: (x/x.shift(1)-1).mean())
data['BB_Middle'] = data['Close'].rolling(window=20).mean()
data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data['Stoch_OSC'] = (data['Close'] - data['Close'].rolling(window=14).min()) / (data['Close'].rolling(window=14).max() - data['Close'].rolling(window=14).min())
data['Force_Index'] = data['Close'].diff() * data['Volume']

# Perform sentiment analysis using nltk
sia = SentimentIntensityAnalyzer()
data['Sentiment'] = data['Close'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# Drop rows with NaN values
data.dropna(inplace=True)

# Define machine learning model using scikit-learn
X = pd.concat([data['Close'], data['RSI'], data['BB_Middle'], data['BB_Upper'], data['BB_Lower'], data['MACD'], data['Stoch_OSC'], data['Force_Index'], data['Sentiment']], axis=1)
y = data['Close'].shift(-1).dropna()

# Align X and y
X = X.iloc[:len(y)]  # Ensure X and y have the same number of rows

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define customizable parameters using streamlit
st.title('Bitcoin Model with Advanced Features')
st.write('Select parameters:')
n_estimators = st.slider('n_estimators', 1, 100, 50)
rsi_period = st.slider('RSI period', 1, 100, 14)
bb_period = st.slider('BB period', 1, 100, 20)
sentiment_threshold = st.slider('Sentiment threshold', -1.0, 1.0, 0.0)

# Train the model
model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)

# Generate buy/sell signals based on predictions
buy_sell_signals = np.where(predictions > X_test['Close'], 'BUY', 'SELL')

# Create a DataFrame for the signals
signals_df = pd.DataFrame({
    'Date': X_test.index,
    'Signal': buy_sell_signals,
    'Actual_Close': y_test,
    'Predicted_Close': predictions
})

# Convert 'Date' column to EST timezone
signals_df['Date'] = signals_df['Date'].apply(to_est)  # Convert dates to EST

# Create true labels based on actual market movement
signals_df['True_Label'] = np.where(signals_df['Actual_Close'].shift(-1) > signals_df['Actual_Close'], 'BUY', 'SELL')

# Calculate accuracy of the signals
accuracy = np.mean(signals_df['Signal'] == signals_df['True_Label'])

# Sort signals to show the latest first
signals_df = signals_df.sort_values(by='Date', ascending=False)

# Display decision section at the top
st.write('### Final Decision:')

# Initialize counters for bullish and bearish signals
bullish_signals = 0
bearish_signals = 0

# Check the latest signal and sentiment
latest_signal = signals_df.iloc[0]['Signal']
latest_sentiment = data['Sentiment'].iloc[-1]

# Count signals
if latest_signal == 'BUY':
    bullish_signals += 1
elif latest_signal == 'SELL':
    bearish_signals += 1

# Analyze sentiment
if latest_sentiment > sentiment_threshold:
    bullish_signals += 1
elif latest_sentiment < sentiment_threshold:
    bearish_signals += 1

# Include Fear and Greed Index in the decision
fear_and_greed_index = fetch_fear_and_greed()
if fear_and_greed_index > 50:
    bullish_signals += 1
elif fear_and_greed_index < 50:
    bearish_signals += 1

# Decision based on the count of signals, sentiment, and Fear and Greed Index
if bullish_signals > bearish_signals:
    decision = "BUY OPTION 🟢⬆️"
    reason = "The model suggests a buy signal, sentiment is positive, and the Fear and Greed Index indicates greed."
elif bearish_signals > bullish_signals:
    decision = "SELL OPTION 🔴⬇️"
    reason = "The model suggests a sell signal, sentiment is negative, and the Fear and Greed Index indicates fear."
else:
    decision = "HOLD OPTION 🟡"
    reason = "The signals are mixed; it's best to hold the current position."

st.write(f"### Decision: {decision}")
st.write(f"**Reason:** {reason}")

# Display the technical indicators
st.write('### Technical Indicators:')
st.write(f"RSI: {data['RSI'].iloc[-1]:.3f} - {'Buy 🟢' if data['RSI'].iloc[-1] < 30 else 'Sell 🔴' if data['RSI'].iloc[-1] > 70 else 'Neutral 🟡'}")
st.write(f"MACD: {data['MACD'].iloc[-1]:.3f} - {'Buy 🟢' if data['MACD'].iloc[-1] > 0 else 'Sell 🔴'}")
st.write(f"Bollinger Bands: Upper = {data['BB_Upper'].iloc[-1]:.3f}, Lower = {data['BB_Lower'].iloc[-1]:.3f}")

# Display buy/sell signals in a table
st.write('### Buy/Sell Signals:')
st.dataframe(signals_df[['Date', 'Signal', 'Actual_Close', 'Predicted_Close']])

# Plot the price chart
st.line_chart(data['Close'])

# Fetch latest news from Yahoo Finance
st.write('### Latest News:')
news = yf_news.get_yf_rss("BTC-USD")
for article in news:
    title = article.get('title', 'No title available')
    link = article.get('link', 'No link available')
    pub_date = article.get('pubDate', 'No publication date available')
    st.write(f"**{title}**")
    st.write(f"Published on: {pub_date}")
    st.write(f"Link: {link}")
    st.write("---")

# Fetch Fear and Greed Index from Alternative.me
def fetch_fear_and_greed():
    url = 'https://alternative.me/crypto/fear-and-greed-index/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    value = soup.find('div', {'class': 'fng-circle'}).text.strip()
    return int(value)

fear_and_greed_index = fetch_fear_and_greed()

st.write('### Fear and Greed Index:')
st.write(f"The current Fear and Greed Index is: **{fear_and_greed_index}**")

# Add JavaScript to auto-refresh the Streamlit app every 60 seconds
components.html("""
<script>
setTimeout(function(){
   window.location.reload();
}, 60000);  // Refresh every 60 seconds
</script>
""", height=0)
