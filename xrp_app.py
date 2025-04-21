<<<<<<< HEAD
import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="XRP Price Predictor", layout="wide")
st.title("📈 XRP Price Prediction App")

# Sidebar Inputs
st.sidebar.header("Model Settings")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

# Fetch Data
@st.cache_data

def load_data():
    df = yf.download("XRP-USD", start=start_date, end=end_date)
    df['Return'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    df['Volatility'] = df['Return'].rolling(window=7).std()
    df.dropna(inplace=True)
    return df

xrp = load_data()
st.subheader("📊 Historical XRP Data")
st.dataframe(xrp.tail(10))

# Feature Selection
features = ['MA7', 'MA21', 'Volatility', 'Volume']
X = xrp[features]
y = xrp['Close'].shift(-1)

X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, shuffle=False)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")

# Prediction
xrp['Predicted_Close'] = model.predict(X)

# Visualization
st.subheader("📉 Actual vs Predicted Close Price")
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(xrp['Close'], label='Actual')
ax.plot(xrp['Predicted_Close'], label='Predicted')
ax.legend()
st.pyplot(fig)

# Trading Signal
xrp['Signal'] = xrp['Predicted_Close'] > xrp['Close']
st.subheader("💡 Buy Signal (True = Buy, False = Hold/Sell)")
st.dataframe(xrp[['Close', 'Predicted_Close', 'Signal']].tail(10))

# Real-Time Signal
st.subheader("🚨 Real-Time Trading Signal")

today = datetime.today().strftime('%Y-%m-%d')
yesterday = (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d')
latest_data = yf.download("XRP-USD", start=yesterday, end=today)

if not latest_data.empty:
    latest_close = latest_data['Close'][-1]
    latest_input = xrp[features].iloc[-1:].copy()
    latest_prediction = model.predict(latest_input)[0]

    signal = "BUY" if latest_prediction > latest_close else "HOLD/SELL"

    st.metric(label="Current XRP Price", value=f"${latest_close:.4f}")
    st.metric(label="Predicted Next Close", value=f"${latest_prediction:.4f}")
    st.markdown(f"### ✅ Suggested Action: **{signal}**")
else:
    st.warning("Could not fetch latest data. Try again later.")
=======
import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="XRP Price Predictor", layout="wide")
st.title("📈 XRP Price Prediction App")

# Sidebar Inputs
st.sidebar.header("Model Settings")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

# Fetch Data
@st.cache_data

def load_data():
    df = yf.download("XRP-USD", start=start_date, end=end_date)
    df['Return'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    df['Volatility'] = df['Return'].rolling(window=7).std()
    df.dropna(inplace=True)
    return df

xrp = load_data()
st.subheader("📊 Historical XRP Data")
st.dataframe(xrp.tail(10))

# Feature Selection
features = ['MA7', 'MA21', 'Volatility', 'Volume']
X = xrp[features]
y = xrp['Close'].shift(-1)

X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, shuffle=False)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")

# Prediction
xrp['Predicted_Close'] = model.predict(X)

# Visualization
st.subheader("📉 Actual vs Predicted Close Price")
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(xrp['Close'], label='Actual')
ax.plot(xrp['Predicted_Close'], label='Predicted')
ax.legend()
st.pyplot(fig)

# Trading Signal
xrp['Signal'] = xrp['Predicted_Close'] > xrp['Close']
st.subheader("💡 Buy Signal (True = Buy, False = Hold/Sell)")
st.dataframe(xrp[['Close', 'Predicted_Close', 'Signal']].tail(10))

# Real-Time Signal
st.subheader("🚨 Real-Time Trading Signal")

today = datetime.today().strftime('%Y-%m-%d')
yesterday = (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d')
latest_data = yf.download("XRP-USD", start=yesterday, end=today)

if not latest_data.empty:
    latest_close = latest_data['Close'][-1]
    latest_input = xrp[features].iloc[-1:].copy()
    latest_prediction = model.predict(latest_input)[0]

    signal = "BUY" if latest_prediction > latest_close else "HOLD/SELL"

    st.metric(label="Current XRP Price", value=f"${latest_close:.4f}")
    st.metric(label="Predicted Next Close", value=f"${latest_prediction:.4f}")
    st.markdown(f"### ✅ Suggested Action: **{signal}**")
else:
    st.warning("Could not fetch latest data. Try again later.")
>>>>>>> 0eed575e7fab532fa696a2d5a7e76e256c23cd12
