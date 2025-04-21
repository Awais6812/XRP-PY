import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Fetch FET/USDT (or FET/USD) data from Yahoo Finance
fet_data = yf.download('FET-USD', start='2021-01-01', end='2025-01-01')

# Fetch USD/PKR exchange rate (from USD to PKR)
usd_pkr_data = yf.download('USDKRW=X', start='2021-01-01', end='2025-01-01')  # Use USDKRW=X for USD/PKR

# Rescale FET to PKR (multiply FET-USD by USD-PKR)
fet_data['Close_PKR'] = fet_data['Close'] * usd_pkr_data['Close']

# Data Preprocessing
fet_data['Price'] = fet_data['Close_PKR']
fet_data['Price_Change'] = fet_data['Price'].pct_change()

# Drop NaN values from the data
fet_data = fet_data.dropna()

# Train a basic linear regression model
X = fet_data[['Close_PKR']].shift(1).dropna()  # Lag the data to predict future
y = fet_data['Price'].iloc[1:]  # Predict the next day's price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
fet_data['Predicted_Close'] = model.predict(X)

# Trading Signal
fet_data['Signal'] = fet_data['Predicted_Close'] > fet_data['Price']

# Display Results
st.subheader("💡 FET/PKR Price Prediction & Trading Signal")
st.dataframe(fet_data[['Price', 'Predicted_Close', 'Signal']].tail(10))

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fet_data.index, fet_data['Price'], label="Actual FET/PKR Price", color='blue')
ax.plot(fet_data.index, fet_data['Predicted_Close'], label="Predicted FET/PKR Price", color='orange')
ax.set_xlabel('Date')
ax.set_ylabel('Price (PKR)')
ax.set_title('FET/PKR Price Prediction')
ax.legend()
st.pyplot(fig)
