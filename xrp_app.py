import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
fet_data = yf.download('FET-USD', start='2021-01-01', end='2025-01-01')[['Close']]
usd_pkr_data = yf.download('USDPKR=X', start='2021-01-01', end='2025-01-01')[['Close']]

# Rename columns for clarity
fet_data.rename(columns={"Close": "FET_Close_USD"}, inplace=True)
usd_pkr_data.rename(columns={"Close": "USD_to_PKR"}, inplace=True)

# Align both dataframes on common dates
combined_data = fet_data.join(usd_pkr_data, how='inner')

# Calculate FET price in PKR
combined_data['Close_PKR'] = combined_data['FET_Close_USD'] * combined_data['USD_to_PKR']
combined_data['Price'] = combined_data['Close_PKR']
combined_data['Price_Change'] = combined_data['Price'].pct_change()

# Drop NA values
combined_data.dropna(inplace=True)

# Lagged data for prediction
combined_data['Lag_Close'] = combined_data['Close_PKR'].shift(1)
combined_data.dropna(inplace=True)

X = combined_data[['Lag_Close']]
y = combined_data['Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
combined_data['Predicted_Close'] = model.predict(X)

# Trading signal
combined_data['Signal'] = combined_data['Predicted_Close'] > combined_data['Price']

# Display results
st.subheader("💡 FET/PKR Price Prediction & Trading Signal")
st.dataframe(combined_data[['Price', 'Predicted_Close', 'Signal']].tail(10))

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(combined_data.index, combined_data['Price'], label="Actual FET/PKR Price", color='blue')
ax.plot(combined_data.index, combined_data['Predicted_Close'], label="Predicted FET/PKR Price", color='orange')
ax.set_xlabel('Date')
ax.set_ylabel('Price (PKR)')
ax.set_title('FET/PKR Price Prediction')
ax.legend()
st.pyplot(fig)