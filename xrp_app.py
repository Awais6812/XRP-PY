import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.title("FET to PKR Price Prediction")

# Download historical price data for FET-USD
fet_data = yf.download('FET-USD', start='2021-01-01', end='2025-01-01')[['Close']]

# Download historical exchange rate data for USD to PKR
usd_pkr_data = yf.download('USDPKR=X', start='2021-01-01', end='2025-01-01')[['Close']]

# Rename columns
fet_data.rename(columns={"Close": "FET_Close_USD"}, inplace=True)
usd_pkr_data.rename(columns={"Close": "USD_to_PKR"}, inplace=True)

# Join datasets on common date index
combined_data = fet_data.join(usd_pkr_data, how='inner')

# Calculate FET price in PKR
combined_data['Close_PKR'] = combined_data['FET_Close_USD'] * combined_data['USD_to_PKR']
combined_data['Price'] = combined_data['Close_PKR']
combined_data['Price_Change'] = combined_data['Price'].pct_change()

# Drop NaN values
combined_data.dropna(inplace=True)

# Prepare features and target
X = combined_data[['Close_PKR']].shift(1).dropna()
y = combined_data['Price'].iloc[1_
