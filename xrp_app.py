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
y = combined_data['Price'].iloc[1:]

# Align shapes
X = X.loc[y.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Show performance
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse:.4f}")

# Plot predictions vs actual
fig, ax = plt.subplots()
ax.plot(y_test.index, y_test, label='Actual')
ax.plot(y_test.index, y_pred, label='Predicted')
ax.set_title('FET/PKR Price Prediction')
ax.legend()
st.pyplot(fig)

# Optional: Show latest prediction
latest_price = combined_data['Close_PKR'].iloc[-1]
predicted_next = model.predict([[latest_price]])[0]
st.write(f"Latest FET/PKR price: {latest_price:.2f}")
st.write(f"Predicted next value: {predicted_next:.2f}")
