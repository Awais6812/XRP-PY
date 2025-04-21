import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Title
st.title("Fetch.ai (FET) Price Predictor in PKR")

# Download FET-USD and USD/PKR data
fet_data = yf.download('FET-USD', start='2021-01-01', end='2025-01-01')[['Close']]
usd_pkr_data = yf.download('USDPKR=X', start='2021-01-01', end='2025-01-01')[['Close']]

# Rename columns to avoid conflicts during merge
fet_data.rename(columns={"Close": "FET_Close_USD"}, inplace=True)
usd_pkr_data.rename(columns={"Close": "USD_to_PKR"}, inplace=True)

# Join on index (Date)
combined_data = fet_data.join(usd_pkr_data, how='inner')

# Calculate FET price in PKR
combined_data['Close_PKR'] = combined_data['FET_Close_USD'] * combined_data['USD_to_PKR']
combined_data['Price'] = combined_data['Close_PKR']
combined_data['Price_Change'] = combined_data['Price'].pct_change()

# Display recent data
st.subheader("Recent Data")
st.dataframe(combined_data.tail())

# Prepare data for model
X = combined_data[['Close_PKR']].shift(1).dropna()  # Previous day's price
y = combined_data['Price'].iloc[1:]  # Today's price

# Align X and y
X = X.loc[y.index]

# Train-test split
if len(X) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Display MSE
    mse = mean_squared_error(y_test, y_pred)
    st.subheader(f"Model Mean Squared Error: {mse:.4f}")

    # Predict next day price
    latest_price = combined_data['Close_PKR'].iloc[-1]
    predicted_price = model.predict([[latest_price]])[0]

    st.subheader(f"Predicted Next Day FET Price in PKR: {predicted_price:.2f}")

    # Plot results
    st.subheader("Prediction vs Actual")
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    st.pyplot(plt)
else:
    st.warning("Not enough data to train the model. Please check the data source or range.")
