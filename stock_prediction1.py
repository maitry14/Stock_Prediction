import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

# Function to fetch stock data
def get_ticker(name):
    company = finance.Ticker(name)
    return company

# Streamlit App
st.title("Build and Deploy Stock Market App Using Streamlit")
st.header("A Basic Data Science Web Application")
st.sidebar.header("Geeksforgeeks \n TrueGeeks")

# Fetch real-time stock data
company1 = yf.Ticker("GOOGL")
company2 = yf.Ticker("MSFT")
google = company1.history(period="1y")  # 1 year of data
microsoft = company2.history(period="1y")

# Calculate Moving Average
google['MA_50'] = google['Close'].rolling(window=50).mean()
microsoft['MA_50'] = microsoft['Close'].rolling(window=50).mean()

# Plot the graph
st.write("### Google")
st.write(company1.info['longBusinessSummary'])
st.line_chart(google[['Close', 'MA_50']])

st.write("### Microsoft")
st.write(company2.info['longBusinessSummary'])
st.line_chart(microsoft[['Close', 'MA_50']])

# Prepare data for Prophet
def prepare_data(data):
    df = data[['Close']].reset_index()
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    # Remove timezone information from the 'ds' column
    df['ds'] = df['ds'].dt.tz_localize(None)
    return df

google_df = prepare_data(google)
microsoft_df = prepare_data(microsoft)

# Build Prophet model
def build_prophet_model(data):
    model = Prophet()
    model.fit(data)
    return model

# Train Prophet models
model_google = build_prophet_model(google_df)
model_microsoft = build_prophet_model(microsoft_df)

# Make future predictions
future_google = model_google.make_future_dataframe(periods=30)  # Predict next 30 days
forecast_google = model_google.predict(future_google)

future_microsoft = model_microsoft.make_future_dataframe(periods=30)  # Predict next 30 days
forecast_microsoft = model_microsoft.predict(future_microsoft)

# Plot forecasts
st.write("### Google Stock Price Forecast")
fig_google = model_google.plot(forecast_google)
st.pyplot(fig_google)

st.write("### Microsoft Stock Price Forecast")
fig_microsoft = model_microsoft.plot(forecast_microsoft)
st.pyplot(fig_microsoft)

# Show forecast components
st.write("### Google Forecast Components")
fig_components_google = model_google.plot_components(forecast_google)
st.pyplot(fig_components_google)

st.write("### Microsoft Forecast Components")
fig_components_microsoft = model_microsoft.plot_components(forecast_microsoft)
st.pyplot(fig_components_microsoft)

