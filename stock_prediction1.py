import streamlit as st
import yfinance as finance
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
company1 = get_ticker("GOOGL")
company2 = get_ticker("MSFT")
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
# import streamlit as st
# import yfinance as finance
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# def get_ticker(name):
#     company = finance.Ticker(name)
#     return company

# # Project Details
# st.title("Build and Deploy Stock Market App Using Streamlit")
# st.header("A Basic Data Science Web Application")
# st.sidebar.header("Geeksforgeeks \n TrueGeeks")

# company1 = get_ticker("GOOGL")
# company2 = get_ticker("MSFT")
# # Fetch real-time stock data
# google = company1.history(period="1d", interval="1m")
# microsoft = company2.history(period="1d", interval="1m")

# # Calculate Moving Average
# google['MA_50'] = google['Close'].rolling(window=50).mean()
# microsoft['MA_50'] = microsoft['Close'].rolling(window=50).mean()

# # Plot the graph
# st.write("### Google")
# st.write(company1.info['longBusinessSummary'])
# st.line_chart(google[['Close', 'MA_50']])

# st.write("### Microsoft")
# st.write(company2.info['longBusinessSummary'])
# st.line_chart(microsoft[['Close', 'MA_50']])

# # Scale the data
# scaler = MinMaxScaler()
# scaled_google = scaler.fit_transform(google[['Close']])
# scaled_microsoft = scaler.fit_transform(microsoft[['Close']])

# # Prepare data for LSTM
# def create_dataset(data, time_step=1):
#     X, y = [], []
#     for i in range(len(data) - time_step - 1):
#         X.append(data[i:(i + time_step), 0])
#         y.append(data[i + time_step, 0])
#     return np.array(X), np.array(y)

# time_step = 60
# X_google, y_google = create_dataset(scaled_google, time_step)
# X_microsoft, y_microsoft = create_dataset(scaled_microsoft, time_step)

# # Reshape input to be [samples, time steps, features]
# X_google = X_google.reshape(X_google.shape[0], X_google.shape[1], 1)
# X_microsoft = X_microsoft.reshape(X_microsoft.shape[0], X_microsoft.shape[1], 1)

# # Build LSTM model
# def build_model():
#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
#     model.add(LSTM(50, return_sequences=False))
#     model.add(Dense(25))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# model_google = build_model()
# model_microsoft = build_model()

# # Train the model
# model_google.fit(X_google, y_google, epochs=10, batch_size=32)
# model_microsoft.fit(X_microsoft, y_microsoft, epochs=10, batch_size=32)

# # Display the models' summaries
# st.write("### Google LSTM Model Summary")
# st.text(model_google.summary())

# st.write("### Microsoft LSTM Model Summary")
# st.text(model_microsoft.summary())