import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf

# Load your trained LSTM model
model = tf.keras.models.load_model('lstm_model.keras')

# Function to load and return the dataset
def load_data():
    data = pd.read_csv('oil_prices_sorted_since_2023.csv')
    return data


# Prepare input data for prediction
def prepare_input(data):
    last_points = data[-3:].values.reshape(-1, 1)
    return last_points.reshape(1, 3, 1)


# Main app function
def main():
    st.title("Time Series Forecasting")

    # Load data
    data = load_data()
    # convert to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    # set index
    data.set_index('Date', inplace=True)

    # Display data as a table and a line chart
    st.write("Recent Oil Price Data:")
    st.dataframe(data.tail())

    st.write("Oil Price Trend:")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Price'], label='Price')
    ax.set_title("Oil Prices Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid()
    plt.xticks(rotation=45)
    # figsize
    fig.set_size_inches(15, 5)
    st.pyplot(fig)

    st.write("Data Visualization:")
    st.line_chart(data)

    # Prepare input data for prediction
    last_3_days = prepare_input(data)
    st.write("Last 3 Points Used for Prediction:")
    st.write(last_3_days.flatten())

    predicted_value = model.predict(last_3_days).flatten()
    st.write("Predicted Next Value:")
    st.write(predicted_value)



if __name__ == "__main__":
    main()
