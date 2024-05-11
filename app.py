import streamlit as st
import pandas as pd
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
    st.title("🔎 Time Series Forecasting (Brent Oil Prices)")

    # Load data
    data = load_data()
    # convert to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    # set index
    data.set_index('Date', inplace=True)

    st.write("## Brent Oil Price since Jan/2023")
    st.line_chart(data)

    # Display data as a table and a line chart
    st.write("## Most recent Brent Oil Prices: (-5 days)")
    st.dataframe(data.tail().sort_values('Date', ascending=False))

    st.write("## Oil Price Trend:")
    st.write("Reason for the up high trend in brent Prices in 2023")
    st.write("""
    > On June 4, 2023, OPEC+ members announced they would extend crude oil production cuts through the end of 2024. The cuts had been set to expire at the end of 2023. Following the June 4 meeting, Saudi Arabia announced an additional voluntary oil production cut of 1.0 million barrels per day (b/d) for July (with the option to extend) in addition to the OPEC+ cuts.
    > In early September, Saudi Arabia announced it would extend the country’s voluntary production cuts through the end of 2023. U.S. commercial crude oil inventories fell and on September 29, 2023 were at the lowest point since December 2, 2022. The limited supply provided upward pressure on crude oil prices, and on September 28, the price of Brent reached its high for the year, at \\$98/b.
    > After declining from the September highs, crude oil prices increased again in early October after the Israel-Hamas conflict began; the price of Brent reached \\$91/b on October 9.
    
    Source: https://www.eia.gov/todayinenergy/detail.php?id=61142
    """)

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



    # Prepare input data for prediction
    last_3_days = prepare_input(data)
    st.write("Last 3 Points Used for Prediction:")
    st.write(last_3_days.flatten())

    predicted_value = model.predict(last_3_days).flatten()
    st.write("Predicted Next Value:")
    st.write(predicted_value)



if __name__ == "__main__":
    main()
