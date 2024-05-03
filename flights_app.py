# flight_delay_app.py
! pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load your preprocessed data (you can adapt this part)
# Example: Load flights.csv
df = pd.read_csv("flights.csv")

# Load the trained LSTM model
model = load_model('lstm_model.h5')

# Sidebar inputs
st.sidebar.header("Flight Details")
flight_number = st.sidebar.text_input("Enter Flight Number")
origin = st.sidebar.selectbox("Select Origin", df["Origin"].unique())
destination = st.sidebar.selectbox("Select Destination", df["Dest"].unique())

# Filter data based on user inputs
filtered_df = df[(df["FlightNum"] == flight_number) & (df["Origin"] == origin) & (df["Dest"] == destination)]

if filtered_df.empty:
    st.warning("No matching flights found.")
else:
    # Prepare input features for prediction
    # (you'll need to adapt this part based on your preprocessing steps)
    input_features = np.array(filtered_df.drop(columns=["Delay"]).iloc[0]).reshape(1, 1, -1)

    # Make predictions
    predicted_delay_probability = model.predict(input_features)[0][0]

    # Determine delay status
    if predicted_delay_probability >= 0.5:
        st.write("Flight is likely to be delayed.")
    else:
        st.write("Flight is likely to be on time.")

# Run the app
if __name__ == "__main__":
    st.title("Flight Delay Prediction App")
    st.write("Enter flight details in the sidebar.")
