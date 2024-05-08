import os
import streamlit as st
import pandas as pd

def main():
    # Set the path to your desktop and the archive folder
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    archive_folder_path = os.path.join(desktop_path, "archive")
    
    # Set the path to the background image
    background_image_path = os.path.join(archive_folder_path, "flight.jpg")

    # Set the background image using CSS
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url("{background_image_path}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Flight Delay Prediction")

    # Load the LSTM model
    lstm_model = load_model('lstm_model.h5')

    # Sidebar for user input
    st.sidebar.title("User Input")
    flight_number = st.sidebar.number_input("Enter Flight Number", min_value=1)
    if st.sidebar.button("Predict"):
        prediction = load_and_predict(flight_number)
        st.write(f"Predicted status of Flight {flight_number}: {prediction}")

    # Display some insights and analysis
    st.subheader("Data Overview")
    analysis_checkbox = st.checkbox("Perform Data Analysis")
    if analysis_checkbox:
        preprocess(analysis=True)

# Run the app
if __name__ == "__main__":
    main()
