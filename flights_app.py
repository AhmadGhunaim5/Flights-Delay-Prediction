!pip install tensorflow
!pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

def preprocess(analysis=False):
    # Load data
    Airlines = pd.read_csv('airlines.csv')
    Airports = pd.read_csv('airports.csv')
    Flights = pd.read_csv('flights.csv')

    if analysis:
        dataOverview(Airlines, Airports, Flights)  

    Flights = Flights.iloc[:,:23]
    Flights.dropna(inplace=True)
    Flights = Flights[Flights["MONTH"] == 1]
    Flights.reset_index(drop=True, inplace=True)

    Airline_Names = {Airlines["IATA_CODE"][i]: Airlines["AIRLINE"][i] for i in range(len(Airlines))}
    Airport_Names = {Airports["IATA_CODE"][i]: Airports["AIRPORT"][i] for i in range(len(Airports))}
    City_Names = {Airports["IATA_CODE"][i]: Airports["CITY"][i] for i in range(len(Airports))}

    df = pd.DataFrame()
    df['DATE'] = pd.to_datetime(Flights[['YEAR', 'MONTH', 'DAY']])
    df['DAY'] = Flights["DAY_OF_WEEK"]
    df['AIRLINE'] = Flights["AIRLINE"]
    df['FLIGHT_NUMBER'] = Flights['FLIGHT_NUMBER']
    df['TAIL_NUMBER'] = Flights['TAIL_NUMBER']
    df['ORIGIN'] = Flights['ORIGIN_AIRPORT']
    df['DESTINATION'] = Flights['DESTINATION_AIRPORT']
    df['DISTANCE'] = Flights['DISTANCE']
    df['SCHEDULED_DEPARTURE'] = Flights['SCHEDULED_DEPARTURE'].apply(Time_Formatx)
    df['SCHEDULED_ARRIVAL'] = Flights['SCHEDULED_ARRIVAL'].apply(Time_Formatx)
    df['TAXI_OUT'] = Flights['TAXI_OUT']
    df['DEPARTURE_DELAY'] = Flights['DEPARTURE_DELAY']
    df['ARRIVAL_DELAY'] = Flights['ARRIVAL_DELAY']
    df = df[df.ARRIVAL_DELAY < 500] 

    if analysis:
        exploratoryDataAnalysis(df)  
        
    le = LabelEncoder()
    df['AIRLINE'] = le.fit_transform(df['AIRLINE'])
    df['ORIGIN'] = le.fit_transform(df['ORIGIN'])
    df['DESTINATION'] = le.fit_transform(df['DESTINATION'])
    df['TAIL_NUMBER_ENCODED'] = le.fit_transform(df['TAIL_NUMBER'])

    X = df[['AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER_ENCODED', 'ORIGIN', 'DESTINATION', 'DISTANCE', 'TAXI_OUT', 'DEPARTURE_DELAY']]
    Y = (df['ARRIVAL_DELAY'] > 15).astype(int)  # Binary classification, delay > 15 min

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if analysis:
        print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def load_and_predict():
    # Load the LSTM model
    loaded_model = load_model('lstm_model.h5')
    
    # Preprocess the data
    X_test, y_test = preprocess()
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # Make predictions
    predictions = loaded_model.predict(X_test)
    return predictions

# Streamlit app title and description
st.title('The US Flight Delays Predictor App')
st.markdown('Enter flight details to predict delay.')

user_input = st.text_input("Enter your flight number")

if st.button("Predict"):
    prediction = predict_sarcasm(user_input)
    sarcasm_probability = prediction[0][1]
    st.write(f"Sarcasm Probability: {sarcasm_probability:.2f}")
    if sarcasm_probability > 0.5:
        st.write("This text is sarcastic.")
    else:
        st.write("This text is not sarcastic.")



# Prediction button
if st.button('Predict Delay'):
    if flight_number:
        delay_probability = predict_delay(flight_number)
        if delay_probability > 0.5:
            st.write("This flight is predicted to be delayed.")
        else:
            st.write("This flight is predicted to be on time.")
    else:
        st.write("Wrong flight number.")

st.markdown(
    """
    <style>
    body {
        background-image: url("flights.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
