import os
import cv2
import pickle
import pyttsx3 
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
t2s = pyttsx3.init()

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av


@st.cache(suppress_st_warning=True)

def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    for key, value in feature_dict.items():
        if val == key:
            return value

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def Time_Formatx(x):
  # Formatting Time
    if pd.isna(x):
        return None
    if x == 2400:
        x = 0
    x = "{0:04d}".format(int(x))
    return datetime.time(int(x[0:2]), int(x[2:4]))
def preprocess():
    # Load data
    Airlines = pd.read_csv('airlines.csv')
    Airports = pd.read_csv('airports.csv')
    Flights = pd.read_csv('flights.csv')

 # Assuming dataOverview is a function defined elsewhere

    # Dropping rows with NaN values and selecting data for January
    Flights = Flights.iloc[:,:23]
    Flights.dropna(inplace=True)
    Flights = Flights[Flights["MONTH"] == 1]
    Flights.reset_index(drop=True, inplace=True)

    # Collecting Names of Airlines and Airports
    Airline_Names = {Airlines["IATA_CODE"][i]: Airlines["AIRLINE"][i] for i in range(len(Airlines))}
    Airport_Names = {Airports["IATA_CODE"][i]: Airports["AIRPORT"][i] for i in range(len(Airports))}
    City_Names = {Airports["IATA_CODE"][i]: Airports["CITY"][i] for i in range(len(Airports))}

    # Merging Datasets & Selecting relevant columns
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
    df = df[df.ARRIVAL_DELAY < 500]  # Filter to manage extreme values

    # Handling Categorical Variables with Label Encoding
    le = LabelEncoder()
    df['AIRLINE'] = le.fit_transform(df['AIRLINE'])
    df['ORIGIN'] = le.fit_transform(df['ORIGIN'])
    df['DESTINATION'] = le.fit_transform(df['DESTINATION'])
    df['TAIL_NUMBER_ENCODED'] = le.fit_transform(df['TAIL_NUMBER'])

    # Selecting Features for Model
    X = df[['AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER_ENCODED', 'ORIGIN', 'DESTINATION', 'DISTANCE', 'TAXI_OUT', 'DEPARTURE_DELAY']]
    Y = (df['ARRIVAL_DELAY'] > 15).astype(int)  # Binary classification, delay > 15 min

    # Splitting into Train, Validation, and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    # Standard Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)



    return X_train, y_train, X_val, y_val, X_test, y_test

def load_and_ppredict(flight_number):
    # Load the model
    loaded_model = load_model('lstm_model.h5')
    
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess()
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    # Make predictions
    predictions = loaded_model.predict(X_test)
    
    # Define a threshold
    threshold = 0.5 

    # Classify flights based on the threshold
    predictions_binary = ['delayed' if prob >= threshold else 'not delayed' for prob in predictions]

    # Load your dataset into a pandas DataFrame
    df = pd.read_csv("sampled_flights.csv")  
    
    # Check if the flight number exists in the DataFrame
    if flight_number in df['FLIGHT_NUMBER'].values:
        # Retrieve the corresponding flight status
        flight_status = df.loc[df['FLIGHT_NUMBER'] == flight_number, 'DELAYED'].iloc[0]
        
        # Convert flight status to readable format
        if flight_status == 0:
            actual_status = 'not delayed'
        else :
            actual_status = 'delayed'
        
        # Return the actual and predicted flight status
        predicted_status = predictions_binary[flight_number - 1]
        return f"Predicted status of Flight {flight_number}: {predicted_status}"
    else:
        return "Flight number not found.", ""


app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction']) 

if app_mode=='Home':    
    st.title('Flight Prediction ')    
    #st.write('App realised by : Jana , Jouna and Ahmad')  
    st.image('flight.jpeg')
    #st.markdown('Dataset')    
    #data=pd.read_csv('flights.csv')    
    #st.write(data.head())   

elif app_mode == 'Prediction':    
    st.title("Flight Delay Prediction")
    user_input = st.text_input('Please enter your flight ID number')

    if st.button('Predict Delay'):
        try:
            flight_id = int(user_input)  # Convert user input to integer
            predicted_status = load_and_ppredict(flight_id)  # Get prediction result
            #st.success(actual_status)  # Display the actual status
            st.success(predicted_status)  # Display the predicted status
        except ValueError:
            st.error("Please enter a valid numeric flight ID")
        except Exception as e:
            st.error(f"Error: {str(e)}")
