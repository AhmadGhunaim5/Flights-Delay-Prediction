import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os

# Load the dataset
data = pd.read_csv('trialF.csv')

# Check if 'Date' column exists, otherwise handle appropriately
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
else:
    st.error("The column 'Date' does not exist in the dataset. Please check the dataset.")

# Add the background image from URL
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://media.istockphoto.com/id/1129440561/photo/aerial-view-of-clouds-seen-from-the-plane.jpg?s=612x612&w=0&k=20&c=nLa64DA0Ej24zRAivR6cLO_9umk5c9SpVpJBDWyW6NE=");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

# Initialize user database in session state if not already present
if 'user_db' not in st.session_state:
    st.session_state.user_db = {
        "AhmadGh": {"password": "12345", "full_name": "Ahmad Gh", "email": "ahmad@gmail.com", "reserved_flights": ["2204"]}
    }

# Function for checking login credentials
def check_login(username, password):
    user_db = st.session_state.user_db
    if username in user_db:
        if user_db[username]["password"] == password:
            return True
    return False

# Function for registering a new user
def register_user(username, password, full_name, email, reserved_flights):
    user_db = st.session_state.user_db
    st.session_state.user_db[username] = {"password": password, "full_name": full_name, "email": email, "reserved_flights": reserved_flights}
    return True

# Main app logic
def main():
    st.title("Your Flight Predictor")
    st.sidebar.title("Navigation")
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""
    
    if 'app_mode' not in st.session_state:
        st.session_state['app_mode'] = 'Login'
        st.session_state['reserved_flights_for_prediction'] = []

    app_mode = st.sidebar.selectbox('Select Page', ['Login', 'Register', 'Profile', 'My Flights', 'Prediction'], index=['Login', 'Register', 'Profile', 'My Flights', 'Prediction'].index(st.session_state['app_mode']))

    if app_mode == 'Login':
        st.title("Login Page")
        
        # Login form
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type='password', key="login_password")

        if st.button("Login"):
            if check_login(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['app_mode'] = 'Profile'
                st.success("Login successful")
                st.experimental_rerun()  # Rerun the script to load the selected page
            else:
                st.error("Invalid username or password")

    elif app_mode == 'Register':
        st.title("Register Page")
        
        # Registration form
        new_username = st.text_input("New Username", key="register_username")
        new_password = st.text_input("New Password", type='password', key="register_password")
        full_name = st.text_input("Full Name", key="register_full_name")
        email = st.text_input("Email", key="register_email")
        flights = st.text_area("Reserved Flight Numbers (comma-separated)", key="register_flights")

        if st.button("Register"):
            reserved_flights = [flight.strip() for flight in flights.split(",")] if flights else []
            register_user(new_username, new_password, full_name, email, reserved_flights)
            st.success("Registration successful. Please log in.")
            st.session_state['app_mode'] = 'Prediction'
            st.session_state['reserved_flights_for_prediction'] = reserved_flights
            st.experimental_rerun()  # Rerun the script to load the prediction page

    elif app_mode == 'Profile':
        if st.session_state['logged_in']:
            st.title("User Profile")
            username = st.session_state['username']
            user_profile = st.session_state.user_db[username]
            st.write(f"**Full Name:** {user_profile['full_name']}")
            st.write(f"**Username:** {username}")
            st.write(f"**Email:** {user_profile['email']}")
            
            if user_profile['reserved_flights']:
                st.write("You have reserved flights:")
                for flight in user_profile['reserved_flights']:
                    st.write(f"Flight Number: {flight}")
                if st.button('Go to Prediction'):
                    st.session_state['app_mode'] = 'Prediction'
                    st.session_state['reserved_flights_for_prediction'] = user_profile['reserved_flights']
                    st.experimental_rerun()  # Rerun the script to load the prediction page
        else:
            st.error("Please log in to access this page")

    elif app_mode == 'My Flights':
        if st.session_state['logged_in']:
            st.title("My Flights")
            username = st.session_state['username']
            user_profile = st.session_state.user_db[username]
            st.write("Here are your reserved flights:")
            for flight in user_profile['reserved_flights']:
                st.write(f"Flight Number: {flight}")
        else:
            st.error("Please log in to access this page")

    elif app_mode == 'Prediction':
        if st.session_state['logged_in']:
            st.title("Flight Delay Prediction")
            reserved_flights = st.session_state.get('reserved_flights_for_prediction', [])
            
            if reserved_flights:
                st.write("Predicting delays for your reserved flights:")
                for flight_number in reserved_flights:
                    flight_data = data[data['Flight Number'].astype(str) == str(flight_number)]
                    if len(flight_data) == 0:
                        st.error(f"No record found for Flight Number {flight_number}")
                    else:
                        actual_status = flight_data['Prediction'].values[0]
                        delay_percentage = flight_data['Prediction Percentage'].values[0]
                        st.success(f"Flight Number {flight_number} is {actual_status} with a Delay percentage of {round(delay_percentage * 100, 2)}%")
            else:
                user_input = st.text_input('Please enter your Flight Number')

                # Print the unique flight numbers for debugging
                #st.write("Available Flight Numbers:", data['Flight Number'].unique())

                if st.button('Predict Delay'):
                    try:
                        flight_data = data[data['Flight Number'].astype(str) == str(user_input)]
                        if len(flight_data) == 0:
                            st.error("No record found for the given Flight Number")
                        else:
                            actual_status = flight_data['Prediction'].values[0]
                            delay_percentage = flight_data['Prediction Percentage'].values[0]
                            st.success(f"The Flight is {actual_status} with a Delay percentage of {round(delay_percentage * 100, 2)}%")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.error("Please log in to access this page")

if __name__ == '__main__':
    main()
