import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pickle 


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

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction']) 

if app_mode=='Home':    
    st.title('Flight Prediction ')    
    #st.write('App realised by : Jana , Jouna and Ahmad')  
    st.image('flight.jpg')
    #st.markdown('Dataset')    
    #data=pd.read_csv('flights.csv')    
    #st.write(data.head())   

elif app_mode == 'Prediction':    
    st.title ("Flight Delay Prediction")
    user_input = st.text_input('Please enter your flight ID number')
    st.button('Click me!')
    df = pd.DataFrame(np.random.randn(500, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])
    st.map(df)
