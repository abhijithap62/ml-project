import numpy as np
import streamlit as st

import joblib
model=joblib.load("global air pollution dataset.csv.pkl")
encoder1=joblib.load("encoder1.pkl")
encoder2=joblib.load("encoder2.pkl")
encoder3=joblib.load("encoder3.pkl")
encoder4=joblib.load("encoder4.pkl")
encoder5=joblib.load("encoder5.pkl")
encoder6=joblib.load("encoder6.pkl")
encoder7=joblib.load("encoder7.pkl")
scaler=joblib.load("scaler.pkl")

st.title("Air Quality Checking Analysis")
st.write("Air quality checking using climate.")

Country=st.selectbox("Enter your country",['Russian Federation','Brazil','Italy','Poland','France','India','United States of America','Malaysia'])
City=st.selectbox("Enter your city",['Praskoveya','Presidente Dutra','Priolo Gargallo','Przasnysz','Punaauia','Gursahaiganj','Westerville','Marang'])
AQI_Value=st.number_input("Enter your AQI Value")
AQI_Category=st.selectbox("Enter your AQI Category",['Moderate','Good','Unhealthy for Sensitive Groups','Unhealthy','Very Unhealthy','Hazardous'])
CO_AQI_Value=st.number_input("Enter CO AQI Value")
CO_AQI_Category=st.selectbox("Enter CO AQI Category",['Moderate','Good','Unhealthy for Sensitive Groups','Unhealthy','Very Unhealthy','Hazardous',])
Ozone_AQI_Value=st.number_input("Enter Ozone AQI Value")
Ozone_AQI_Category=st.selectbox("Enter Ozone your AQI Category",['Moderate','Good','Unhealthy for Sensitive Groups','Unhealthy','Very Unhealthy','Hazardous'])
NO2_AQI_Value=st.number_input("Enter NO2 AQI Value	")
NO2_AQI_Category=st.selectbox("Enter NO2 AQI Category",['Moderate','Good','Unhealthy for Sensitive Groups','Unhealthy','Very Unhealthy','Hazardous'])
PM25_AQI_Value=st.number_input("Enter PM2.5 AQI Value")

Country=encoder1.transform([Country])[0]
City=encoder2.transform([City])[0]
AQI_Category=encoder3.transform([AQI_Category])[0]
CO_AQI_Category=encoder4.transform([CO_AQI_Category])[0]
Ozone_AQI_Category=encoder5.transform([Ozone_AQI_Category])[0]
NO2_AQI_Category=encoder6.transform([NO2_AQI_Category])[0]


if st.button("predict"):
    result=model.predict(scaler.transform([[Country,City,AQI_Value,AQI_Category,CO_AQI_Value,CO_AQI_Category,Ozone_AQI_Value,Ozone_AQI_Category,NO2_AQI_Value,NO2_AQI_Category,
                                            PM25_AQI_Value]]))[0]

    st.success("the output is {}".format(result))   
