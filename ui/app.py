import streamlit as st
from PIL import Image
from time import sleep
import os
import glob
from datetime import date
from os.path import exists
import pandas as pd
import numpy as np
import joblib

#method to convert sparse vector to array
def convert_sparse_to_array(x):
    return x.toarray()

#loading the dataset to get the columns and categories
df = pd.read_csv(r"../data/CO2 Emissions_Canada.csv")

#Assigining Cylinders variable as object
df['Cylinders'] = df['Cylinders'].astype('O')
# Dropping unwanted columns
df.drop(['CO2 Emissions(g/km)', 'Model', 'Fuel Consumption Comb (L/100 km)'], axis = 1, inplace = True)
data_columns = list(df.columns)

#Dictionary to represent the fuel type
fuel_dict = {
    'Regular Gasoline':'X',
    'Premium Gasoline': 'Z',
    'Diesel': 'D',
    'Ethanol': 'E',
    'Natural Gas': 'N'
}

# STREAMLIT code

st.set_page_config(page_title="Vehicle's CO2 Emission prediction", page_icon=":car:")

st.title("CO2 Emission Predictor [v1]")

st.write('This is a simple app designed to predict the CO2 emission (in grams) of a particular vehicle')

image = Image.open(r'./img/cover-photo.jpeg')
st.image(image, use_column_width=True)


#Having 2 tabs to separate about section and prediction section
tab_1, tab_2 = st.tabs(["About", "Prediction"])

#Code for About section
tab_1.markdown("<p>Transport accounts for around one-fifth of global carbon dioxide (CO2) emissions [24% if we only consider CO2 emissions from energy].</p>", unsafe_allow_html = True)
tab_1.markdown("<p>Road travel accounts for three-quarters of transport emissions. Most of this comes from passenger vehicles – cars and buses – which contribute 45.1%. The other 29.4% comes from trucks carrying freight.</p>", unsafe_allow_html = True)
tab_1.markdown("<p>Since the entire transport sector accounts for 21% of total emissions, and road transport accounts for three-quarters of transport emissions, road transport accounts for 15% of total CO2 emissions.</p>", unsafe_allow_html = True)
tab_1.markdown("<p> <b> <u>This application attempts to predict the CO2 emissions of a particular vehicle, based on certain featurs, to make sure the vehicles abide the maximum CO2 emission limit, to control carbon limit.</u></b></p>", unsafe_allow_html = True)

#Code for Prediction section

#Getting the CO2 limit from the user
co2_limit = tab_2.number_input(label = "The maximum limit of CO2 emission (in grams per kilometer):", min_value = 0.0)

tab_2.header("Please input the necessary details of the vehicles as stated below:")

#Getting the input values from the user
make = tab_2.selectbox(label = "Select the Make of the vehicle", options = list(df.Make.unique()) + ['Other'], help = "Choose the vehicle's manufacturer from the drop-down list. If not in list, select 'Other'")
vehicle_class = tab_2.selectbox(label = "Select the type of vehicle", options = list(df['Vehicle Class'].unique()) + ['Other'], help = "Choose the type of the vehicle, whether it is a SUV, Van, 2-seater, etc.")
engine_size = float(tab_2.number_input(label = "What is the size of the engine (in litres)?", min_value = 0.0, max_value = 60.0, help = "Generally, the engine size of vehicles are between 0.0 to 10.0 litres."))
cylinders = float(tab_2.selectbox(label = "How many cylinders are there in the vehicle's engine", options = list(df.Cylinders.unique()), help = "Choose the number of engines in the vehicle."))
transmission = tab_2.selectbox(label = "Select the transmission of the vehicle", options = list(df.Transmission.unique()) + ['Other'], help = "Select the Vehicle Transmission. A: Automatic | AM: Automated Manual | AS: Automatic with select shift | AV: Continuously Variable | M: Manual | 3-10 : no of gears")
fuel_type = tab_2.selectbox(label = "Select the fuel type of the vehicle", options = list(fuel_dict.keys()), help = "Select the type of fuel that the vehicle runs on")
fuel_consump_city = float(tab_2.number_input(label = "What is the expected fuel consumption (in litres) of the vehicle in cities for 100km?", min_value = 0.0, max_value = 60.0))
fuel_consump_hwy = float(tab_2.number_input(label = "What is the expected fuel consumption (in litres) of the vehicle in highways for 100km?", min_value = 0.0, max_value = 60.0))
fuel_consump_comb = float(tab_2.number_input(label = "What is the expected miles per gallon (mpg) for 100km?", min_value = 0.0, max_value = 75.0))

#Model inference code
if tab_2.button(label = "Submit"):
    #Loading the saved model pipeline
    model_pipeline = joblib.load(filename = r"../artefacts/lda_model_pipeline.pkl")

    #Getting the correct fuel type from dictionary
    fuel_type_real = fuel_dict[fuel_type]

    #Creating a dataframe based on the input values
    real_time_df = pd.DataFrame(
        np.array([[make, vehicle_class, float(engine_size), cylinders, transmission, fuel_type_real, float(fuel_consump_city), float(fuel_consump_hwy), float(fuel_consump_comb)]]),
        columns = data_columns
    )

    #Pre-processing certain fields of inference dataframe to input to the model
    real_time_df['Engine Size(L)'] = real_time_df['Engine Size(L)'].astype('float')
    real_time_df['Cylinders'] = real_time_df['Cylinders'].astype('O')
    real_time_df['Fuel Consumption City (L/100 km)'] = real_time_df['Fuel Consumption City (L/100 km)'].astype('float')
    real_time_df['Fuel Consumption Hwy (L/100 km)'] = real_time_df['Fuel Consumption Hwy (L/100 km)'].astype('float')
    real_time_df['Fuel Consumption Comb (mpg)'] = real_time_df['Fuel Consumption Comb (mpg)'].astype('float')

    #Model prediction
    co2_prediction = model_pipeline.predict(real_time_df)

    tab_2.markdown(f"<p style=\"font-size: 20px;\">The expected CO2 emission of this particular vehicle is {co2_prediction[0]} grams per kilometer</p>", unsafe_allow_html = True)

    #If block to check whether the predicted CO2 emission is greater/lesser than the assigned limit
    if(float(co2_limit) < co2_prediction):
        print("Prediction successful. CO2 predicted greater than the assigned CO2 limit")
        tab_2.markdown(f"<p style=\"font-size: 20px; color:red;\">ALERT<br/>The Predicted CO2 emissions exceeds the stated CO2 emission limit </p>", unsafe_allow_html = True)
    else:
        print("Prediction successful. CO2 predicted lesser than the assigned CO2 limit")
        tab_2.markdown(f"<p style=\"font-size: 20px; color:green;\">SAFE<br/>The Predicted CO2 emissions is within the stated CO2 emission limit </p>", unsafe_allow_html = True)
