import streamlit as st
import pandas as pd
import joblib
import requests

def download_model(url, output_path):
    response = requests.get(url)
    with open(output_path, "wb") as f:
        f.write(response.content)

model_url = "https://drive.google.com/uc?export=download&id=1UJ-T-vAtrMCqnUJrNA44-hossadSh7Wk"
model_path = "final_rf_model.pkl"

try:
    with open(model_path, "rb") as f:
        st.write("Model file already exists locally.")
except FileNotFoundError:
    st.write("Downloading the model file...")
    download_model(model_url, model_path)
    st.write("Model file downloaded successfully.")

model = joblib.load(model_path)

st.title("Car Price Prediction App")
st.write("Enter the car details below to predict the price in Euros.")

power_kw = st.number_input("Power (kW):", min_value=0.0, step=0.1)
power_ps = st.number_input("Power (PS):", min_value=0.0, step=0.1)
fuel_consumption = st.number_input("Fuel Consumption (g/km):", min_value=0.0, step=0.1)
mileage_in_km = st.number_input("Mileage (in km):", min_value=0.0, step=100.0)
car_age = st.number_input("Car Age (Years):", min_value=0, step=1)

brands = [
    'brand_alfa-romeo', 'brand_audi', 'brand_bmw', 'brand_cadillac',
    'brand_chevrolet', 'brand_chrysler', 'brand_citroen', 'brand_dodge',
    'brand_ferrari', 'brand_fiat', 'brand_ford', 'brand_honda',
    'brand_hyundai', 'brand_infiniti', 'brand_isuzu', 'brand_jaguar',
    'brand_jeep', 'brand_kia', 'brand_lada', 'brand_lamborghini',
    'brand_lancia', 'brand_land-rover', 'brand_maserati', 'brand_mazda'
]
colors = [
    'color_black', 'color_blue', 'color_bronze', 'color_brown',
    'color_gold', 'color_green', 'color_grey', 'color_orange',
    'color_red', 'color_silver', 'color_violet', 'color_white',
    'color_yellow'
]
transmissions = [
    'transmission_type_Manual', 'transmission_type_Automatic',
    'transmission_type_Semi-automatic', 'transmission_type_Unknown'
]
fuel_types = [
    'fuel_type_Diesel', 'fuel_type_Petrol', 'fuel_type_Hybrid',
    'fuel_type_Electric', 'fuel_type_Ethanol', 'fuel_type_Hydrogen',
    'fuel_type_LPG', 'fuel_type_CNG', 'fuel_type_Other'
]

selected_brand = st.selectbox("Select Brand:", brands)
selected_color = st.selectbox("Select Color:", colors)
selected_transmission = st.selectbox("Select Transmission Type:", transmissions)
selected_fuel_type = st.selectbox("Select Fuel Type:", fuel_types)

brand_data = {col: 1 if col == selected_brand else 0 for col in brands}
color_data = {col: 1 if col == selected_color else 0 for col in colors}
transmission_data = {col: 1 if col == selected_transmission else 0 for col in transmissions}
fuel_type_data = {col: 1 if col == selected_fuel_type else 0 for col in fuel_types}

if st.button("Predict"):
    input_data = pd.DataFrame({
        "power_kw": [power_kw],
        "power_ps": [power_ps],
        "fuel_consumption_g_km": [fuel_consumption],
        "mileage_in_km": [mileage_in_km],
        "car_age_years": [car_age],
        **brand_data,
        **color_data,
        **transmission_data,
        **fuel_type_data
    })

    prediction = model.predict(input_data)[0]

    st.success(f"The predicted price is €{prediction:,.2f}")

