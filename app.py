import joblib
import requests
import streamlit as st

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

brands = [col for col in df.columns if col.startswith('brand_')]
selected_brand = st.selectbox("Select Brand:", brands)
brand_data = {col: 1 if col == selected_brand else 0 for col in brands}

colors = [col for col in df.columns if col.startswith('color_')]
selected_color = st.selectbox("Select Color:", colors)
color_data = {col: 1 if col == selected_color else 0 for col in colors}

transmissions = [col for col in df.columns if col.startswith('transmission_type_')]
selected_transmission = st.selectbox("Select Transmission Type:", transmissions)
transmission_data = {col: 1 if col == selected_transmission else 0 for col in transmissions}

fuel_types = [col for col in df.columns if col.startswith('fuel_type_')]
selected_fuel_type = st.selectbox("Select Fuel Type:", fuel_types)
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
