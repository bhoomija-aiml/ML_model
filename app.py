import streamlit as st
import pickle
import pandas as pd

# load model + feature columns
model, feature_columns = pickle.load(open("model.pkl", "rb"))

st.title("🌾 Crop Yield Prediction App")

st.write("Enter details:")

# 🔹 Numeric Inputs
soil_moisture = st.number_input("Soil Moisture (%)")
soil_ph = st.number_input("Soil pH")
temperature = st.number_input("Temperature (°C)")
rainfall = st.number_input("Rainfall (mm)")
humidity = st.number_input("Humidity (%)")
sunlight = st.number_input("Sunlight Hours")
pesticide = st.number_input("Pesticide Usage (ml)")
days = st.number_input("Total Days")
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")
ndvi = st.number_input("NDVI Index")

# 🔹 Categorical Inputs
region = st.selectbox("Region", [
    "Central USA", "East Africa", "North India", "South India", "South USA"
])

crop = st.selectbox("Crop Type", [
    "Cotton", "Maize", "Rice", "Soybean", "Wheat"
])

irrigation = st.selectbox("Irrigation Type", [
    "Drip", "Manual", "Sprinkler"
])

fertilizer = st.selectbox("Fertilizer Type", [
    "Inorganic", "Mixed", "Organic"
])

disease = st.selectbox("Crop Disease Status", [
    "Mild", "Moderate", "Severe"
])

# 🔹 Prediction
if st.button("Predict Yield"):

    input_dict = {
        'soil_moisture_%': soil_moisture,
        'soil_pH': soil_ph,
        'temperature_C': temperature,
        'rainfall_mm': rainfall,
        'humidity_%': humidity,
        'sunlight_hours': sunlight,
        'pesticide_usage_ml': pesticide,
        'total_days': days,
        'latitude': latitude,
        'longitude': longitude,
        'NDVI_index': ndvi
    }

    # one-hot encoding
    input_dict[f"region_{region}"] = 1
    input_dict[f"crop_type_{crop}"] = 1
    input_dict[f"irrigation_type_{irrigation}"] = 1
    input_dict[f"fertilizer_type_{fertilizer}"] = 1
    input_dict[f"crop_disease_status_{disease}"] = 1

    input_df = pd.DataFrame([input_dict])

    # match columns
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_df)

    st.success(f"🌱 Predicted Yield: {prediction[0]:.2f} kg/hectare")