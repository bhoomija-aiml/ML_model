import streamlit as st
import pickle
import numpy as np

# load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🌾 Crop Yield Prediction App")

st.write("Enter the details below:")

# INPUTS (change based on dataset columns)
rainfall = st.number_input("Rainfall")
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")

# prediction button
if st.button("Predict Yield"):
    input_data = np.array([[rainfall, temperature, humidity]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Yield: {prediction[0]}")