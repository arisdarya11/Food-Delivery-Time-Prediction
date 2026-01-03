import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    layout="centered"
)

# =========================
# LOAD IMAGE
# =========================
image = Image.open("dataset-cover.jpg")
st.image(image, use_container_width=True)

# =========================
# TITLE & DESCRIPTION
# =========================
st.title("🍔 Food Delivery Time Prediction")
st.write(
    "Aplikasi ini digunakan untuk memprediksi waktu pengantaran makanan "
    "berdasarkan jarak, kondisi cuaca, lalu lintas, waktu pengantaran, "
    "jenis kendaraan, dan pengalaman kurir."
)

st.divider()

# =========================
# LOAD MODEL & PREPROCESSOR
# =========================
model = joblib.load("model_delivery.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("onehot_encoder.pkl")  # 🔧 FIX DI SINI

# =========================
# USER INPUT
# =========================
st.subheader("📥 Input Order Details")

distance = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0)
prep_time = st.number_input("Preparation Time (minutes)", min_value=1, max_value=120, value=20)
experience = st.number_input("Courier Experience (years)", min_value=0.0, max_value=20.0, value=2.0)

weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Snowy", "Windy"])
traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
time_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
vehicle = st.selectbox("Vehicle Type", ["Scooter", "Car"])

# =========================
# DATAFRAME INPUT
# =========================
input_df = pd.DataFrame([{
    "Distance_km": distance,
    "Preparation_Time_min": prep_time,
    "Courier_Experience_yrs": experience,
    "Weather": weather,
    "Traffic_Level": traffic,
    "Time_of_Day": time_day,
    "Vehicle_Type": vehicle
}])

# =========================
# PREPROCESSING
# =========================
num_cols = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
num_scaled = scaler.transform(input_df[num_cols])
num_df = pd.DataFrame(num_scaled, columns=num_cols)

cat_cols = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
cat_encoded = encoder.transform(input_df[cat_cols])
cat_df = pd.DataFrame(
    cat_encoded.toarray(),
    columns=encoder.get_feature_names_out()
)

final_input = pd.concat([num_df, cat_df], axis=1)

# =========================
# PREDICTION
# =========================
if st.button("🔮 Predict Delivery Time"):
    prediction = model.predict(final_input)[0]

    st.success(f"⏱ Estimated Delivery Time: **{prediction:.1f} minutes**")
    st.info(f"📦 ETA Range: **{prediction-5:.0f} – {prediction+5:.0f} minutes**")

# =========================
# FOOTER
# =========================
st.caption("Model: Linear Regression | Use case: Food Delivery ETA Prediction")
