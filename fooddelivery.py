import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

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
for ext in ["jpg", "png"]:
    if os.path.exists(f"dataset-cover.{ext}"):
        image = Image.open(f"dataset-cover.{ext}")
        st.image(image, use_container_width=True)
        break

# =========================
# TITLE
# =========================
st.title("🍔 Food Delivery Time Prediction")
st.write(
    "Aplikasi ini memprediksi estimasi waktu pengantaran makanan "
    "berdasarkan jarak, waktu persiapan, pengalaman kurir, "
    "kondisi cuaca, lalu lintas, waktu pengantaran, dan jenis kendaraan."
)

st.divider()

# =========================
# LOAD MODEL & PREPROCESSOR
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_delivery.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("onehot_encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_artifacts()

# =========================
# USER INPUT
# =========================
st.subheader("📥 Input Order Details")

distance = st.number_input("Distance (km)", 0.1, 50.0, 5.0)
prep_time = st.number_input("Preparation Time (minutes)", 1, 120, 20)
experience = st.number_input("Courier Experience (years)", 0.0, 20.0, 2.0)

weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Snowy", "Windy"])
traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
time_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
vehicle = st.selectbox("Vehicle Type", ["Scooter", "Car"])

# =========================
# BUILD INPUT DATA
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
# PREPROCESSING (SAMA DENGAN TRAINING)
# =========================
num_cols = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs"
]

cat_cols = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type"
]

# Scale numerik
num_scaled = scaler.transform(input_df[num_cols])

# Encode kategorikal
cat_encoded = encoder.transform(input_df[cat_cols])
if hasattr(cat_encoded, "toarray"):
    cat_encoded = cat_encoded.toarray()

# Gabungkan → numpy array
X_input = np.hstack([num_scaled, cat_encoded])

# =========================
# 🔥 SAMAKAN JUMLAH FITUR (PALING PENTING)
# =========================
expected_features = model.n_features_in_
current_features = X_input.shape[1]

if current_features < expected_features:
    X_input = np.hstack([
        X_input,
        np.zeros((1, expected_features - current_features))
    ])
elif current_features > expected_features:
    X_input = X_input[:, :expected_features]

# =========================
# PREDICTION
# =========================
if st.button("🔮 Predict Delivery Time"):
    prediction = model.predict(X_input)[0]

    st.success(f"⏱ Estimated Delivery Time: **{prediction:.1f} minutes**")
    st.info(f"📦 ETA Range: **{prediction - 5:.0f} – {prediction + 5:.0f} minutes**")

# =========================
# FOOTER
# =========================
st.caption("Model: Linear Regression | Food Delivery ETA Prediction")
