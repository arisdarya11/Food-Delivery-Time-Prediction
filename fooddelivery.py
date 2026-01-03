import streamlit as st
import pandas as pd
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
image_path = None
for ext in ["jpg", "png"]:
    if os.path.exists(f"dataset-cover.{ext}"):
        image_path = f"dataset-cover.{ext}"
        break

if image_path:
    image = Image.open(image_path)
    st.image(image, use_container_width=True)

# =========================
# TITLE & DESCRIPTION
# =========================
st.title("🍔 Food Delivery Time Prediction")
st.write(
    "Aplikasi ini digunakan untuk memprediksi waktu pengantaran makanan "
    "berdasarkan jarak, waktu persiapan, pengalaman kurir, kondisi cuaca, "
    "lalu lintas, waktu pengantaran, dan jenis kendaraan."
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
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, encoder, feature_names

model, scaler, encoder, feature_names = load_artifacts()

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
# Numerical
num_cols = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
num_scaled = scaler.transform(input_df[num_cols])
num_df = pd.DataFrame(num_scaled, columns=num_cols)

# Categorical
cat_cols = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
cat_encoded = encoder.transform(input_df[cat_cols])

if hasattr(cat_encoded, "toarray"):
    cat_encoded = cat_encoded.toarray()

cat_df = pd.DataFrame(
    cat_encoded,
    columns=encoder.get_feature_names_out(cat_cols)
)

# Gabungkan
final_input = pd.concat([num_df, cat_df], axis=1)

# Samakan urutan fitur dengan model
final_input = final_input.reindex(columns=feature_names, fill_value=0)

# =========================
# ✅ PREDICTION (FIX UTAMA DI SINI)
# =========================
if st.button("🔮 Predict Delivery Time"):
    # UBAH KE NUMPY ARRAY AGAR TIDAK ERROR FEATURE NAMES
    prediction = model.predict(final_input.values)[0]

    st.success(f"⏱ Estimated Delivery Time: **{prediction:.1f} minutes**")
    st.info(f"📦 ETA Range: **{prediction - 5:.0f} – {prediction + 5:.0f} minutes**")

# =========================
# FOOTER
# =========================
st.caption("Model: Linear Regression | Use case: Food Delivery ETA Prediction")
