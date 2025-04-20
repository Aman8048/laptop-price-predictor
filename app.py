import streamlit as st
import pandas as pd
import pickle

# Load pipeline
pipe = pickle.load(open("pipe.pkl", "rb"))

st.title("💻 Laptop Price Predictor")

# --- User Inputs ---
brand = st.selectbox("Brand", ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer"])
ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64])
storage = st.selectbox("Storage (GB)", [128, 256, 512, 1024, 2048])
processor = st.selectbox("Processor", ["i3", "i5", "i7", "i9", "Ryzen 5", "Ryzen 7"])
os = st.selectbox("Operating System", ["Windows", "macOS", "Linux", "DOS"])
screen_size = st.number_input("Screen Size (inches)", min_value=11.0, max_value=18.0, value=15.6, step=0.1)
touchscreen = st.selectbox("Touchscreen", ["Yes", "No"])

# --- Convert to DataFrame ---
input_df = pd.DataFrame([{
    "Brand": brand,
    "RAM": ram,
    "Storage": storage,
    "Processor": processor,
    "OS": os,
    "Screen Size": screen_size,
    "Touchscreen": touchscreen
}])

# Optional: Debug
st.write("🔍 Input DataFrame Preview:")
st.dataframe(input_df)

# --- Predict Button ---
if st.button("Predict Price"):
    try:
        prediction = pipe.predict(input_df)[0]
        st.success(f"🎯 Estimated Laptop Price: ₹{int(prediction)}")
    except ValueError as e:
        st.error(f"❌ Prediction failed. Error: {e}")
