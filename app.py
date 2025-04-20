import streamlit as st
import pandas as pd
import pickle

# Load pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title("Price Prediction App")

# Example inputs (update with your actual feature names)
brand = st.selectbox("Brand", ["Apple", "Samsung", "Xiaomi"])
ram = st.number_input("RAM (GB)", step=1)
storage = st.number_input("Storage (GB)", step=1)
# Add more fields as needed

if st.button("Predict"):
    # Use same column order as your model was trained on
    input_df = pd.DataFrame([[brand, ram, storage]], columns=['Brand', 'RAM', 'Storage'])
    print("DEBUG INPUT DF COLUMNS:", input_df.columns)

    prediction = pipe.predict(input_df)[0]
    st.success(f"Predicted Price: ₹{int(prediction):,}")


