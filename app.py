import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

# Load the trained model pipeline (assuming it's saved as 'pipe.pkl')
pipe = pickle.load(open('pipe.pkl', 'rb'))

# --- Streamlit UI ---
st.title('Laptop Price Prediction App')

# Inputs
company = st.selectbox("Brand", ['Apple', 'Dell', 'HP', 'Lenovo', 'Acer', 'Asus', 'Microsoft'])
typename = st.selectbox("Laptop Type", ['Ultrabook', 'Gaming', 'Notebook', 'Business', 'Workstation'])
ram = st.selectbox("RAM (GB)", [4, 6, 8, 12, 16, 32])
weight = st.slider("Weight (kg)", 0.5, 5.0, 2.5)
touchscreen = st.selectbox("Touchscreen", ['Yes', 'No'])
ips = st.selectbox("IPS", ['Yes', 'No'])
ppi = st.number_input("PPI", min_value=100, max_value=600, step=10, value=150)
cpu = st.selectbox("CPU", ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Intel Core i9', 
                           'AMD Ryzen 5', 'AMD Ryzen 7', 'AMD Ryzen 9', 'Other'])
hdd = st.number_input("HDD (GB)", min_value=0, max_value=2000, step=10, value=500)
ssd = st.number_input("SSD (GB)", min_value=0, max_value=2000, step=10, value=512)
gpu = st.selectbox("GPU Brand", ['NVIDIA', 'AMD', 'Intel', 'Other'])
os = st.selectbox("Operating System", ['Windows', 'MacOS', 'Linux'])

# --- Prediction Button ---
if st.button('Predict Price'):
    try:
        # --- Encoding categorical inputs correctly ---
        touchscreen_encoded = 1 if touchscreen == 'Yes' else 0
        ips_encoded = 1 if ips == 'Yes' else 0
        
        # Handle OS encoding based on training assumptions
        os_mapping = {'Windows': 1, 'MacOS': 2, 'Linux': 3}
        os_encoded = os_mapping.get(os, 0)  # Default to 0 if unknown
        
        # Handle CPU brand extraction (assuming the model was trained on just the brand)
        cpu_brand = cpu.split()[0]  # Extract the brand name only (e.g., 'Intel' or 'AMD')

        # Handle GPU encoding (assuming 'Intel', 'NVIDIA', 'AMD', 'Other' were part of the model's training categories)
        gpu_mapping = {'NVIDIA': 1, 'AMD': 2, 'Intel': 3, 'Other': 4}
        gpu_encoded = gpu_mapping.get(gpu, 0)  # Default to 0 if unknown

        # Prepare the input dataframe for prediction
        input_df = pd.DataFrame([{
            'Company': company,
            'TypeName': typename,
            'Ram': ram,
            'Weight': weight,
            'Touchscreen': touchscreen_encoded,  # Encoded touch screen value
            'Ips': ips_encoded,  # Encoded IPS value
            'ppi': ppi,
            'Cpu brand': cpu_brand,  # Only the brand, not the full string
            'HDD': hdd,
            'SSD': ssd,
            'Gpu brand': gpu_encoded,  # Encoded GPU value
            'os': os_encoded  # Encoded OS value
        }])

        # --- Check if the model was trained with a LabelEncoder for categorical features ---
        # For example, if you used a LabelEncoder on CPU/GPU/OS in training, we need to fit it here similarly
        encoder_cpu = LabelEncoder()
        encoder_gpu = LabelEncoder()
        encoder_os = LabelEncoder()

        # Assuming that the model training included fitting these label encoders:
        # Fit encoders based on the possible categories
        encoder_cpu.fit(['Intel', 'AMD', 'Other'])
        encoder_gpu.fit(['Intel', 'NVIDIA', 'AMD', 'Other'])
        encoder_os.fit(['Windows', 'MacOS', 'Linux'])

        # Apply the label encoders (ensure the input values are transformed similarly to how the model was trained)
        input_df['Cpu brand'] = encoder_cpu.transform(input_df['Cpu brand'])
        input_df['Gpu brand'] = encoder_gpu.transform(input_df['Gpu brand'])
        input_df['os'] = encoder_os.transform(input_df['os'])

        # --- Predict using the pipeline ---
        prediction = pipe.predict(input_df)[0]
        st.write(f"Predicted Price: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed. Error: {e:}")
