import streamlit as st
st.set_page_config(page_title="Bangalore Home Price Prediction", page_icon="🏠", layout="centered")

import pickle
import json
import numpy as np
import os

# Load model and columns
@st.cache_resource
def load_artifacts():
    artifacts_dir = os.path.join(os.path.dirname(__file__), 'server', 'artifacts')
    with open(os.path.join(artifacts_dir, 'banglore_home_prices_model_columns.json'), 'r') as f:
        data_columns = json.load(f)['data_columns']
    with open(os.path.join(artifacts_dir, 'banglore_home_prices_model.pickle'), 'rb') as f:
        model = pickle.load(f)
    locations = data_columns[3:]
    return model, data_columns, locations

model, data_columns, locations = load_artifacts()

def predict_price(location, sqft, bhk, bath):
    try:
        loc_index = data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(model.predict([x])[0], 2)

# --- UI ---
st.title("🏠 Bangalore Home Price Prediction")
st.markdown("Estimate the price of a home in Bangalore based on area, bedrooms, bathrooms, and location.")

col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("Area (Square Feet)", min_value=100, max_value=50000, value=1000, step=50)
    bhk = st.selectbox("BHK (Bedrooms)", options=[1, 2, 3, 4, 5], index=1)

with col2:
    bath = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5], index=1)
    location = st.selectbox("Location", options=sorted([loc.title() for loc in locations]))

if st.button("Estimate Price", type="primary", use_container_width=True):
    price = predict_price(location, sqft, bhk, bath)
    if price >= 100:
        st.success(f"### Estimated Price: ₹ {price / 100:.2f} Crore")
    else:
        st.success(f"### Estimated Price: ₹ {price} Lakh")
