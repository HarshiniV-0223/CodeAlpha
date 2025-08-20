import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and features
model = pickle.load(open('car_price_model.pkl', 'rb'))
model_features = pickle.load(open('model_features.pkl', 'rb'))

st.title("🚗 Car Price Prediction App")
st.markdown("Enter the details of the car below to predict its selling price.")

# Input fields
year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, value=5.0)
driven_kms = st.number_input("Driven Kilometers", min_value=0, value=50000)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
brand = st.text_input("Enter Car Brand").strip().title()


# Automatically assign goodwill based on brand
brand_info = {
    'Maruti': [82, 7],
    'Hyundai': [100, 8],
    'Toyota': [120, 9],
    'BMW': [190, 10],
    'Tata': [85, 6],
    'Ford': [110, 7],
    'Chevrolet': [95, 6],
    'Renault': [88, 6],
    'Kia': [98, 7],
    'Nissan': [90, 6],
    'Volkswagen': [110, 8],
    'Skoda': [105, 7],
    'Mahindra': [100, 7],
    'Honda': [105, 8],
    'MG': [110, 7],
    'Jeep': [130, 8],
    'Mercedes': [180, 10],
    'Audi': [185, 10],
    'Lexus': [170, 9],
    'Volvo': [160, 9],
    'Porsche': [250, 10],
    'Jaguar':[200,9]
}

if st.button("Predict Price"):
    # Prepare input dict
    input_dict = {
        'Year': year,
        'Present_Price': present_price,
        'Driven_kms': driven_kms,
        'Owner': owner,
        'Fuel_Type': {'Petrol': 1, 'Diesel': 0, 'CNG': 2}[fuel_type],
        'Seller_type': 1 if seller_type == 'Dealer' else 0,
        'Transmission': 1 if transmission == 'Manual' else 0,
        'Brand':brand
    }
    default_horsepower = 100
    default_goodwill = 5

    # Use values from brand_info if available, else fallback to defaults
    if brand in brand_info:
        horsepower, goodwill = brand_info[brand]
    else:
        st.warning(f"⚠ Brand '{brand}' not found in our database. Using default values.")
        horsepower = default_horsepower
        goodwill = default_goodwill
    input_dict['Horsepower'] = horsepower
    input_dict['Goodwill']=goodwill

    input_df = pd.DataFrame([input_dict])

    # Fill any missing columns
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[model_features]

    # Predict
    predicted_price = model.predict(input_df)[0]
    predicted_price = max(0, predicted_price)  # Clamp negative prices to zero
    st.write(f"🔧 Horsepower: {horsepower} HP")
    st.write(f"🏷 Brand Goodwill: {goodwill}/10")

    st.success(f"💰 Estimated Selling Price: ₹ {predicted_price:.2f} Lakhs")
    st.caption("ℹ Horsepower and goodwill values are estimated for academic use only and do not reflect actual brand metrics.")

