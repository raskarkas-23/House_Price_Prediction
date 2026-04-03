import streamlit as st
import pandas as pd
import joblib

# Load saved items
model = joblib.load("model/model.pkl")
encoders = joblib.load("model/encoders.pkl")
features = joblib.load("model/features.pkl")

# Load dataset (for dropdown values)
df = pd.read_csv("data/house_price.csv")

st.title("🏠 House Price Prediction App")

st.write("Enter property details")

user_input = {}

# -------- AUTO INPUT GENERATION --------
for feature in features:

    if df[feature].dtype == "object":
        user_input[feature] = st.selectbox(
            feature,
            sorted(df[feature].dropna().unique())
        )
    else:
        user_input[feature] = st.number_input(
            feature,
            value=float(df[feature].mean())
        )

# -------- Prediction --------
if st.button("Predict Price"):

    input_df = pd.DataFrame([user_input])

    # Encode categorical columns
    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Price: {prediction:.2f} Lakhs")