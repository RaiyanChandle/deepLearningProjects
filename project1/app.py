import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os

# ================== PATH SETUP ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
GENDER_PATH = os.path.join(BASE_DIR, "le_gender.pkl")
GEO_PATH = os.path.join(BASE_DIR, "ohe_geo.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ================== LOAD ASSETS ==================
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(GENDER_PATH, "rb") as f:
        le_gender = pickle.load(f)
    with open(GEO_PATH, "rb") as f:
        ohe_geo = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, le_gender, ohe_geo, scaler

model, le_gender, ohe_geo, scaler = load_assets()

# ================== STREAMLIT UI ==================
st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", ohe_geo.categories_[0])
gender = st.selectbox("Gender", le_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance", min_value=0.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# ================== DATA PREP ==================
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [le_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
})

geo_encoded = ohe_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=ohe_geo.get_feature_names_out(["Geography"])
)

input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

input_scaled = scaler.transform(input_data)

# ================== PREDICTION ==================
prediction = model.predict(input_scaled)
probability = float(prediction[0][0])

st.subheader(f"Churn Probability: {probability:.2f}")

if probability > 0.5:
    st.error("The customer is likely to churn.")
else:
    st.success("The customer is not likely to churn.")
