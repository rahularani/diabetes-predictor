# app.py
!pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load and prepare dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=column_names)
    return df

# Train model
@st.cache_resource
def train_models(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr_model = LogisticRegression()
    lr_model.fit(X_scaled, y)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y)

    return scaler, lr_model, rf_model

# Streamlit UI
st.title("🩺 Diabetes Prediction App")
st.write("This app uses machine learning to predict whether a person has diabetes based on health data.")

df = load_data()
scaler, lr_model, rf_model = train_models(df)

# Sidebar input
st.sidebar.header("Enter Patient Data:")

def user_input_features():
    pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, 1)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 140, 70)
    skin = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0.0, 67.0, 25.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
    age = st.sidebar.slider('Age', 10, 100, 30)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
scaled_input = scaler.transform(input_df)

# Model selection
model_option = st.radio("Choose model:", ("Logistic Regression", "Random Forest"))

if st.button("Predict"):
    if model_option == "Logistic Regression":
        prediction = lr_model.predict(scaled_input)[0]
        proba = lr_model.predict_proba(scaled_input)[0][1]
    else:
        prediction = rf_model.predict(scaled_input)[0]
        proba = rf_model.predict_proba(scaled_input)[0][1]

    st.subheader("Prediction Result:")
    st.write("**Diabetic**" if prediction == 1 else "**Not Diabetic**")
    st.write(f"Probability: **{proba:.2f}**")

st.markdown("---")
st.markdown("Developed by Rahul R and Team | Department of IT, Mailam Engineering College")
