import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ------------------ Load Models & Scalers ------------------
heart_model = joblib.load("hearts_model.joblib")
heart_scaler = joblib.load("hearts_scaler.joblib")

lung_model = joblib.load("lungs_model.joblib")
lung_scaler = joblib.load("lungs_scaler.joblib")

# ------------------ Page Config ------------------
st.set_page_config(page_title="Health Prediction System", page_icon="💓", layout="wide")

# ------------------ CSS Styling ------------------
st.markdown("""
    <style>
    body {
        background-color: #F8F9FA;
    }
    .main-title {
        text-align: center;
        color: #d63384;
        font-size: 42px !important;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        color: #6c757d;
        font-size: 20px !important;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #d63384;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        border: none;
    }
    .stButton>button:hover {
        background-color: #b02a6b;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Login System ------------------
def login_page():
    st.markdown("<h1 class='main-title'>💓 Health Prediction Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Please log in to continue</p>", unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("✅ Login successful!")
        else:
            st.error("❌ Invalid username or password")

# ------------------ Heart Disease Prediction ------------------
def heart_prediction():
    st.markdown("<h2 style='text-align:center;color:#dc3545;'>❤️ Heart Disease Prediction</h2>", unsafe_allow_html=True)
    st.write("### Enter Patient Details:")

    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
    trestbps = st.number_input("Resting Blood Pressure", 80, 200)
    chol = st.number_input("Cholesterol", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No)", [1, 0])
    restecg = st.number_input("Resting ECG (0-2)", 0, 2)
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, step=0.1)
    slope = st.number_input("Slope (0-2)", 0, 2)
    ca = st.number_input("Number of Major Vessels (0-4)", 0, 4)
    thal = st.number_input("Thal (0-3)", 0, 3)

    if st.button("🔍 Predict Heart Disease"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])
        scaled = heart_scaler.transform(input_data)
        pred = heart_model.predict(scaled)

        if pred[0] == 0:
            st.success("✅ No Heart Disease Detected!")
        else:
            st.error("⚠️ Heart Disease Detected!")

# ------------------ Lung Disease Prediction ------------------
def lung_prediction():
    st.markdown("<h2 style='text-align:center;color:#0d6efd;'>🫁 Lung Disease Prediction</h2>", unsafe_allow_html=True)
    st.write("### Enter Patient Details:")

    age = st.number_input("Age", 1, 120)
    smoking = st.selectbox("Smoking (1=Yes, 0=No)", [1, 0])
    yellow_fingers = st.selectbox("Yellow Fingers (1=Yes, 0=No)", [1, 0])
    anxiety = st.selectbox("Anxiety (1=Yes, 0=No)", [1, 0])
    peer_pressure = st.selectbox("Peer Pressure (1=Yes, 0=No)", [1, 0])
    chronic_disease = st.selectbox("Chronic Disease (1=Yes, 0=No)", [1, 0])
    fatigue = st.selectbox("Fatigue (1=Yes, 0=No)", [1, 0])
    allergy = st.selectbox("Allergy (1=Yes, 0=No)", [1, 0])
    wheezing = st.selectbox("Wheezing (1=Yes, 0=No)", [1, 0])
    alcohol = st.selectbox("Alcohol Consuming (1=Yes, 0=No)", [1, 0])
    coughing = st.selectbox("Coughing (1=Yes, 0=No)", [1, 0])
    shortness = st.selectbox("Shortness of Breath (1=Yes, 0=No)", [1, 0])
    swallowing = st.selectbox("Swallowing Difficulty (1=Yes, 0=No)", [1, 0])
    chest_pain = st.selectbox("Chest Pain (1=Yes, 0=No)", [1, 0])

    if st.button("🔍 Predict Lung Disease"):
        input_data = np.array([[age, smoking, yellow_fingers, anxiety, peer_pressure,
                                chronic_disease, fatigue, allergy, wheezing, alcohol,
                                coughing, shortness, swallowing, chest_pain]])
        scaled = lung_scaler.transform(input_data)
        pred = lung_model.predict(scaled)

        if pred[0] == 0:
            st.success("✅ No Lung Disease Detected!")
        else:
            st.error("⚠️ Lung Disease Detected!")

# ------------------ Main App ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
else:
    st.sidebar.title("🩺 Disease Prediction System")
    choice = st.sidebar.radio("Choose Prediction Type:", ("Heart Disease", "Lung Disease", "Logout"))

    if choice == "Heart Disease":
        heart_prediction()
    elif choice == "Lung Disease":
        lung_prediction()
    elif choice == "Logout":
        st.session_state.logged_in = False
        st.experimental_rerun()
