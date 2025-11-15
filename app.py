import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# -----------------------------------------------------------
# Load Models
# -----------------------------------------------------------
heart_model = joblib.load("hearts_model.joblib")
heart_scaler = joblib.load("hearts_scaler.joblib")

lung_model = joblib.load("lungs_model.joblib")
lung_scaler = joblib.load("lungs_scaler.joblib")

# -----------------------------------------------------------
# App Config
# -----------------------------------------------------------
st.set_page_config(page_title="Health Prediction System", page_icon="💓", layout="wide")

# -----------------------------------------------------------
# Load / Create User Database
# -----------------------------------------------------------
USER_DB = "users.csv"
if not os.path.exists(USER_DB):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_DB, index=False)

def save_user(username, password):
    df = pd.read_csv(USER_DB)
    if username in df["username"].values:
        return False
    df.loc[len(df)] = [username, password]
    df.to_csv(USER_DB, index=False)
    return True

def validate_login(username, password):
    df = pd.read_csv(USER_DB)
    return ((df["username"] == username) & (df["password"] == password)).any()

# -----------------------------------------------------------
# CSS Styling
# -----------------------------------------------------------
st.markdown("""
    <style>
    .banner-img {
        width: 100%;
        border-radius: 15px;
        margin-bottom: 20px;
    }
    .title-main {
        text-align: center;
        font-size: 45px;
        font-weight: bold;
        color: #d63384;
    }
    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #6c757d;
        margin-bottom: 30px;
    }
    .card {
        background: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #d63384;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #b02a6b;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# Signup Page
# -----------------------------------------------------------
def signup_page():
    st.image("assets/banner_health.jpg", use_column_width=True, caption="Health Prediction System")

    st.markdown("<h1 class='title-main'>Create an Account</h1>", unsafe_allow_html=True)

    with st.form("signup_form"):
        username = st.text_input("Choose Username")
        password = st.text_input("Choose Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Sign Up")

    if submit:
        if password != confirm:
            st.error("❌ Passwords do not match!")
        else:
            if save_user(username, password):
                st.success("✅ Account created successfully! Go to Login page.")
            else:
                st.error("❌ Username already exists!")

# -----------------------------------------------------------
# Login Page
# -----------------------------------------------------------
def login_page():
    st.image("assets/banner_health.jpg", use_column_width=True)

    st.markdown("<h1 class='title-main'>Welcome to Health Prediction Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Login to continue</p>", unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if validate_login(username, password):
            st.session_state.logged_in = True
            st.success("🎉 Login Successful!")
        else:
            st.error("❌ Incorrect Username or Password!")

    if st.button("Create New Account"):
        st.session_state.signup = True

# -----------------------------------------------------------
# Heart Prediction
# -----------------------------------------------------------
def heart_prediction():
    st.image("assets/heart.jpg", use_column_width=True)
    st.markdown("<h2 class='title-main'>❤️ Heart Disease Prediction</h2>", unsafe_allow_html=True)

    with st.container():
        with st.form("heart_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", 1, 120)
                sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
                cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
                trestbps = st.number_input("Resting BP", 80, 200)

            with col2:
                chol = st.number_input("Cholesterol", 100, 600)
                fbs = st.selectbox("Fasting Blood Sugar > 120", [1, 0])
                restecg = st.number_input("Resting ECG (0-2)", 0, 2)
                thalach = st.number_input("Max Heart Rate", 60, 220)

            with col3:
                exang = st.selectbox("Exercise Angina", [1, 0])
                oldpeak = st.number_input("ST Depression", 0.0, 10.0, step=0.1)
                slope = st.number_input("Slope (0-2)", 0, 2)
                ca = st.number_input("Major Vessels (0-4)", 0, 4)
                thal = st.number_input("Thal (0-3)", 0, 3)

            submit = st.form_submit_button("🔍 Predict Heart Disease")

        if submit:
            data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]])
            scaled = heart_scaler.transform(data)
            pred = heart_model.predict(scaled)

            if pred[0] == 0:
                st.success("✅ No Heart Disease Detected!")
            else:
                st.error("⚠️ Heart Disease Detected!")

# -----------------------------------------------------------
# Lung Prediction
# -----------------------------------------------------------
def lung_prediction():
    st.image("assets/lungs.jpg", use_column_width=True)
    st.markdown("<h2 class='title-main'>🫁 Lung Disease Prediction</h2>", unsafe_allow_html=True)

    with st.form("lung_form"):
        cols = st.columns(4)
        fields = [
            "age","gender","air_pollution","alcohol_use","dust_allergy","occupational_hazards",
            "genetic_risk","chronic_lung_disease","balanced_diet","obesity","smoking",
            "passive_smoker","chest_pain","coughing_blood","fatigue","weight_loss",
            "shortness_breath","wheezing","swallowing","clubbing","cold","dry_cough","snoring"
        ]
        values = []

        for i, field in enumerate(fields):
            with cols[i % 4]:
                values.append(st.number_input(field.replace("_", " ").title(), 0, 7))

        level = st.selectbox("Level", [2, 1, 0])
        values.append(level)

        submit = st.form_submit_button("🔍 Predict Lung Disease")

        if submit:
            data = np.array([values])
            scaled = lung_scaler.transform(data)
            pred = lung_model.predict(scaled)

            if pred[0] == 0:
                st.success("✅ No Lung Disease Detected!")
            else:
                st.error("⚠️ Lung Disease Detected!")

# -----------------------------------------------------------
# Main Controller
# -----------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "signup" not in st.session_state:
    st.session_state.signup = False

if st.session_state.logged_in:
    st.sidebar.image("assets/banner_health.jpg")
    st.sidebar.title("🩺 Prediction Menu")
    menu = st.sidebar.radio("Select Option", ["Heart Disease", "Lung Disease", "Logout"])

    if menu == "Heart Disease":
        heart_prediction()
    elif menu == "Lung Disease":
        lung_prediction()
    elif menu == "Logout":
        st.session_state.logged_in = False
        st.experimental_rerun()

else:
    if st.session_state.signup:
        signup_page()
    else:
        login_page()
