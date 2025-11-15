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
# ------------------ User Data Handling ------------------
# ------------------ Beautiful Login + Signup UI ------------------
# ------------------ Login + Signup Page (PRO UI) ------------------
def login_page():
    st.markdown("""
        <style>
        /* Center the login container */
        .login-container {
            max-width: 420px;
            margin: auto;
            margin-top: 50px;
            padding: 30px;
            background: #ffffff;
            border-radius: 18px;
            box-shadow: 0px 10px 25px rgba(0,0,0,0.12);
        }

        .login-title {
            text-align: center;
            font-size: 32px;
            font-weight: 800;
            color: #d63384;
        }

        .login-sub {
            text-align: center;
            font-size: 16px;
            color: #6c757d;
            margin-bottom: 15px;
        }

        .stButton>button {
            width: 100%;
            border-radius: 12px;
            padding: 10px;
            background-color: #d63384 !important;
            color: white !important;
            font-size: 18px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #b02a6b !important;
        }

        .radio-label {
            text-align:center;
            padding:10px;
            font-size: 18px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='login-title'>💓 Health & Lungs Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p class='login-sub'>Your secure health prediction dashboard</p>", unsafe_allow_html=True)

    # Centered container card
    with st.container():
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)

        page = st.radio(" ", ["Login", "Sign Up"], horizontal=True)

        # ---------------- LOGIN ----------------
        if page == "Login":
            st.markdown("<p class='login-sub'>Login to continue</p>", unsafe_allow_html=True)

            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if validate_user(username, password):
                    st.session_state.logged_in = True
                    st.success("✅ Logged in successfully!")
                    st.rerun()
                else:
                    st.error("❌ Wrong username or password")

        # ---------------- SIGN UP ----------------
        else:
            st.markdown("<p class='login-sub'>Create your account</p>", unsafe_allow_html=True)

            new_user = st.text_input("Choose Username")
            new_pass = st.text_input("Choose Password", type="password")

            if st.button("Sign Up"):
                if save_user(new_user, new_pass):
                    st.success("🎉 Account created! You can now login.")
                else:
                    st.error("⚠ Username already taken. Try another one.")

        st.markdown("</div>", unsafe_allow_html=True)



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

    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.number_input("Gender (1=Male, 2=Female)", min_value=1, max_value=2, value=1)
    air_pollution = st.number_input("Air Pollution (0-7)", min_value=0, max_value=7, value=3)
    alcohol_use = st.number_input("Alcohol Use (0-7)", min_value=0, max_value=7, value=1)
    dust_allergy = st.number_input("Dust Allergy (0-7)", min_value=0, max_value=7, value=2)
    occupational_hazards = st.number_input("Occupational Hazards (0-7)", min_value=0, max_value=7, value=2)
    genetic_risk = st.number_input("Genetic Risk (0-7)", min_value=0, max_value=7, value=2)
    chronic_lung_disease = st.number_input("Chronic Lung Disease (0=No, 1=Yes)", min_value=0, max_value=1, value=0)
    balanced_diet = st.number_input("Balanced Diet (0-7)", min_value=0, max_value=7, value=5)
    obesity = st.number_input("Obesity (0-7)", min_value=0, max_value=7, value=2)
    smoking = st.number_input("Smoking (0=No, 1=Yes)", min_value=0, max_value=1, value=0)
    passive_smoker = st.number_input("Passive Smoker (0=No, 1=Yes)", min_value=0, max_value=1, value=0)
    chest_pain = st.number_input("Chest Pain (0-7)", min_value=0, max_value=7, value=1)
    coughing_blood = st.number_input("Coughing of Blood (0-7)", min_value=0, max_value=7, value=0)
    fatigue = st.number_input("Fatigue (0-7)", min_value=0, max_value=7, value=1)
    weight_loss = st.number_input("Weight Loss (0-7)", min_value=0, max_value=7, value=0)
    shortness_breath = st.number_input("Shortness of Breath (0-7)", min_value=0, max_value=7, value=1)
    wheezing = st.number_input("Wheezing (0-7)", min_value=0, max_value=7, value=1)
    swallowing = st.number_input("Swallowing Difficulty (0-7)", min_value=0, max_value=7, value=0)
    clubbing = st.number_input("Clubbing of Finger Nails (0-7)", min_value=0, max_value=7, value=0)
    cold = st.number_input("Frequent Cold (0-7)", min_value=0, max_value=7, value=1)
    dry_cough = st.number_input("Dry Cough (0-7)", min_value=0, max_value=7, value=1)
    snoring = st.number_input("Snoring (0-7)", min_value=0, max_value=7, value=1)
    level = st.selectbox("Level (2=High, 1=Medium, 0=Low",[2, 1, 0])

    
    if st.button("🔍 Predict Lung Disease"):
        input_data = np.array([[age, gender, air_pollution, alcohol_use, dust_allergy, occupational_hazards, genetic_risk, chronic_lung_disease,
                                balanced_diet, obesity, smoking, passive_smoker, chest_pain,
                                coughing_blood, fatigue, weight_loss, shortness_breath,
                                wheezing, swallowing, clubbing, cold, dry_cough, snoring, level]])
        
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
        st.rerun()

