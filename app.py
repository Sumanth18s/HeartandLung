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
# ------------------ Login + Signup Page (PRO UI) ------------------
# ----------------- PAGE CONFIG -----------------
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Health & Lungs Portal", page_icon="💓", layout="centered")

# ----------------- CUSTOM CSS -----------------
st.markdown("""
<style>

/* Background */
body, .stApp {
    background: linear-gradient(135deg, #0f0f11, #1b1c1f);
    color: white;
    font-family: 'Poppins', sans-serif;
}

/* Title */
.main-title {
    font-size: 40px;
    text-align: center;
    font-weight: 700;
    color: #ff4b6e;
    margin-top: 20px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #cccccc;
}

/* Centered Card */
.login-card {
    background: rgba(255, 255, 255, 0.05);
    padding: 30px 40px;
    border-radius: 16px;
    width: 430px;
    margin: auto;
    margin-top: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 0px 20px rgba(255, 75, 110, 0.2);
    animation: fadein 1.2s ease;
}

/* Input & Button Tweaks */
.stTextInput>div>div>input {
    background-color: rgba(255,255,255,0.08);
    border-radius: 10px;
    color: white;
}

.stButton>button {
    background: #ff4b6e;
    color: white;
    padding: 10px 20px;
    border-radius: 10px;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    background: #ff3357;
}

/* Fade animation */
@keyframes fadein {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

</style>
""", unsafe_allow_html=True)
# ------------------ User Data Handling ------------------
USER_FILE = "users.csv"

# Create CSV if not exists
if not os.path.exists(USER_FILE):
    df = pd.DataFrame(columns=["username", "password"])
    df.to_csv(USER_FILE, index=False)


def save_user(username, password):
    df = pd.read_csv(USER_FILE)
    if username in df["username"].values:
        return False

    df.loc[len(df)] = [username, password]
    df.to_csv(USER_FILE, index=False)
    return True


def validate_user(username, password):
    df = pd.read_csv(USER_FILE)
    return not df[(df["username"] == username) &
                  (df["password"] == password)].empty
# ------------------ Login + Signup Page ------------------
def login_page():
    st.markdown("<h1 class='main-title'>💓 Health & Lungs Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Your secure health dashboard</p>", unsafe_allow_html=True)

    page = st.radio("", ["Login", "Sign Up"], horizontal=True)

    # Container Card
    with st.container():
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)

        # -------- LOGIN --------
        if page == "Login":
            st.markdown("<p class='sub-title'>Login to continue</p>", unsafe_allow_html=True)

            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")

            if submit:
                if validate_user(username, password):
                    st.session_state.logged_in = True
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")

        # -------- SIGNUP --------
        else:
            st.markdown("<p class='sub-title'>Create your new account</p>", unsafe_allow_html=True)

            with st.form("signup_form"):
                new_user = st.text_input("Choose Username")
                new_pass = st.text_input("Choose Password", type="password")
                signup = st.form_submit_button("Sign Up")

            if signup:
                if save_user(new_user, new_pass):
                    st.success("🎉 Account created! Please login.")
                else:
                    st.error("⚠️ Username already exists!")

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

