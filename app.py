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
# streamlit_app.py
import streamlit as st
import pandas as pd
import os
import re
from pathlib import Path

# ------------------ File path (stable) ------------------
def get_user_file_path(filename="users.csv"):
    # Prefer same folder as this file if available, otherwise use current working dir
    try:
        base = Path(__file__).resolve().parent
    except NameError:
        base = Path.cwd()
    base.mkdir(parents=True, exist_ok=True)
    return str(base / filename)

USER_FILE = get_user_file_path("users.csv")

# Create file if not exists
if not os.path.exists(USER_FILE):
    df_init = pd.DataFrame(columns=["username", "password"])
    df_init.to_csv(USER_FILE, index=False)

# ------------------ Utility functions ------------------
def read_users():
    # Always read as strings and fill NA with empty strings
    df = pd.read_csv(USER_FILE, dtype=str).fillna("")
    # Ensure columns exist
    if "username" not in df.columns or "password" not in df.columns:
        df = pd.DataFrame(columns=["username", "password"])
    return df

def save_user(username, password):
    username = str(username).strip()
    password = str(password)
    df = read_users()

    # protect against empty username
    if username == "":
        return False

    # check exists (case-sensitive)
    if username in df["username"].astype(str).str.strip().values:
        return False

    new_row = pd.DataFrame([[username, password]], columns=["username", "password"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(USER_FILE, index=False)
    return True

def validate_user(username, password):
    username = str(username).strip()
    password = str(password)
    df = read_users()
    # compare stripped username to avoid accidental whitespace mismatch
    matches = df[
        (df["username"].astype(str).str.strip() == username) &
        (df["password"].astype(str) == password)
    ]
    return not matches.empty

# ------------------ Password Rule Check ------------------
def check_password_rules(pw):
    pw = str(pw)
    return {
        "has_upper": bool(re.search(r"[A-Z]", pw)),
        "has_lower": bool(re.search(r"[a-z]", pw)),
        "has_digit": bool(re.search(r"[0-9]", pw)),
        "has_special": bool(re.search(r"[!@#$%^&*()_+=\-]", pw)),
        "len_ok": 4 <= len(pw) <= 12
    }

# ------------------ Streamlit app ------------------
def login_page():
    # initialize session state keys
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    st.markdown(
        """
        <style>
        .title-main {
            text-align:center;
            color:#d63384;
            font-size: 34px;
            font-weight: bold;
        }
        .sub {
            text-align:center;
            color:#6c757d;
            font-size: 18px;
        }
        .card {
            max-width:700px;
            margin:auto;
            padding:18px;
            border-radius:8px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h1 class='title-main'>💓 Health & Lungs Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # if logged in show simple dashboard
    if st.session_state.logged_in:
        st.success(f"✅ Logged in as: {st.session_state.username}")
        st.write("Welcome — you can now access the Health & Lungs Prediction features.")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    page = st.radio("Select Option", ["Login", "Sign Up"])

    # -------- LOGIN --------
    if page == "Login":
        st.markdown("<p class='sub'>Login to your account</p>", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

        if submit:
            if username.strip() == "" or password == "":
                st.error("Please enter both username and password.")
            elif validate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username.strip()
                st.success("✅ Login successful!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid username or password")

    # -------- SIGNUP --------
    else:
        st.markdown("<p class='sub'>Create your new account</p>", unsafe_allow_html=True)

        with st.form("signup_form"):
            new_user = st.text_input("Choose Username")
            new_pass = st.text_input("Choose Password", type="password")

            # show rules live (based on current input)
            checks = check_password_rules(new_pass)
            st.markdown("### Password Rules")
            st.markdown(f"- {'✅' if checks['has_upper'] else '⏩'} Must contain **Uppercase (A-Z)**")
            st.markdown(f"- {'✅' if checks['has_lower'] else '⏩'} Must contain **Lowercase (a-z)**")
            st.markdown(f"- {'✅' if checks['has_digit'] else '⏩'} Must contain **Digit (0-9)**")
            st.markdown(f"- {'✅' if checks['has_special'] else '⏩'} Must contain **Special char (!@#$%)**")
            st.markdown(f"- {'✅' if checks['len_ok'] else '⏩'} Length **4–12 characters**")
            signup = st.form_submit_button("Sign Up")

        if signup:
            # re-evaluate rules at submit time to be safe
            checks = check_password_rules(new_pass)
            if new_user.strip() == "":
                st.error("Please enter a username.")
            elif not all(checks.values()):
                st.error("❌ Password does NOT meet all the rules.")
            else:
                ok = save_user(new_user, new_pass)
                if ok:
                    st.success("🎉 Account created successfully! Now login.")
                else:
                    st.error("⚠️ Username already exists or invalid. Try another one.")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    login_page()


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
    gender = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
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

