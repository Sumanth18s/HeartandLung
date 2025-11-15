import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import plotly.express as px  # For advanced charts
import requests  # For image fetching (optional)

# ------------------ Load Models & Scalers with Caching ------------------
@st.cache_resource
def load_models():
    try:
        heart_model = joblib.load("hearts_model.joblib")
        heart_scaler = joblib.load("hearts_scaler.joblib")
        lung_model = joblib.load("lungs_model.joblib")
        lung_scaler = joblib.load("lungs_scaler.joblib")
        return heart_model, heart_scaler, lung_model, lung_scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'hearts_model.joblib', 'hearts_scaler.joblib', 'lungs_model.joblib', and 'lungs_scaler.joblib' are in the directory.")
        st.stop()

heart_model, heart_scaler, lung_model, lung_scaler = load_models()

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Advanced Health Prediction System",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Theme Management ------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def apply_theme():
    if st.session_state.theme == "dark":
        return """
        body { background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); color: white; }
        .card { background: #34495e; color: white; }
        .hero { color: white; }
        """
    else:
        return """
        body { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); color: black; }
        .card { background: white; color: black; }
        """

st.markdown(f"<style>{apply_theme()}</style>", unsafe_allow_html=True)

# ------------------ Custom CSS for Enhanced UI ------------------
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
    body { font-family: 'Roboto', sans-serif; }
    .hero { background-size: cover; height: 300px; border-radius: 15px; display: flex; align-items: center; justify-content: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); margin-bottom: 20px; animation: slideIn 1s ease-out; }
    @keyframes slideIn { from { transform: translateX(-100%); } to { transform: translateX(0); } }
    .card { border-radius: 15px; box-shadow: 0 8px 16px rgba(0,0,0,0.2); padding: 20px; margin: 10px 0; transition: all 0.3s; animation: fadeIn 0.5s ease-in; }
    .card:hover { transform: scale(1.02); box-shadow: 0 12px 24px rgba(0,0,0,0.3); }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .pulse { animation: pulse 2s infinite; }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
    .footer { text-align: center; color: #6c757d; font-size: 14px; margin-top: 50px; padding: 20px; background: #f8f9fa; border-radius: 10px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
    </style>
""", unsafe_allow_html=True)

# ------------------ Disclaimer ------------------
st.markdown("""
    <div style='text-align: center; color: #6c757d; font-size: 14px; margin-bottom: 20px; background: #fff3cd; padding: 10px; border-radius: 5px;'>
    ⚠️ <strong>Disclaimer:</strong> This app provides predictions based on machine learning models and is for informational purposes only. It is not a substitute for professional medical advice. Always consult a healthcare provider for diagnosis and treatment.
    </div>
""", unsafe_allow_html=True)

# ------------------ Utility Functions ------------------
@st.cache_data
def get_image_urls():
    return [
        "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80",
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80",
        "https://images.unsplash.com/photo-1559757175-0eb30cd8c063?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80"
    ]

def display_hero():
    images = get_image_urls()
    if "hero_index" not in st.session_state:
        st.session_state.hero_index = 0
    st.session_state.hero_index = (st.session_state.hero_index + 1) % len(images)
    st.markdown(f"<div class='hero' style='background-image: url({images[st.session_state.hero_index]});'><h1>💓 Advanced Health Prediction Portal</h1></div>", unsafe_allow_html=True)

# ------------------ Login System ------------------
def login_page():
    display_hero()
    st.markdown("<p style='text-align:center; color:#6c757d; font-size:20px;'>Secure Login to Access Predictions</p>", unsafe_allow_html=True)

    with st.form("login_form"):
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username", placeholder="Enter username")
        with col2:
            password = st.text_input("Password", type="password", placeholder="Enter password")
        submit = st.form_submit_button("🔐 Login")

    if submit:
        correct_username = os.getenv("USERNAME", "admin")
        correct_password = os.getenv("PASSWORD", "1234")
        if username == correct_username and password == correct_password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Try again or contact support.")

    st.markdown("[Forgot Password?](mailto:support@example.com) (Placeholder - implement reset logic)")

# ------------------ Sidebar with Enhancements ------------------
def setup_sidebar():
    with st.sidebar:
        st.title("🩺 Navigation")
        if st.session_state.logged_in:
            # User Profile
            if "user_profile" not in st.session_state:
                st.session_state.user_profile = {"name": "User", "age": 30, "avatar": "https://via.placeholder.com/100"}
            st.image(st.session_state.user_profile["avatar"], width=80, caption=st.session_state.user_profile["name"])
            if st.button("Edit Profile"):
                with st.form("profile_form"):
                    name = st.text_input("Name", value=st.session_state.user_profile["name"])
                    age = st.number_input("Age", 1, 120, value=st.session_state.user_profile["age"])
                    avatar = st.text_input("Avatar URL", value=st.session_state.user_profile["avatar"])
                    save = st.form_submit_button("Save")
                    if save:
                        st.session_state.user_profile = {"name": name, "age": age, "avatar": avatar}
                        st.success("Profile updated!")
            
            # Theme Toggle
            theme_toggle = st.toggle("Dark Mode", value=(st.session_state.theme == "dark"))
            if theme_toggle != (st.session_state.theme == "dark"):
                st.session_state.theme = "dark" if theme_toggle else "light"
                st.rerun()
            
            # Live Clock
            st.write(f"🕒 {datetime.now().strftime('%H:%M:%S')}")
            
            choice = st.radio("Choose Page:", ("Dashboard", "Heart Disease", "Lung Disease", "Health Tips", "Help", "Logout"))
        else:
            choice = "Login"
        return choice

# ------------------ Dashboard/Home Page ------------------
def dashboard():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:#d63384;'>🏠 Your Health Dashboard</h2>", unsafe_allow_html=True)
    
    # Stats Grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", len(st.session_state.get("prediction_history", [])))
    with col2:
        st.metric("Health Score", "85%")  # Placeholder - calculate based on history
    with col3:
        st.metric("Last Prediction", st.session_state.get("last_pred", "None"))
    
    # Interactive Chart
    if "prediction_history" in st.session_state and st.session_state.prediction_history:
        df = pd.DataFrame(st.session_state.prediction_history)
        fig = px.line(df, x="Date", y="Prediction", title="Prediction Timeline")
        st.plotly_chart(fig)
    
    # Quick Actions in Grid
    st.subheader("Quick Actions")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("❤️ Heart Prediction"):
                st.session_state.page = "Heart Disease"
                st.rerun()
        with col2:
            if st.button("🫁 Lung Prediction"):
                st.session_state.page = "Lung Disease"
                st.rerun()
        with col3:
            if st.button("💡 Health Tips"):
                st.session_state.page = "Health Tips"
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Heart Disease Prediction ------------------
def heart_prediction():
    st.markdown("<div class='card pulse'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:#dc3545;'>❤️ Heart Disease Prediction</h2>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1559757175-0eb30cd8c063?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80", caption="Heart Health Diagram", use_column_width=True)
    
    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 1, 120, value=50, help="Patient's age in years")
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.slider("Chest Pain Type (0-3)", 0, 3, 0)
            trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
            chol = st.slider("Cholesterol", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        with col2:
            restecg = st.slider("Resting ECG (0-2)", 0, 2, 0)
            thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.slider("ST Depression", 0.0, 10.0, 0.0)
            slope = st.slider("Slope (0-2)", 0, 2, 0)
            ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 0)
            thal = st.slider("Thal (0-3)", 0, 3, 0)
        
        submit = st.form_submit_button("🔍 Predict Heart Disease")
    
    if submit:
        progress = st.progress(0)
        for i in range(100):
            progress.progress(i + 1)
        sex = 1 if sex == "Male" else 0
        fbs = 1 if fbs == "Yes" else 0
        exang = 1 if exang == "Yes" else 0
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        scaled = heart_scaler.transform(input_data)
        pred = heart_model.predict(scaled)
        
        # Store and Update
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []
        st.session_state.prediction_history.append({
            "Type": "Heart", "Prediction": "Disease Detected" if pred[0] == 1 else "No Disease",
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        st.session_state.last_pred = "Heart - " + ("Disease" if pred[0] == 1 else "No Disease")
        
        if pred[0] == 0:
            st.success("✅ No Heart Disease Detected!")
            st.balloons()
            st.toast("Great! Keep up the healthy habits!", icon="🎉")
        else:
            st.error("⚠️ Heart Disease Detected!")
            st.toast("Consult a doctor immediately!", icon="⚠️")
        
        # Interactive Radar Chart
        fig = px.line_polar(r={"r": [age/120, chol/600, trestbps/200, thalach/220], "theta": ["Age", "Cholesterol", "BP", "Heart Rate"]}, title="Risk Factors")
        st.plotly_chart(fig)
        
        # Feedback Form
        with st.expander("Rate This Prediction"):
            rating = st.slider("How accurate do you think this is?", 1, 5, 3)
            comment = st.text_area("Comments")
            if st.button("Submit Feedback"):
                st.success("Thanks for your feedback!")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Lung Disease Prediction ------------------
def lung_prediction():
    st.markdown("<div class='card pulse'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:#0d6efd;'>🫁 Lung Disease Prediction</h2>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80", caption="Lung Health Diagram", use_column_width=True)
    
    with st.form("lung_form"):
        tabs = st.tabs(["Demographics", "Factors", "Symptoms"])
        with tabs[0]:
            age = st.slider("Age", 1, 120, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            passive_smoker = st.selectbox("Passive Smoker",
