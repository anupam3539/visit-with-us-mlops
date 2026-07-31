from pathlib import Path
import joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🌿", layout="wide")
MODEL_PATH = Path(__file__).with_name("tourism_model.joblib")
@st.cache_resource
def load_model(): return joblib.load(MODEL_PATH)
model = load_model()

st.title("🌿 Wellness Tourism Package Predictor")
st.caption("Estimate whether a customer is likely to purchase before the sales call.")
with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 100, 35)
        contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        city = st.selectbox("City tier", [1, 2, 3])
        duration = st.number_input("Pitch duration (minutes)", 0.0, 120.0, 15.0)
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        persons = st.number_input("Persons visiting", 1, 10, 2)
        followups = st.number_input("Follow-ups", 0, 10, 3)
        product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        stars = st.selectbox("Preferred property stars", [3, 4, 5])
        marital = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
        trips = st.number_input("Annual trips", 0, 30, 2)
    with c3:
        passport = st.selectbox("Has passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
        satisfaction = st.slider("Pitch satisfaction", 1, 5, 3)
        car = st.selectbox("Owns car", [0, 1], format_func=lambda x: "Yes" if x else "No")
        children = st.number_input("Children visiting", 0, 10, 0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        income = st.number_input("Monthly income", 0.0, 500000.0, 25000.0, step=1000.0)
    submitted = st.form_submit_button("Predict purchase likelihood", type="primary")
if submitted:
    row = pd.DataFrame([{"Age": age, "TypeofContact": contact, "CityTier": city,
        "DurationOfPitch": duration, "Occupation": occupation, "Gender": gender,
        "NumberOfPersonVisiting": persons, "NumberOfFollowups": followups,
        "ProductPitched": product, "PreferredPropertyStar": stars, "MaritalStatus": marital,
        "NumberOfTrips": trips, "Passport": passport, "PitchSatisfactionScore": satisfaction,
        "OwnCar": car, "NumberOfChildrenVisiting": children, "Designation": designation,
        "MonthlyIncome": income}])
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    st.metric("Purchase probability", f"{probability:.1%}")
    (st.success if prediction else st.info)("Likely to purchase — prioritize follow-up." if prediction else "Less likely to purchase — consider nurturing first.")
    with st.expander("Input record"): st.dataframe(row, use_container_width=True)
st.caption("Decision support only; periodically monitor model performance and customer-segment fairness.")
