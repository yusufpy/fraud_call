import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("fraud_rf_model.pkl", "rb") as model_file:
    saved_objects = pickle.load(model_file)

rf_model = saved_objects["model"]
label_encoders = saved_objects["encoders"]

# Streamlit App Title
st.title("📞 Fraud Detection in Call Data")

# User Input Form
st.sidebar.header("Enter Call Details")

call_type = st.sidebar.selectbox("Call Type", ["Missed Call", "Incoming", "Outgoing"])
call_duration = st.sidebar.number_input("Call Duration (secs)", min_value=0)
location = st.sidebar.selectbox("Location", ["Enugu", "Kaduna", "Abeokuta", "Onitsha"])
cost = st.sidebar.number_input("Cost (NGN)", min_value=0.0, format="%.2f")
device_type = st.sidebar.selectbox("Device Type", ["Feature Phone", "Smartphone"])
suspicious_number = st.sidebar.checkbox("Suspicious Number?")
call_frequency = st.sidebar.number_input("Call Frequency (daily)", min_value=0)
unusual_location = st.sidebar.checkbox("Unusual Location?")
hour = st.sidebar.slider("Call Hour (0-23)", 0, 23)
day = st.sidebar.slider("Call Day (1-31)", 1, 31)
month = st.sidebar.slider("Call Month (1-12)", 1, 12)

# Preprocess user input
input_data = pd.DataFrame({
    "Call Type": [label_encoders["Call Type"].transform([call_type])[0]],
    "Call Duration (secs)": [call_duration],
    "Location": [label_encoders["Location"].transform([location])[0]],
    "Cost (NGN)": [cost],
    "Device Type": [label_encoders["Device Type"].transform([device_type])[0]],
    "Suspicious Number?": [int(suspicious_number)],
    "Call Frequency (daily)": [call_frequency],
    "Unusual Location?": [int(unusual_location)],
    "Hour": [hour],
    "Day": [day],
    "Month": [month],
})

# Predict fraud
if st.sidebar.button("Check Fraud"):
    prediction = rf_model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ This call is flagged as FRAUDULENT!")
    else:
        st.success("✅ This call is NOT fraudulent.")

# Run: streamlit run streamlit_app.py
