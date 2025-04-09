import streamlit as st
import pickle as pkl
import pandas as pd

st.title("Customer Chrun Prediction")

st.subheader("Firts, Customer Info", divider=True)
Gender=st.selectbox("Gender",options=["Male","Female"])
SeniorCitizen=st.selectbox("Senior", options=[0,1])
Partner=st.selectbox("partner",options=["Yes","No"])
Dependents=st.selectbox("Dependents",options=["Yes","No"])
tenure=st.number_input("Tenure",min_value=0.0, max_value=72.0)

st.subheader("Second, Service Info", divider=True)
PhoneService=st.selectbox("PhoneService",options=["Yes","No"])
MultipleLines=st.selectbox("MultipleLines",options=["Yes","No","No phone service"])
InternetService=st.selectbox("InternetService",options=['DSL', 'Fiber optic', 'No'])
OnlineSecurity=st.selectbox("OnlineSecurity",options=['No', 'Yes', 'No internet service'])
OnlineBackup=st.selectbox("OnlineBackup",options=['No', 'Yes', 'No internet service'])
DeviceProtection=st.selectbox("DeviceProtection",options=['No', 'Yes', 'No internet service'])
TechSupport=st.selectbox("TechSupport",options=['No', 'Yes', 'No internet service'])
StreamingTV=st.selectbox("StreamingTV",options=['No', 'Yes', 'No internet service'])
StreamingMovies=st.selectbox("StreamingMovies",options=['No', 'Yes', 'No internet service'])

st.subheader("Third, Contract Info", divider=True)
Contract=st.selectbox("Contract",options=['Month-to-month', 'One year', 'Two year'])
PaperlessBilling=st.selectbox("PaperlessBilling",options=["Yes","No"])
PaymentMethod=st.selectbox("PaymentMethod",options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'])

st.subheader("Fourth, Billing Info", divider=True)
MonthlyCharges=st.number_input("MonthlyCharges",min_value=18.25, max_value=118.75)
TotalCharges=st.number_input("TotalCharges",min_value=0.0, max_value=1000.0)

encoders=pkl.load(open("C:/Users/DELL/Documents/Projects/Intern Intelligence/Customr Chrun Prediction/Artifacts/encoders.pkl","rb"))
with open("C:/Users/DELL/Documents/Projects/Intern Intelligence/Customr Chrun Prediction/Artifacts/rfc.pkl", "rb") as f:
    model_data = pkl.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]

input_data = {
    'gender': Gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}
input_data_df=pd.DataFrame([input_data])
# encode categorical featires using teh saved encoders
for column, encoder in encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])

# make a prediction
prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

st.text(prediction)

# results
st.text(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
st.text(f"Prediciton Probability: {pred_prob}")