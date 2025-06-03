import streamlit as st
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        /* (Keep your existing CSS styles here) */
    </style>
""", unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_artifacts():
    with open("Artifacts/encoders.pkl", "rb") as f:
        encoders = pkl.load(f)
    with open("Artifacts/rfc.pkl", "rb") as f:
        model_data = pkl.load(f)
    return encoders, model_data

encoders, model_data = load_artifacts()
loaded_model = model_data["model"]
feature_names = model_data["features_names"]

# Sample profile data loader
def load_sample_profile(profile_type):
    if profile_type == "low_risk":
        return {
            'Gender': "Female",
            'SeniorCitizen': 0,
            'Partner': "Yes",
            'Dependents': "Yes",
            'tenure': 60,
            'PhoneService': "Yes",
            'MultipleLines': "No",
            'InternetService': "DSL",
            'OnlineSecurity': "Yes",
            'OnlineBackup': "Yes",
            'DeviceProtection': "Yes",
            'TechSupport': "Yes",
            'StreamingTV': "No",
            'StreamingMovies': "No",
            'Contract': "Two year",
            'PaperlessBilling': "No",
            'PaymentMethod': "Bank transfer (automatic)",
            'MonthlyCharges': 50.0,
            'TotalCharges': 3000.0
       }
        for key, value in sample_data.items():
            st.session_state[key] = value
        st.experimental_rerun()   
    else:  # high_risk
        return {
            'Gender': "Male",
            'SeniorCitizen': 0,
            'Partner': "No",
            'Dependents': "No",
            'tenure': 1,
            'PhoneService': "Yes",
            'MultipleLines': "No",
            'InternetService': "Fiber optic",
            'OnlineSecurity': "No",
            'OnlineBackup': "No",
            'DeviceProtection': "No",
            'TechSupport': "No",
            'StreamingTV': "Yes",
            'StreamingMovies': "Yes",
            'Contract': "Month-to-month",
            'PaperlessBilling': "Yes",
            'PaymentMethod': "Electronic check",
            'MonthlyCharges': 100.0,
            'TotalCharges': 100.0
               }
        for key, value in sample_data.items():
            st.session_state[key] = value
        st.experimental_rerun()   
# App header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2331/2331725.png", width=100)
with col2:
    st.title("Customer Churn Prediction Dashboard")
    st.markdown("Predict customer churn probability and identify at-risk customers")

# Sidebar
with st.sidebar:
    st.header("Settings")
    show_details = st.checkbox("Show feature importance", value=True)
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This app predicts customer churn using machine learning.")

# Input form with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Customer Info", "Service Info", "Contract Info", "Billing Info"])

with tab1:
    st.subheader("Customer Demographics", divider="blue")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='input-explanation'>Select the customer's gender:</div>", unsafe_allow_html=True)
        Gender = st.selectbox("Gender", options=["Male", "Female"], 
                            key="Gender", label_visibility="collapsed")
        
        st.markdown("<div class='input-explanation'>Is the customer a senior citizen?</div>", unsafe_allow_html=True)
        SeniorCitizen = st.selectbox("Senior Citizen", options=[0, 1], 
                                   format_func=lambda x: "Yes" if x == 1 else "No",
                                   key="SeniorCitizen", label_visibility="collapsed")
    with col2:
        st.markdown("<div class='input-explanation'>Does the customer have a partner?</div>", unsafe_allow_html=True)
        Partner = st.selectbox("Has Partner", options=["Yes", "No"], 
                             key="Partner", label_visibility="collapsed")
        
        st.markdown("<div class='input-explanation'>Does the customer have dependents?</div>", unsafe_allow_html=True)
        Dependents = st.selectbox("Has Dependents", options=["Yes", "No"], 
                                key="Dependents", label_visibility="collapsed")
    
    st.markdown("<div class='input-explanation'>Tenure (months):</div>", unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12,
                      key="tenure", label_visibility="collapsed")

with tab2:
    st.subheader("Service Details", divider="blue")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='input-explanation'>Phone service:</div>", unsafe_allow_html=True)
        PhoneService = st.selectbox("Phone Service", options=["Yes", "No"], 
                                  key="PhoneService", label_visibility="collapsed")
        
        st.markdown("<div class='input-explanation'>Multiple lines:</div>", unsafe_allow_html=True)
        MultipleLines = st.selectbox("Multiple Lines", options=["Yes", "No", "No phone service"], 
                                   key="MultipleLines", label_visibility="collapsed")
        
        st.markdown("<div class='input-explanation'>Internet service:</div>", unsafe_allow_html=True)
        InternetService = st.selectbox("Internet Service", options=['DSL', 'Fiber optic', 'No'], 
                                      key="InternetService", label_visibility="collapsed")
        
        st.markdown("<div class='input-explanation'>Online security:</div>", unsafe_allow_html=True)
        OnlineSecurity = st.selectbox("Online Security", options=['No', 'Yes', 'No internet service'], 
                                     key="OnlineSecurity", label_visibility="collapsed")
    with col2:
        st.markdown("<div class='input-explanation'>Online backup:</div>", unsafe_allow_html=True)
        OnlineBackup = st.selectbox("Online Backup", options=['No', 'Yes', 'No internet service'], 
                                   key="OnlineBackup", label_visibility="collapsed")
        
        st.markdown("<div class='input-explanation'>Device protection:</div>", unsafe_allow_html=True)
        DeviceProtection = st.selectbox("Device Protection", options=['No', 'Yes', 'No internet service'], 
                                       key="DeviceProtection", label_visibility="collapsed")
        
        st.markdown("<div class='input-explanation'>Tech support:</div>", unsafe_allow_html=True)
        TechSupport = st.selectbox("Tech Support", options=['No', 'Yes', 'No internet service'], 
                                  key="TechSupport", label_visibility="collapsed")

    st.subheader("Streaming Services", divider="blue")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='input-explanation'>Streaming TV:</div>", unsafe_allow_html=True)
        StreamingTV = st.selectbox("Streaming TV", options=['No', 'Yes', 'No internet service'], 
                                  key="StreamingTV", label_visibility="collapsed")
    with col2:
        st.markdown("<div class='input-explanation'>Streaming movies:</div>", unsafe_allow_html=True)
        StreamingMovies = st.selectbox("Streaming Movies", options=['No', 'Yes', 'No internet service'], 
                                      key="StreamingMovies", label_visibility="collapsed")

with tab3:
    st.subheader("Contract Details", divider="blue")
    st.markdown("<div class='input-explanation'>Contract type:</div>", unsafe_allow_html=True)
    Contract = st.selectbox("Contract Type", options=['Month-to-month', 'One year', 'Two year'],
                          key="Contract", label_visibility="collapsed")
    
    st.markdown("<div class='input-explanation'>Paperless billing:</div>", unsafe_allow_html=True)
    PaperlessBilling = st.selectbox("Paperless Billing", options=["Yes", "No"],
                                  key="PaperlessBilling", label_visibility="collapsed")
    
    st.markdown("<div class='input-explanation'>Payment method:</div>", unsafe_allow_html=True)
    PaymentMethod = st.selectbox("Payment Method", 
                               options=['Electronic check', 'Mailed check', 
                                       'Bank transfer (automatic)', 'Credit card (automatic)'],
                               key="PaymentMethod", label_visibility="collapsed")

with tab4:
    st.subheader("Billing Information", divider="blue")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='input-explanation'>Monthly charges ($):</div>", unsafe_allow_html=True)
        MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=18.25, max_value=118.75, value=50.0,
                                       key="MonthlyCharges", label_visibility="collapsed")
    with col2:
        st.markdown("<div class='input-explanation'>Total charges ($):</div>", unsafe_allow_html=True)
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0,
                                     key="TotalCharges", label_visibility="collapsed")

# Sample profiles section
with st.expander("💡 Try Sample Customer Profiles"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load High Risk Profile"):
            sample_data = load_sample_profile("high_risk")
            for key, value in sample_data.items():
                st.session_state[key] = value
            st.rerun()
            
    with col2:
        if st.button("Load Low Risk Profile"):
            sample_data = load_sample_profile("low_risk")
            for key, value in sample_data.items():
                st.session_state[key] = value
            st.rerun()

# Prediction button
st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("Predict Churn Probability", use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

if predict_button:
    input_data = {
        'gender': st.session_state.Gender,
        'SeniorCitizen': st.session_state.SeniorCitizen,
        'Partner': st.session_state.Partner,
        'Dependents': st.session_state.Dependents,
        'tenure': st.session_state.tenure,
        'PhoneService': st.session_state.PhoneService,
        'MultipleLines': st.session_state.MultipleLines,
        'InternetService': st.session_state.InternetService,
        'OnlineSecurity': st.session_state.OnlineSecurity,
        'OnlineBackup': st.session_state.OnlineBackup,
        'DeviceProtection': st.session_state.DeviceProtection,
        'TechSupport': st.session_state.TechSupport,
        'StreamingTV': st.session_state.StreamingTV,
        'StreamingMovies': st.session_state.StreamingMovies,
        'Contract': st.session_state.Contract,
        'PaperlessBilling': st.session_state.PaperlessBilling,
        'PaymentMethod': st.session_state.PaymentMethod,
        'MonthlyCharges': st.session_state.MonthlyCharges,
        'TotalCharges': st.session_state.TotalCharges
    }
    
    input_data_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for column, encoder in encoders.items():
        input_data_df[column] = encoder.transform(input_data_df[column])
    
    # Make prediction
    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)
    churn_prob = pred_prob[0][1] * 100
    
    # Display results
    if prediction[0] == 1:
        st.markdown(f"""
        <div class="prediction-box churn">
            <h2>🚨 High Churn Risk Detected</h2>
            <p>This customer has a <strong>{churn_prob:.1f}%</strong> probability of churning.</p>
            <p>Recommended action: Immediate retention efforts needed.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box no-churn">
            <h2>✅ Low Churn Risk</h2>
            <p>This customer has only a <strong>{churn_prob:.1f}%</strong> probability of churning.</p>
            <p>Recommended action: Standard engagement recommended.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(['Churn Probability'], [churn_prob], color='#ff6b6b' if prediction[0] == 1 else '#51cf66')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)')
    ax.set_title('Churn Probability Visualization')
    st.pyplot(fig)
    
    # Feature importance
    if show_details and hasattr(loaded_model, 'feature_importances_'):
        st.subheader("Feature Importance", divider="gray")
        importances = loaded_model.feature_importances_
        indices = importances.argsort()[::-1]
        feature_importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis', ax=ax)
        ax.set_title('Top 10 Features Influencing Churn Prediction')
        st.pyplot(fig)
        
        with st.expander("View Complete Feature Importance"):
            st.dataframe(feature_importance_df.style.background_gradient(cmap='Blues'))

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>Customer Churn Prediction App • Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)
