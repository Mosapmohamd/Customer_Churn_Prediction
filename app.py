import streamlit as st
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration with dark theme
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme custom CSS
st.markdown("""
    <style>
        :root {
            --primary-color: #1a1a1a;
            --secondary-color: #2d2d2d;
            --text-color: #f0f0f0;
            --accent-color: #4a89dc;
            --churn-color: #e74c3c;
            --no-churn-color: #2ecc71;
        }
        
        .main {
            background-color: var(--primary-color);
            color: var(--text-color);
        }
        
        .stButton>button {
            background-color: var(--accent-color);
            color: var(--text-color);
            border-radius: 5px;
            padding: 10px 24px;
            border: none;
            font-weight: bold;
        }
        
        .stButton>button:hover {
            background-color: #3a70c2;
        }
        
        .stSelectbox, .stNumberInput, .stSlider {
            background-color: var(--secondary-color);
            color: var(--text-color);
            border-radius: 5px;
            border: 1px solid #444;
        }
        
        .header {
            color: var(--text-color);
        }
        
        .prediction-box {
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.3);
            color: var(--text-color);
        }
        
        .churn {
            background-color: rgba(231, 76, 60, 0.15);
            border-left: 5px solid var(--churn-color);
        }
        
        .no-churn {
            background-color: rgba(46, 204, 113, 0.15);
            border-left: 5px solid var(--no-churn-color);
        }
        
        .feature-importance {
            margin-top: 30px;
        }
        
        .input-explanation {
            font-size: 0.9em;
            color: #bdc3c7;
            margin-bottom: 5px;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: var(--secondary-color);
            color: var(--text-color);
            border-radius: 5px 5px 0 0;
            padding: 10px 20px;
            border: none;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--accent-color);
        }
        
        .css-1aumxhk {
            background-color: var(--secondary-color);
        }
        
        footer {
            color: #888;
        }
        
        /* Adjust matplotlib chart colors */
        .css-1lcbmhc {
            background-color: var(--secondary-color) !important;
        }
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

# App header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2331/2331725.png", width=100)
with col2:
    st.title("Customer Churn Prediction Dashboard")
    st.markdown("Predict customer churn probability and identify at-risk customers")

# Sidebar for additional options
with st.sidebar:
    st.header("Settings")
    show_details = st.checkbox("Show feature importance", value=True)
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This app predicts customer churn using a machine learning model trained on historical customer data.")
    st.markdown("Adjust the inputs below and click 'Predict' to see results.")

# Input form with tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["Customer Info", "Service Info", "Contract Info", "Billing Info"])

with tab1:
    st.subheader("Customer Demographics", divider="blue")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='input-explanation'>Select the customer's gender:</div>", unsafe_allow_html=True)
        Gender = st.selectbox("Gender", options=["Male", "Female"], label_visibility="collapsed")
        
        st.markdown("<div class='input-explanation'>Is the customer a senior citizen (65+ years old)?</div>", unsafe_allow_html=True)
        SeniorCitizen = st.selectbox("Senior Citizen", options=[0, 1], 
                                   format_func=lambda x: "Yes" if x == 1 else "No",
                                   label_visibility="collapsed")
    with col2:
        st.markdown("<div class='input-explanation'>Does the customer have a partner (spouse or domestic partner)?</div>", unsafe_allow_html=True)
        Partner = st.selectbox("Has Partner", options=["Yes", "No"], label_visibility="collapsed")
        
        st.markdown("<div class='input-explanation'>Does the customer have dependents (children or other dependents)?</div>", unsafe_allow_html=True)
        Dependents = st.selectbox("Has Dependents", options=["Yes", "No"], label_visibility="collapsed")
    
    st.markdown("<div class='input-explanation'>How many months has the customer been with the company?</div>", unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12,
                      label_visibility="collapsed")

with tab2:
    st.subheader("Service Details", divider="blue")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='input-explanation'>Does the customer have phone service?</div>", unsafe_allow_html=True)
        PhoneService = st.selectbox("Phone Service", options=["Yes", "No"], label_visibility="collapsed")
        
        st.markdown("<div class='input-explanation'>Does the customer have multiple phone lines?</div>", unsafe_allow_html=True)
        MultipleLines = st.selectbox("Multiple Lines", options=["Yes", "No", "No phone service"], 
                                   label_visibility="collapsed")
        
        st.markdown("<div class='input-explanation'>What type of internet service does the customer have?</div>", unsafe_allow_html=True)
        InternetService = st.selectbox("Internet Service", options=['DSL', 'Fiber optic', 'No'], 
                                      label_visibility="collapsed")
        
        st.markdown("<div class='input-explanation'>Does the customer have online security service?</div>", unsafe_allow_html=True)
        OnlineSecurity = st.selectbox("Online Security", options=['No', 'Yes', 'No internet service'], 
                                     label_visibility="collapsed")
    with col2:
        st.markdown("<div class='input-explanation'>Does the customer have online backup service?</div>", unsafe_allow_html=True)
        OnlineBackup = st.selectbox("Online Backup", options=['No', 'Yes', 'No internet service'], 
                                   label_visibility="collapsed")
        
        st.markdown("<div class='input-explanation'>Does the customer have device protection service?</div>", unsafe_allow_html=True)
        DeviceProtection = st.selectbox("Device Protection", options=['No', 'Yes', 'No internet service'], 
                                       label_visibility="collapsed")
        
        st.markdown("<div class='input-explanation'>Does the customer have tech support service?</div>", unsafe_allow_html=True)
        TechSupport = st.selectbox("Tech Support", options=['No', 'Yes', 'No internet service'], 
                                  label_visibility="collapsed")

    st.subheader("Streaming Services", divider="blue")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='input-explanation'>Does the customer stream TV content?</div>", unsafe_allow_html=True)
        StreamingTV = st.selectbox("Streaming TV", options=['No', 'Yes', 'No internet service'], 
                                  label_visibility="collapsed")
    with col2:
        st.markdown("<div class='input-explanation'>Does the customer stream movies?</div>", unsafe_allow_html=True)
        StreamingMovies = st.selectbox("Streaming Movies", options=['No', 'Yes', 'No internet service'], 
                                      label_visibility="collapsed")

with tab3:
    st.subheader("Contract Details", divider="blue")
    st.markdown("<div class='input-explanation'>What type of contract does the customer have?</div>", unsafe_allow_html=True)
    Contract = st.selectbox("Contract Type", options=['Month-to-month', 'One year', 'Two year'],
                          label_visibility="collapsed")
    
    st.markdown("<div class='input-explanation'>Does the customer use paperless billing?</div>", unsafe_allow_html=True)
    PaperlessBilling = st.selectbox("Paperless Billing", options=["Yes", "No"],
                                  label_visibility="collapsed")
    
    st.markdown("<div class='input-explanation'>What payment method does the customer use?</div>", unsafe_allow_html=True)
    PaymentMethod = st.selectbox("Payment Method", 
                               options=['Electronic check', 'Mailed check', 
                                       'Bank transfer (automatic)', 'Credit card (automatic)'],
                               label_visibility="collapsed")

with tab4:
    st.subheader("Billing Information", divider="blue")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='input-explanation'>What is the customer's monthly charge amount?</div>", unsafe_allow_html=True)
        MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=18.25, max_value=118.75, value=50.0,
                                       label_visibility="collapsed")
    with col2:
        st.markdown("<div class='input-explanation'>What is the total amount charged to the customer?</div>", unsafe_allow_html=True)
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0,
                                     label_visibility="collapsed")

# Prediction button centered
st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("Predict Churn Probability", use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

if predict_button:
    # Prepare input data
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
    
    input_data_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for column, encoder in encoders.items():
        input_data_df[column] = encoder.transform(input_data_df[column])
    
    # Make prediction
    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)
    
    # Display results with dark theme colors
    churn_prob = pred_prob[0][1] * 100  # Probability of churn
    
    if prediction[0] == 1:
        st.markdown(f"""
        <div class="prediction-box churn">
            <h2 style="color: var(--churn-color);">🚨 High Churn Risk Detected</h2>
            <p>This customer has a <strong style="color: var(--churn-color);">{churn_prob:.1f}%</strong> probability of churning.</p>
            <p>Recommended action: Immediate retention efforts needed.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box no-churn">
            <h2 style="color: var(--no-churn-color);">✅ Low Churn Risk</h2>
            <p>This customer has only a <strong style="color: var(--no-churn-color);">{churn_prob:.1f}%</strong> probability of churning.</p>
            <p>Recommended action: Standard engagement recommended.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualize probability with dark theme colors
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(['Churn Probability'], [churn_prob], 
            color='#e74c3c' if prediction[0] == 1 else '#2ecc71')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)', color='white')
    ax.set_title('Churn Probability Visualization', color='white')
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    st.pyplot(fig)
    
    # Show feature importance if enabled
    if show_details and hasattr(loaded_model, 'feature_importances_'):
        st.subheader("Feature Importance", divider="gray")
        
        # Get feature importances
        importances = loaded_model.feature_importances_
        indices = importances.argsort()[::-1]
        
        # Create dataframe for display
        feature_importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        # Plot top 10 features with dark theme
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), 
                    palette='viridis', ax=ax)
        ax.set_title('Top 10 Features Influencing Churn Prediction', color='white')
        ax.set_facecolor('#1a1a1a')
        fig.patch.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('white')
        st.pyplot(fig)
        
        # Show full feature importance table
        with st.expander("View Complete Feature Importance"):
            st.dataframe(feature_importance_df.style.background_gradient(cmap='Blues'))
    
    # Show raw prediction data in expandable section
    with st.expander("View Raw Prediction Data"):
        st.write("Input Features:", input_data_df)
        st.write("Prediction:", "Churn" if prediction[0] == 1 else "No Churn")
        st.write("Prediction Probabilities:", pred_prob)

# Add some sample customer profiles for quick testing
with st.expander("💡 Try Sample Customer Profiles"):
    sample1, sample2 = st.columns(2)
    
    with sample1:
        if st.button("High Risk Profile"):
            st.session_state.Gender = "Male"
            st.session_state.SeniorCitizen = 0
            st.session_state.Partner = "No"
            st.session_state.Dependents = "No"
            st.session_state.tenure = 1
            st.session_state.PhoneService = "Yes"
            st.session_state.MultipleLines = "No"
            st.session_state.InternetService = "Fiber optic"
            st.session_state.OnlineSecurity = "No"
            st.session_state.OnlineBackup = "No"
            st.session_state.DeviceProtection = "No"
            st.session_state.TechSupport = "No"
            st.session_state.StreamingTV = "Yes"
            st.session_state.StreamingMovies = "Yes"
            st.session_state.Contract = "Month-to-month"
            st.session_state.PaperlessBilling = "Yes"
            st.session_state.PaymentMethod = "Electronic check"
            st.session_state.MonthlyCharges = 100.0
            st.session_state.TotalCharges = 100.0
    
    with sample2:
        if st.button("Low Risk Profile"):
            st.session_state.Gender = "Female"
            st.session_state.SeniorCitizen = 0
            st.session_state.Partner = "Yes"
            st.session_state.Dependents = "Yes"
            st.session_state.tenure = 60
            st.session_state.PhoneService = "Yes"
            st.session_state.MultipleLines = "No"
            st.session_state.InternetService = "DSL"
            st.session_state.OnlineSecurity = "Yes"
            st.session_state.OnlineBackup = "Yes"
            st.session_state.DeviceProtection = "Yes"
            st.session_state.TechSupport = "Yes"
            st.session_state.StreamingTV = "No"
            st.session_state.StreamingMovies = "No"
            st.session_state.Contract = "Two year"
            st.session_state.PaperlessBilling = "No"
            st.session_state.PaymentMethod = "Bank transfer (automatic)"
            st.session_state.MonthlyCharges = 50.0
            st.session_state.TotalCharges = 3000.0

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9em; padding: 20px;">
        <p>Customer Churn Prediction App • Powered by Machine Learning</p>
        <p>For business inquiries or support, please contact our team</p>
    </div>
""", unsafe_allow_html=True)
