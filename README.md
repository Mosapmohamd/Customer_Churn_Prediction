# Customer Churn Prediction Dashboard

## Overview  
A Streamlit web application that predicts customer churn probability using machine learning. The application features a modern dark theme interface with comprehensive customer data collection and detailed risk analysis.

## Key Features  

### 🖥️ Enhanced User Interface  
- **Dark theme** with professional color scheme  
- **Tabbed organization** of input fields:  
  - Customer Info  
  - Service Info  
  - Contract Info  
  - Billing Info  
- **Clear explanatory text** above each input field  
- **Responsive design** that works on different screen sizes  

### 🔮 Prediction Features  
- **Churn probability** calculation with percentage score  
- **Risk classification** (High/Low) with color-coded alerts:  
  - 🟢 Green = Low risk  
  - 🔴 Red = High risk  
- **Visual probability indicator** (horizontal bar chart)  
- **Feature importance** visualization (when enabled)  
- **Raw prediction data** in expandable section  

### ✨ User Experience Improvements  
- **Sample customer profiles** for quick testing:  
  - High Risk Demo  
  - Low Risk Demo  
- **One-click prediction** with prominent action button  
- **Detailed recommendations** based on risk level  
- **Modern styling** with custom CSS  

## 🛠️ Technical Details  

### Requirements  
```
Python 3.7+
Streamlit
Pandas
Scikit-learn
Matplotlib
Seaborn
Pickle
```

### Installation  
1. Clone the repository  
2. Install requirements: `pip install -r requirements.txt`  
3. Run the app: `streamlit run app.py`  

### Model Information  
- **Algorithm**: Random Forest Classifier  
- **Input Features**:  
  - 19 total features (categorical + numerical)  
- **Output**: Churn probability (0-100%)  

## 📋 Usage Instructions  
1. Fill in customer information across the four tabs  
2. Click "Predict Churn Probability" button  
3. View the risk assessment and probability score  
4. Explore feature importance and raw data if needed  

## 🎨 Sample Outputs  

**Low Risk Example**  
```markdown
✅ Low Churn Risk  
This customer has only a 28.0% probability of churning.  
Recommended action: Standard engagement recommended.  
```

**High Risk Example**  
```markdown
🚨 High Churn Risk Detected  
This customer has a 82.5% probability of churning.  
Recommended action: Immediate retention efforts needed.  
```

## ⚙️ Customization  
The app can be easily modified by:  
- Adjusting the CSS variables in the style section  
- Changing the model artifacts  
- Modifying the input fields as needed  

