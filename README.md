# Customer Churn Prediction  

## Overview  
A Streamlit web application that predicts customer churn probability using machine learning. The application features a modern dark theme interface with comprehensive customer data collection and detailed risk analysis.  

**📌 Live Demo:** [https://customer1churn1prediction.streamlit.app/](https://customer1churn1prediction.streamlit.app/)  

## ✨ Key Features  

### 🖥️ Enhanced User Interface  
- **Dark theme** with professional color scheme  
- **Tabbed organization** of input fields:  
  - 📝 Customer Info  
  - ⚙️ Service Info  
  - 📄 Contract Info  
  - 💰 Billing Info  
- **Clear explanatory text** above each input field  
- **Responsive design** that works on different screen sizes  

### 🔮 Prediction Features  
- **📊 Churn probability** calculation with percentage score  
- **⚠️ Risk classification** (High/Low) with color-coded alerts:  
  - 🟢 Green = Low risk (<50%)  
  - 🔴 Red = High risk (≥50%)  
- **📈 Visual probability indicator** (horizontal bar chart)  
- **🔍 Feature importance** visualization (when enabled)  
- **📂 Raw prediction data** in expandable section  

## 🛠️ Technical Implementation  

### ⚙️ Requirements  
```python
Python 3.7+
Streamlit >=1.0
Pandas >=1.0
Scikit-learn >=0.24
Matplotlib >=3.0
Seaborn >=0.11
Pickle
