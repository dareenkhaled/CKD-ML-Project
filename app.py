import streamlit as st

st.title("Chronic Kidney Disease Prediction Using Machine Learning")

st.write("""
This web app is part of a machine learning project for predicting chronic kidney disease risk
using clinical and laboratory patient data.
""")

st.header("Project Workflow")
st.write("""
- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering and selection
- Model training using different algorithms
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Deployment using Streamlit
""")

st.header("Best Model")
st.write("""
The Random Forest model was selected as the final model because it achieved strong performance
compared with the other tested algorithms.
""")

st.header("Conclusion")
st.write("""
The project shows how machine learning can support medical risk prediction using patient data.
The model can help classify patients based on kidney disease risk.
""")
