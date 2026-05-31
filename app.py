import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("Chronic Kidney Disease Prediction App")

st.write("This app predicts chronic kidney disease risk using a small machine learning model.")

df = pd.read_csv("kidney_disease_dataset.csv")

# choose target column
target_col = "Target" if "Target" in df.columns else df.columns[-1]

# choose simple input features
features = [
    "Age of the patient",
    "Blood pressure (mm/Hg)",
    "Specific gravity of urine",
    "Albumin in urine",
    "Sugar in urine"
]

features = [col for col in features if col in df.columns]

X = df[features].copy()
y = df[target_col].copy()

# encode categorical columns
encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

if y.dtype == "object":
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y.astype(str))
else:
    target_encoder = None

model = RandomForestClassifier(
    n_estimators=20,
    max_depth=5,
    random_state=42
)

model.fit(X, y)

st.header("Enter Patient Data")

user_input = {}

for col in features:
    if col in encoders:
        options = list(encoders[col].classes_)
        value = st.selectbox(col, options)
        user_input[col] = encoders[col].transform([value])[0]
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=mean_val)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]

    if target_encoder is not None:
        prediction_label = target_encoder.inverse_transform([prediction])[0]
    else:
        prediction_label = prediction

    st.success(f"Predicted CKD Risk Level: {prediction_label}")

st.markdown("---")
st.write("This is a simplified deployed version of the machine learning project.")
