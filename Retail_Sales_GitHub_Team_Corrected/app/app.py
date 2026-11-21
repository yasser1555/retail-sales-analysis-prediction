"""Streamlit prediction UI, adapted from the supplied root-level app.py.

Form inputs, prediction logic, and demand messages are retained.
Only loading paths and missing-artifact feedback were made portable.
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILE = MODEL_DIR / "sales_model.pkl"
FEATURES_FILE = MODEL_DIR / "model_features.pkl"

st.title("Retail Sales Prediction Tool")
st.write(
    "This tool predicts weekly sales based on store characteristics, department, "
    "holiday effects, and external economic factors. Predictions are based on a "
    "trained Random Forest model."
)

if not MODEL_FILE.exists() or not FEATURES_FILE.exists():
    st.error(
        "Model artifacts are not included in the public repository. "
        "Follow the README to prepare the data and train the model, "
        "or place your trusted original sales_model.pkl and "
        "model_features.pkl files in the models/ directory."
    )
    st.stop()

@st.cache_resource
def load_artifacts():
    # joblib/pickle should only be loaded from sources you trust.
    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURES_FILE)
    return model, features

try:
    model, features = load_artifacts()
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()

store = st.number_input("Store ID", min_value=1, max_value=45, value=1)
dept = st.number_input("Department ID", min_value=1, max_value=99, value=1)
is_holiday = st.selectbox("Is Holiday Week?", [False, True])

size = st.number_input("Store Size", min_value=1, value=150000)
store_type = st.selectbox("Store Type", ["A", "B", "C"])

temperature = st.number_input("Temperature", value=50.0)
fuel_price = st.number_input("Fuel Price", value=3.0)
cpi = st.number_input("CPI", value=220.0)
unemployment = st.number_input("Unemployment Rate", value=7.0)

type_b = 1 if store_type == "B" else 0
type_c = 1 if store_type == "C" else 0

input_data = pd.DataFrame([{
    "Store": store,
    "Dept": dept,
    "IsHoliday": is_holiday,
    "Size": size,
    "Type_B": type_b,
    "Type_C": type_c,
    "Temperature": temperature,
    "Fuel_Price": fuel_price,
    "CPI": cpi,
    "Unemployment": unemployment,
}])
input_data = input_data[features]

if st.button("Predict Weekly Sales"):
    if size <= 0:
        st.error("Store size must be greater than 0.")
    else:
        prediction = model.predict(input_data)[0]
        st.subheader(f"Predicted Weekly Sales: ${prediction:,.2f}")
        if prediction >= 30000:
            st.success("High expected demand. Consider increasing inventory and staffing.")
        elif prediction >= 10000:
            st.info("Moderate expected demand. Maintain normal inventory levels.")
        else:
            st.warning("Low expected demand. Avoid overstocking.")
        st.caption(
            "Note: Predictions are based on historical data patterns and may "
            "vary in real-world scenarios."
        )
