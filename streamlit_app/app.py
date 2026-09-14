from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "modelo_xgboost.pkl"
SCALER_PATH = APP_DIR / "scaler.pkl"
DATA_PATH = APP_DIR / "brasilia-rent-price-prediction-df.csv"

st.set_page_config(
    page_title="Brasília Rental Price Estimator",
    page_icon="🏠",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    """Load serialized model artifacts once per Streamlit session."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_reference_data():
    """Load reference data used to populate categorical input options."""
    return pd.read_csv(DATA_PATH)


model, scaler = load_artifacts()
df = load_reference_data()

property_types = sorted(df["tipo"].dropna().unique())
neighborhoods = sorted(df["bairro"].dropna().unique())


def preprocess_input(property_type: str, neighborhood: str, area: float, bedrooms: int) -> pd.DataFrame:
    """Transform user input into the feature schema expected by the trained model."""
    input_values = {
        "area": area,
        "quartos": bedrooms,
    }

    for value in property_types:
        input_values[f"tipo_{value}"] = int(value == property_type)

    for value in neighborhoods:
        input_values[f"bairro_{value}"] = int(value == neighborhood)

    input_df = pd.DataFrame([input_values])
    trained_columns = model.get_booster().feature_names

    for column in set(trained_columns) - set(input_df.columns):
        input_df[column] = 0

    input_df = input_df[trained_columns]
    numeric_columns = ["area", "quartos"]
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

    return input_df


st.title("Brasília Rental Price Estimator")
st.caption("Machine-learning estimate powered by an XGBoost regression model.")

st.markdown(
    "Enter the main characteristics of a property in Brazil's Federal District "
    "to obtain an estimated monthly rental price."
)

with st.form("rental_estimator"):
    property_type = st.selectbox("Property type", property_types)
    neighborhood = st.selectbox("Neighborhood", neighborhoods)

    col_area, col_bedrooms = st.columns(2)
    with col_area:
        area = st.number_input(
            "Area (m²)",
            min_value=10.0,
            max_value=300.0,
            value=70.0,
            step=1.0,
        )

    with col_bedrooms:
        bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=2)

    submitted = st.form_submit_button("Estimate monthly rent", use_container_width=True)

if submitted:
    model_input = preprocess_input(property_type, neighborhood, area, bedrooms)
    estimated_price = float(model.predict(model_input)[0])

    st.metric(
        label="Estimated monthly rent",
        value=f"R$ {estimated_price:,.2f}",
    )

    st.caption(
        "This estimate is for educational and portfolio purposes and should not be "
        "treated as a professional real-estate appraisal."
    )

with st.expander("About this model"):
    st.markdown(
        "The model uses property type, neighborhood, area and number of bedrooms. "
        "It was trained on historical rental-listing data from Brazil's Federal District. "
        "Because market conditions and individual property characteristics vary, predictions "
        "should be interpreted as approximate benchmarks."
    )
