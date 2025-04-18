import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Title
st.title("💰 Estimativa de Aluguel no Distrito Federal")

# Description
st.write("Preencha as informações do imóvel e veja a estimativa de aluguel com base no modelo XGBoost.")

# Load the pre-trained model and scaler
model = joblib.load("streamlit_app/modelo_xgboost.pkl")
scaler = joblib.load("streamlit_app/scaler.pkl")  # Load the scaler

# Read the dataset and identify property types and neighborhoods
df = pd.read_csv("streamlit_app/brasilia-rent-price-prediction-df.csv", header=0)
tipos = sorted(df['tipo'].dropna().unique())
bairros = sorted(df['bairro'].dropna().unique())

# User inputs
tipo = st.selectbox("Tipo de imóvel", tipos)
bairro = st.selectbox("Bairro", bairros)
area = st.number_input("Área (m²)", min_value=10.0, max_value=300.0, value=70.0, step=1.0)
quartos = st.slider("Número de quartos", 1, 10, 2)

# Function to preprocess input data
def preprocess_input(tipo, bairro, area, quartos):
    # Initialize dictionary with input values
    input_dict = {
        'area': area,
        'quartos': quartos,
    }

    # One-hot encoding for the input values (make sure all possible columns exist)
    for t in tipos:
        input_dict[f"tipo_{t}"] = 1 if t == tipo else 0
    for b in bairros:
        input_dict[f"bairro_{b}"] = 1 if b == bairro else 0

    # Create dataframe from input dictionary
    input_df = pd.DataFrame([input_dict])

    # Align the columns to match the training data
    # Get the column names from the trained model
    trained_columns = model.get_booster().feature_names
    missing_columns = set(trained_columns) - set(input_df.columns)

    # Add missing columns with value 0
    for col in missing_columns:
        input_df[col] = 0

    # Ensure columns are in the same order as in the trained model
    input_df = input_df[trained_columns]

    # Normalize the numeric features using the pre-fitted scaler
    numeric_cols = ['area', 'quartos']  # Columns to scale
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])  # Apply scaling

    return input_df

# Prediction button
if st.button("🔍 Estimar Aluguel"):
    input_data = preprocess_input(tipo, bairro, area, quartos)
    preco_estimado = model.predict(input_data)[0]
    st.success(f"🏠 Estimativa de aluguel: R$ {preco_estimado:,.2f}")
