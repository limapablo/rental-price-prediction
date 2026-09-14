# Rental Price Prediction — Brasília, Brazil

End-to-end machine learning project for estimating residential rental prices in Brazil's Federal District using **Python, XGBoost and Streamlit**.

The project covers the full workflow from raw-data exploration and preprocessing to model training, serialization and an interactive web application for inference.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-FF6600)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-not%20specified-lightgrey)](#license)

> **Portfolio focus:** applied machine learning, exploratory data analysis, feature engineering, model deployment and product-oriented communication.

## Live Demo

**Streamlit application:** [Open the rental price estimator](https://rental-price-prediction-apktbzfvgnenub8uc4bxrv.streamlit.app/)

The app receives a property profile — property type, neighborhood, area and number of bedrooms — and returns an estimated monthly rent.

## Business Problem

Rental prices vary substantially across neighborhoods and property characteristics. A data-driven estimator can support faster price benchmarking by translating a property profile into an expected rental value.

This project frames that problem as a **supervised regression task**:

- **Target:** monthly rental price (`preco`)
- **Numerical features:** property area and number of bedrooms
- **Categorical features:** property type and neighborhood
- **Output:** estimated monthly rent in BRL

The goal is not to replace a professional real-estate appraisal, but to demonstrate how a machine-learning workflow can transform historical listing data into a usable prediction product.

## Dataset

The original dataset is publicly available on Kaggle:

**[Preço do aluguel de imóveis no Distrito Federal](https://www.kaggle.com/datasets/matheusnbrega/preo-do-aluguel-de-imveis-no-distrito-federal)** — Matheus Nóbrega.

The raw dataset contains **2,871 records and 5 original columns**. During preprocessing, invalid/missing observations are removed and basic outlier rules are applied, leaving **2,610 observations** in the modeled dataset.

Core fields:

| Feature | Description |
|---|---|
| `preco` | Monthly rental price |
| `tipo` | Property type |
| `area` | Property area in square meters |
| `quartos` | Number of bedrooms |
| `bairro` | Neighborhood |

## Machine Learning Workflow

```text
Raw rental listings
        │
        ▼
Data cleaning & validation
        │
        ▼
Exploratory data analysis
        │
        ▼
Feature engineering
        │
        ├── numerical scaling
        └── categorical one-hot encoding
        │
        ▼
Train / test split
        │
        ▼
Regression model comparison
        │
        ▼
XGBoost final model
        │
        ▼
Serialized model + scaler
        │
        ▼
Streamlit inference app
```

The notebook explores multiple regression approaches and uses an **XGBoost regressor** as the deployed model. Preprocessing artifacts are persisted with `joblib` so the Streamlit application can reproduce the same transformation logic at inference time.

## Repository Structure

```text
rental-price-prediction/
├── data/
│   └── imoveis-df.csv
├── notebook/
│   └── exploratory_analysis_brasilia_rent_prices.ipynb
├── streamlit_app/
│   ├── app.py
│   ├── brasilia-rent-price-prediction-df.csv
│   ├── modelo_xgboost.pkl
│   ├── scaler.pkl
│   └── requirements.txt
├── MODEL_CARD.md
├── .gitignore
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python** — analysis and application logic
- **Pandas / NumPy** — data manipulation
- **Scikit-learn** — preprocessing, model evaluation and baseline regressors
- **XGBoost** — final regression model
- **Matplotlib / Seaborn / Plotly** — exploratory visualization
- **Joblib** — model artifact persistence
- **Streamlit** — interactive deployment
- **Jupyter Notebook** — experimentation and documented analysis

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/limapablo/rental-price-prediction.git
cd rental-price-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run streamlit_app/app.py
```

## Reproducibility Notes

The repository includes the trained model and preprocessing scaler used by the deployed application. The modeling notebook documents the exploratory analysis, cleaning rules, feature transformation and training workflow.

For a production-grade system, the next step would be to consolidate preprocessing and prediction into a single versioned `Pipeline`, add automated tests, pin dependency versions and introduce experiment tracking/model monitoring.

## Model Limitations

This is a portfolio project trained on a historical dataset of rental listings from the Federal District. Predictions can become inaccurate when:

- market conditions change materially;
- the requested property profile is poorly represented in the training data;
- listing quality or neighborhood definitions differ from the source dataset;
- important real-estate variables are unavailable, such as floor, building age, parking, furnishing, condominium amenities or exact geolocation.

See [`MODEL_CARD.md`](MODEL_CARD.md) for a concise description of intended use and limitations.

## Possible Next Steps

- Build a unified Scikit-learn preprocessing/model pipeline
- Add cross-validation and hyperparameter optimization
- Track model experiments and metrics
- Add prediction intervals or uncertainty estimates
- Introduce unit tests and CI with GitHub Actions
- Containerize the application with Docker
- Add geospatial features and richer property attributes
- Monitor model drift after deployment

## Author

**Pablo Henrique da Silva Lima**

- [LinkedIn](https://www.linkedin.com/in/limapablo/)
- [GitHub](https://github.com/limapablo)

## License

No explicit open-source license is currently defined for this repository. If you plan to reuse the code or data, review the source dataset terms and contact the author when appropriate.
