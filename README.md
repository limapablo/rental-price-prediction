🏡 Rental Price Prediction App

This project is a complete data science workflow for predicting rental prices of properties in Brasília, Brazil 🇧🇷. It includes data cleaning, exploratory data analysis (EDA), model training using XGBoost, and deployment of a Streamlit app.

## 🔍 Project Overview

- Predict rental prices based on property features such as size, number of bedrooms, neighborhood, and property type.
- Clean and preprocess real data collected from online sources.
- Train a regression model using **XGBoost**.
- Build and deploy an interactive app using **Streamlit**.

## 📊 Dataset

The data was obtained from Kaggle:

📁 **[Preço do aluguel de imóveis no Distrito Federal](https://www.kaggle.com/datasets/matheusnbrega/preo-do-aluguel-de-imveis-no-distrito-federal)**  
Author: Matheus Nóbrega

## 🧠 Technologies Used

- Python
- Pandas, NumPy, Scikit-learn
- XGBoost
- Streamlit
- Jupyter Notebook

## 📁 Repository Structure

```
📁 streamlit_app/                
│   ├── app.py                  → Streamlit application
│   ├── modelo_xgboost.pkl      → Trained XGBoost model
│   ├── scaler.pkl              → Scaler used in preprocessing
│   ├── brasilia-rent-price-prediction-df.csv       → Final dataset used by the app
│   └── requirements.txt        → Python dependencies
├── data/
│   └── imoveis-df.csv          → Raw data
📁 notebooks/
│   └── exploratory_analysis_brasilia_rent_prices.ipynb  → Jupyter notebook for EDA and modeling
README.md
```

## 🚀 Try the App

👉 [Click here to open the app on Streamlit](https://rental-price-prediction-apktbzfvgnenub8uc4bxrv.streamlit.app/)

> ⚠️ Note: The app may take a few seconds to load because it needs to load the model, dataset, and libraries.

## 📊 Exploratory Analysis

All data preprocessing, feature engineering, and model evaluation steps are documented in the Jupyter notebook found in `notebooks/exploratory_analysis_brasilia_rent_prices.ipynb`.

## 🧠 Model Details

The final model used is **XGBoostRegressor**, trained on a one-hot encoded dataset that includes categorical features like `property type` and `neighborhood`, and numerical features like `area` and `bedrooms`.

The trained model (`modelo_xgboost.pkl`) and any necessary preprocessing objects (like `scaler.pkl`) are loaded inside the Streamlit app to perform real-time predictions.

## 📦 How to Run Locally

Clone the repo:

```bash
git clone https://github.com/limapablo/rental-price-prediction.git
cd rental-price-prediction/app
```

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## 📌 Author

**Pablo Henrique da Silva Lima**   
📫 [LinkedIn](https://www.linkedin.com/in/limapablo/)  
📧 lima.pablohs@gmail.com  

---

⭐️ If you like this project, give it a star!
