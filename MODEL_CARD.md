# Model Card — Brasília Rental Price Estimator

## Model Summary

This repository contains a regression model built to estimate monthly residential rental prices in Brazil's Federal District (Distrito Federal), using structured listing attributes.

- **Model family:** XGBoost Regressor
- **Task:** supervised regression
- **Target:** monthly rental price (`preco`)
- **Deployment:** Streamlit web application
- **Model artifacts:** `streamlit_app/modelo_xgboost.pkl` and `streamlit_app/scaler.pkl`

## Intended Use

The model is intended for:

- portfolio demonstration of an end-to-end machine-learning workflow;
- exploratory rental-price benchmarking;
- educational examples of tabular regression and lightweight deployment.

It should **not** be treated as a certified valuation model, legal appraisal, credit decision system, or substitute for a professional real-estate assessment.

## Input Features

The deployed application receives:

- property type;
- neighborhood;
- property area in square meters;
- number of bedrooms.

Categorical fields are converted to model-compatible indicator variables and numerical fields are transformed using the persisted scaler.

## Training Data

The source dataset is the public Kaggle dataset **“Preço do aluguel de imóveis no Distrito Federal”**, authored by Matheus Nóbrega.

The raw data contains 2,871 observations. After cleaning and filtering, the modeled dataset contains 2,610 observations.

## Preprocessing

The notebook applies the following main transformations:

1. converts the area field to numeric values;
2. removes observations with missing area or price;
3. derives price per square meter for exploratory analysis;
4. filters implausible/extreme values using explicit area and price thresholds;
5. one-hot encodes property type and neighborhood;
6. standardizes numerical features;
7. creates a train/test split with a fixed random seed.

## Evaluation

The project notebook contains the model-comparison and evaluation workflow used during development. The deployed model is an XGBoost regressor selected from the tested regression approaches.

For future iterations, the project should persist a machine-readable evaluation report containing metrics such as MAE, RMSE and R² alongside the model artifact.

## Limitations

Model quality is constrained by the available attributes and historical coverage of the source dataset. Important omitted variables may include:

- exact geolocation;
- parking spaces;
- furnishing;
- floor level;
- condominium amenities;
- building age and condition;
- listing date and market regime;
- proximity to transport, employment centers and services.

The model may also perform poorly for rare neighborhoods, unusual property profiles or values outside the training distribution.

## Risks

Potential risks include:

- overconfidence in point estimates;
- stale estimates as the rental market changes;
- bias caused by uneven geographic or property-type representation;
- incorrect inference when user inputs differ materially from training data.

Predictions should therefore be interpreted as approximate estimates, not authoritative valuations.

## Recommended Improvements

- migrate preprocessing and model inference to a single Scikit-learn `Pipeline`;
- use cross-validation and systematic hyperparameter tuning;
- persist evaluation metrics and experiment metadata;
- add uncertainty estimates or prediction intervals;
- add automated tests for preprocessing and inference;
- introduce model/version tracking and drift monitoring;
- enrich the feature set with geospatial and temporal variables.
