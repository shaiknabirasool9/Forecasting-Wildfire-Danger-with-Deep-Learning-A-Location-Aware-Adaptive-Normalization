# Forecasting Wildfire Danger with Deep Learning

## Overview

A deep learning based wildfire danger prediction system using location-aware adaptive normalization techniques.

The model analyzes environmental factors to forecast wildfire risk levels.

## Technologies

- Python
- Deep Learning
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Data Visualization

## Features

- Location-based wildfire risk prediction
- Environmental data preprocessing
- Deep learning model training
- Risk level forecasting dashboard

## Workflow

Data Collection
↓
Data Cleaning
↓
Adaptive Normalization
↓
Deep Learning Model
↓
Risk Prediction
↓
Visualization

## Run Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python main.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

If the model file does not exist, run `python main.py` first to generate `saved_model/custom_model.h5`.

## Streamlit Community Cloud Deployment

This project can be deployed on Streamlit Community Cloud using the following settings:

- Repository: `shaiknabirasool9/Forecasting-Wildfire-Danger-with-Deep-Learning-A-Location-Aware-Adaptive-Normalization`
- Branch: `main`
- App file path: `app.py`
- Recommended subdomain: `wildfire-danger-prediction`

> Note: `saved_model/custom_model.h5` is not included in the repo by default, so the app will show an error if the model file is missing. To deploy successfully, either add the trained model file to the repo or modify `app.py` to download it from a hosted location.
