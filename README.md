# Customer Churn Prediction

This project predicts whether a telecom customer is likely to churn using machine learning.

## Files

- `analyse.py` - trains the models, evaluates them, and saves `churn_model.pkl` and `scaler.pkl`
- `app.py` - Flask API for churn prediction
- `streamlit_a.py` - Streamlit app for interactive prediction
- `telco_dataset.csv` - customer churn dataset

## Requirements

Install the main Python packages used in the project:

powershell
python -m pip install pandas numpy scikit-learn matplotlib seaborn xgboost flask streamlit


## Train the model

Run the analysis script to train the model and create the pickle files:

```powershell
python analyse.py
```

## Run the Flask app

Start the Flask prediction API:

```powershell
python app.py
```

## Run the Streamlit app

Start the Streamlit interface:

```powershell
python -m streamlit run streamlit_a.py
```

## Notes

- `churn_model.pkl` and `scaler.pkl` must exist before running the apps.
- If you retrain the model, rerun `analyse.py` to update the saved files.
