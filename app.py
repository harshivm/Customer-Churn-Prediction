from flask import Flask, request, jsonify
import pickle
import numpy as np


try:
    model = pickle.load(open("churn_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError as e:
    # Provide a clear error message when running the server without pickle files
    raise FileNotFoundError(
        "Model or scaler pickle not found. Run `analyse.py` to train and save the model (creates churn_model.pkl and scaler.pkl)." 
    ) from e

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "running", "message": "Churn prediction API. POST JSON to /predict"})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert input to array (preserve column order expected by the model)
    input_data = np.array([list(data.values())])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = int(model.predict(input_scaled)[0])
    probability = float(model.predict_proba(input_scaled)[0][1])

    result = {
        "churn_prediction": prediction,
        "churn_probability": probability
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)