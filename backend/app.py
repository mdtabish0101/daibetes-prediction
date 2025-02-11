from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the trained model and scaler
with open("../model/diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return "Diabetes Prediction API is Running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debugging line

        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]

        print("Prediction:", prediction)  # Debugging line
        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        print("Error:", str(e))  # Debugging line
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
