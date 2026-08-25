"""
app.py
Flask ML application for Iris flower prediction.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# --------------------------------------------------
# Load ML Model
# --------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_names = bundle["feature_names"]
class_names = bundle["class_names"]


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# --------------------------------------------------
# Prediction from HTML Form
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from HTML form
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Create input array
        features = np.array([
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]).reshape(1, -1)

        # Make prediction
        pred_idx = int(model.predict(features)[0])

        # Get probability
        probabilities = model.predict_proba(features)[0]

        prediction = class_names[pred_idx]

        confidence = round(
            float(probabilities[pred_idx]) * 100,
            2
        )

        # Send result back to HTML
        return render_template(
            "index.html",
            prediction=prediction,
            confidence=confidence
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


# --------------------------------------------------
# JSON API Endpoint
# --------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():

    try:

        payload = request.get_json(force=True)

        features = payload.get("features")

        if features is None:
            return jsonify({
                "error": "Missing 'features' field in JSON body."
            }), 400

        if len(features) != len(feature_names):
            return jsonify({
                "error": (
                    f"Expected {len(feature_names)} features "
                    f"{feature_names}, got {len(features)}."
                )
            }), 400

        X = np.array(
            features,
            dtype=float
        ).reshape(1, -1)

        pred_idx = int(
            model.predict(X)[0]
        )

        proba = model.predict_proba(X)[0].tolist()

        return jsonify({

            "prediction": class_names[pred_idx],

            "prediction_index": pred_idx,

            "probabilities": {
                class_names[i]: round(
                    p,
                    4
                )
                for i, p in enumerate(proba)
            }

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
