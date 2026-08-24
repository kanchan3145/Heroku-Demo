"""
app.py
A minimal Flask API that serves predictions from model.joblib.

Endpoints:
  GET  /              -> health check / basic info
  POST /predict        -> JSON in, JSON prediction out
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_names = bundle["feature_names"]
class_names = bundle["class_names"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Iris classifier API is running.",
        "expected_features": feature_names,
        "classes": class_names,
        "usage": {
            "endpoint": "/predict",
            "method": "POST",
            "body_example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        features = payload.get("features")

        if features is None:
            return jsonify({"error": "Missing 'features' field in JSON body."}), 400
        if len(features) != len(feature_names):
            return jsonify({
                "error": f"Expected {len(feature_names)} features {feature_names}, "
                         f"got {len(features)}."
            }), 400

        X = np.array(features, dtype=float).reshape(1, -1)
        pred_idx = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0].tolist()

        return jsonify({
            "prediction": class_names[pred_idx],
            "prediction_index": pred_idx,
            "probabilities": {
                class_names[i]: round(p, 4) for i, p in enumerate(proba)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
