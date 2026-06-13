import os
from flask import Flask, request, jsonify
from model import predict_churn, predict_risk

app = Flask(__name__)


@app.route("/predict/churn", methods=["POST"])
def churn():
    """
    POST JSON with patient features.
    Returns churn_probability and will_churn flag.
    """
    data = request.get_json(force=True)
    try:
        result = predict_churn(data)
        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/predict/risk", methods=["POST"])
def risk():
    """
    POST JSON with consultation/symptom features.
    Returns risk_probability and high_risk flag.
    """
    data = request.get_json(force=True)
    try:
        result = predict_risk(data)
        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Telemedicine ML API"})


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, port=5000)
