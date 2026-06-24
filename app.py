# app.py - Iris Species Prediction API
# I import Flask to build the web server, joblib to load my saved model,
# and pandas to construct a properly named feature DataFrame that matches
# exactly how the model was trained.

from flask import Flask, request, jsonify
import joblib
import pandas as pd

# I create the Flask application object.
# __name__ tells Flask where to find resources relative to this file.
app = Flask(__name__)

# I load the saved Logistic Regression model into memory when the server starts.
# This means the model is ready to make predictions the moment the first request arrives —
# I don't have to reload it for every single request, which keeps the API fast.
model = joblib.load('iris_model.pkl')

# I define the feature names as a constant so they match exactly what the model
# was trained on. This is the single source of truth — if the order or names
# ever need to change, I only update this one list.
FEATURE_NAMES = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

print("Iris prediction model loaded successfully.")


# --- Endpoint 1: Welcome ---
# This is the root endpoint that greets anyone who visits the API base URL.
# I use it to provide a quick overview of what this API does and which
# endpoints are available, so a new user or tester knows exactly where to go.
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Welcome to the Iris Species Prediction API.",
        "description": "Send flower measurements to the /predict endpoint to identify the Iris species.",
        "endpoints": {
            "welcome": "GET /",
            "health":  "GET /health",
            "predict": "POST /predict"
        }
    })


# --- Endpoint 2: Health Check ---
# This endpoint tells anyone who checks whether the API server is alive.
# It is useful for monitoring tools and for verifying the server started correctly.
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "model": "Logistic Regression Iris Species Classifier",
        "message": "API is healthy and ready to receive predictions."
    })


# --- Endpoint 3: Prediction ---
# This is the main endpoint. A client sends four flower measurements as JSON,
# and I return the predicted species name along with confidence probabilities.
@app.route('/predict', methods=['POST'])
def predict():
    # I retrieve the JSON body sent by the client.
    data = request.get_json()

    # I validate the incoming data to make sure all 4 required features are present.
    # If any feature is missing, I return a 400 error with a helpful message
    # so the user knows exactly what they need to fix.
    for feature in FEATURE_NAMES:
        if feature not in data:
            return jsonify({
                "error": f"Missing required field: {feature}",
                "required_fields": FEATURE_NAMES
            }), 400

    # I also validate that every value is numeric before passing it to the model.
    # This lets me return a clean 400 error instead of an unexpected server crash
    # if the client accidentally sends a string or null in place of a number.
    for feature in FEATURE_NAMES:
        try:
            float(data[feature])
        except (TypeError, ValueError):
            return jsonify({
                "error": f"Invalid value for '{feature}': must be a number.",
                "received": data[feature]
            }), 400

    # I construct a pandas DataFrame with explicit column names that match
    # the feature names the model was trained on.
    # This preserves the named structure scikit-learn expects and keeps
    # the prediction pipeline consistent with how the model was built.
    features_df = pd.DataFrame([{
        feature: float(data[feature]) for feature in FEATURE_NAMES
    }])

    # I use the loaded model to make the prediction.
    prediction = model.predict(features_df)[0]

    # I also get the probability for each class so I can show confidence scores.
    # predict_proba returns the probability that the input belongs to each species.
    probabilities = model.predict_proba(features_df)[0]
    class_names = model.classes_

    # I build a confidence dictionary that maps each species to its probability.
    confidence_scores = {
        cls: round(float(prob) * 100, 2)
        for cls, prob in zip(class_names, probabilities)
    }

    # I return the prediction as a clean JSON response.
    return jsonify({
        "predicted_species": prediction,
        "confidence_scores_percent": confidence_scores,
        "input_received": data
    })


# I run the Flask development server on port 5000.
# debug=True makes it easier to spot errors during testing — it shows detailed error messages
# and automatically restarts the server when I edit the code.
# NOTE: For production deployment, debug should be set to False.
if __name__ == '__main__':
    app.run(debug=True, port=5000)