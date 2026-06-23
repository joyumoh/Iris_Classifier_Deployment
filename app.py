# app.py — Iris Species Prediction API
# I already import Flask to build the web server, joblib to load my saved model,
# and numpy to format the input before passing it to the model.


# I create the Flask application object.
# __name__ tells Flask where to find resources relative to this file.
app = Flask(__name__)

# I load the saved Logistic Regression model into memory when the server starts.
# This means the model is ready to make predictions the moment the first request arrives —
# I don't have to reload it for every single request, which keeps the API fast.
model = joblib.load('iris_model.pkl')

print("Iris prediction model loaded successfully.")


# --- Endpoint 1: Health Check ---
# This endpoint tells anyone who checks whether the API server is alive.
# It is useful for monitoring tools and for verifying the server started correctly.
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "model": "Logistic Regression — Iris Species Classifier",
        "message": "API is healthy and ready to receive predictions."
    })


# --- Endpoint 2: Prediction ---
# This is the main endpoint. A client sends four flower measurements as JSON,
# and I return the predicted species name along with confidence probabilities.
@app.route('/predict', methods=['POST'])
def predict():
    # I retrieve the JSON body sent by the client.
    data = request.get_json()

    # I validate the incoming data to make sure all 4 required features are present.
    # If any feature is missing, I return a 400 error with a helpful message
    # so the user knows exactly what they need to fix.
    required_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

    for feature in required_features:
        if feature not in data:
            return jsonify({
                "error": f"Missing required field: {feature}",
                "required_fields": required_features
            }), 400

    # I extract the values and reshape them into a 2D array.
    # scikit-learn's predict() expects a 2D array (even for a single sample),
    # so I use [[ ]] to wrap my 4 values as a single-row matrix.
    features = np.array([[
        data['SepalLengthCm'],
        data['SepalWidthCm'],
        data['PetalLengthCm'],
        data['PetalWidthCm']
    ]])

    # I use the loaded model to make the prediction.
    prediction = model.predict(features)[0]

    # I also get the probability for each class so I can show confidence scores.
    # predict_proba returns the probability that the input belongs to each species.
    probabilities = model.predict_proba(features)[0]
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