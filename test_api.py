# test_api.py - Test script for the Iris Prediction API
# Run this after starting app.py in a separate terminal.

import requests
import json

BASE_URL = "http://127.0.0.1:5000"


def print_response(label, response):
    """Helper function to display API responses clearly."""
    print(f"{'='*50}")
    print(f"TEST: {label}")
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))


# --- Test 1: Health Check ---
# I verify the server is running and the model loaded correctly.
response = requests.get(f"{BASE_URL}/health")
print_response("Health Check", response)


# --- Test 2: Iris-setosa prediction ---
# These measurements (small sepal, very small petal) are characteristic of Iris-setosa.
# I expect the model to predict Iris-setosa with very high confidence.
setosa_input = {
    "SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.2
}
response = requests.post(f"{BASE_URL}/predict", json=setosa_input)
print_response("Predict Iris-setosa", response)


# --- Test 3: Iris-versicolor prediction ---
# These medium-sized measurements are typical of Iris-versicolor.
versicolor_input = {
    "SepalLengthCm": 6.0,
    "SepalWidthCm": 2.9,
    "PetalLengthCm": 4.5,
    "PetalWidthCm": 1.5
}
response = requests.post(f"{BASE_URL}/predict", json=versicolor_input)
print_response("Predict Iris-versicolor", response)


# --- Test 4: Iris-virginica prediction ---
# These large measurements are typical of Iris-virginica.
virginica_input = {
    "SepalLengthCm": 6.9,
    "SepalWidthCm": 3.1,
    "PetalLengthCm": 5.4,
    "PetalWidthCm": 2.1
}
response = requests.post(f"{BASE_URL}/predict", json=virginica_input)
print_response("Predict Iris-virginica", response)


# --- Test 5: Missing field (error handling test) ---
# I intentionally leave out PetalWidthCm to test that my API returns
# a proper 400 error with a helpful message instead of crashing.
incomplete_input = {
    "SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4
    # PetalWidthCm intentionally omitted
}
response = requests.post(f"{BASE_URL}/predict", json=incomplete_input)
print_response("Missing Field Error (Expected 400)", response)


print("" + "="*50)
print("All tests complete!")