# Iris Species Prediction API

## Project Title
Iris Species Classifier — Machine Learning Model Deployment with Flask

---

## Model Description
This project deploys a **Logistic Regression** classifier trained on the classic Iris dataset.
The model predicts the species of an Iris flower (Setosa, Versicolor, or Virginica)
based on four physical measurements. It was selected over a Random Forest Classifier
because it achieved higher cross-validation accuracy and fewer test-set misclassifications.

The model is served through a REST API built with Flask, making it accessible to any
application or developer who can send an HTTP POST request.

---

## Input Features

| Field          | Type  | Unit | Description              |
|----------------|-------|------|--------------------------|
| SepalLengthCm  | float | cm   | Length of the sepal      |
| SepalWidthCm   | float | cm   | Width of the sepal       |
| PetalLengthCm  | float | cm   | Length of the petal      |
| PetalWidthCm   | float | cm   | Width of the petal       |

All four fields are required. The API returns a 400 error if any field is missing.

---

## API Endpoints

### GET /health
Returns the current status of the API.

**Example Response:**
```json
{
  "status": "running",
  "model": "Logistic Regression — Iris Species Classifier",
  "message": "API is healthy and ready to receive predictions."
}
```

### POST /predict
Accepts flower measurements and returns a species prediction with confidence scores.

**Example Request:**
```json
{
  "SepalLengthCm": 5.1,
  "SepalWidthCm": 3.5,
  "PetalLengthCm": 1.4,
  "PetalWidthCm": 0.2
}
```

**Example Response:**
```json
{
  "predicted_species": "Iris-setosa",
  "confidence_scores_percent": {
    "Iris-setosa": 99.85,
    "Iris-versicolor": 0.14,
    "Iris-virginica": 0.01
  },
  "input_received": {
    "SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.2
  }
}
```

---

## How to Run the API

### Prerequisites
- Python 3.8 or higher
- pip

### Step 1: Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/iris-api.git
cd iris-api
```

### Step 2: Install required packages
```bash
pip install flask scikit-learn joblib numpy
```

### Step 3: Start the Flask server
```bash
python app.py
```
The server will start at: `http://127.0.0.1:5000`

### Step 4: Test the API
In a second terminal, run:
```bash
python test_api.py
```
Or use curl:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"SepalLengthCm": 5.1, "SepalWidthCm": 3.5, "PetalLengthCm": 1.4, "PetalWidthCm": 0.2}'
```

---

## Project Files

| File                              | Description                                |
|-----------------------------------|--------------------------------------------|
| `iris_week7_deployment.ipynb`     | Full deployment walkthrough notebook       |
| `app.py`                          | Flask API source code                      |
| `iris_model.pkl`                  | Saved Logistic Regression model            |
| `test_api.py`                     | API test script                            |
| `Iris.csv`                        | Original dataset                           |
| `README.md`                       | This documentation file                   |

---

## Assumptions and Limitations

- Input measurements must be provided in **centimetres** (cm)
- The model was trained on the UCI Iris dataset (150 samples, 3 balanced classes)
- The API is designed for local/development use; for production deployment, replace `debug=True` with `debug=False` and use a production WSGI server (e.g., Gunicorn)
- The model may perform less reliably on measurements significantly outside the training range (sepal length: 4.3–7.9 cm, petal length: 1.0–6.9 cm)

---

**Developed by:** Joy Harrison Umoh
**Internship:** AnalystLab Africa
**Week:** 7 — Model Deployment
