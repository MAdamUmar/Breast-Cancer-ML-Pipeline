from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model pipeline
model = joblib.load("model/pipeline.pkl")

# List of feature names used in the model (25 total)
feature_names = [
    'perimeter_worst', 'area_worst', 'radius_worst', 'concave points_mean',
    'concave points_worst', 'perimeter_mean', 'concavity_mean',
    'radius_mean', 'area_mean', 'area_se', 'concavity_worst',
    'perimeter_se', 'radius_se', 'compactness_worst', 'compactness_mean',
    'concave points_se', 'texture_worst', 'concavity_se',
    'smoothness_worst', 'symmetry_worst', 'texture_mean', 'smoothness_mean',
    'compactness_se', 'fractal_dimension_worst', 'symmetry_mean'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read all 30 features from form
        features = [float(request.form[name]) for name in feature_names]
        features = np.array(features).reshape(1, -1)

        # Predict and return result
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][int(prediction)]

        result = "Malignant" if prediction == 1 else "Benign"
        return render_template('index.html',
                               prediction_text=f'Prediction: {result} (Confidence: {proba:.2f})')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

