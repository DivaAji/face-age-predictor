from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model_path = 'app/models/svm_model.pkl'
model = joblib.load(model_path)

# Function to extract LBP features
def extract_lbp_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    lbp = cv2.calcHist([image], [0], None, [256], [0, 256])
    return lbp.flatten()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        # Save uploaded file temporarily
        file_path = 'temp_uploaded_image.jpg'
        file.save(file_path)

        # Extract features and predict age
        features = extract_lbp_features(file_path)
        predicted_age = model.predict([features])[0]

        # Return result in the UI
        return render_template('index.html', prediction=predicted_age)
    else:
        return "No file uploaded!", 400

if __name__ == '__main__':
    app.run(debug=True)
