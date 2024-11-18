from flask import Flask, request, jsonify
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
    image = cv2.resize(image, (128, 128))  # Resize to match model input size
    lbp = cv2.calcHist([image], [0], None, [256], [0, 256])  # Calculate LBP histogram
    lbp_features = lbp.flatten()
    return lbp_features  # Select the first 50 features

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image_path = 'temp_image.jpg'
    image_file.save(image_path)

    # Ekstraksi fitur LBP
    features = extract_lbp_features(image_path)
    
    # Prediksi usia menggunakan model SVM
    predicted_age = model.predict([features])[0]  # Mengambil prediksi usia
    
    # Pastikan prediksi usia dalam format int
    predicted_age = int(predicted_age)  # Konversi ke tipe data int jika perlu

    # Kembalikan prediksi usia dalam format JSON
    return jsonify({'predicted_age': predicted_age})


if __name__ == '__main__':
    app.run(debug=True)
