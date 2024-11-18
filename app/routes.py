from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import joblib
from .utils import extract_lbp_features

# Inisialisasi blueprint
main = Blueprint('main', __name__)

# Muat model
model_path = 'app/models/svm_model.pkl'
model = joblib.load(model_path)

@main.route('/', methods=['GET'])
def home():
    return "Welcome to Face Age Predictor API"

@main.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Simpan file sementara
    image_path = f"temp/{file.filename}"
    file.save(image_path)
    
    # Ekstraksi fitur dan prediksi
    try:
        features = extract_lbp_features(image_path)
        predicted_age = model.predict([features])[0]
        return jsonify({'predicted_age': int(predicted_age)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
