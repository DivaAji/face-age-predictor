import cv2
import numpy as np
import joblib
from sklearn.svm import SVC

# Fungsi untuk ekstraksi fitur LBP
def extract_lbp_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    lbp = cv2.calcHist([image], [0], None, [256], [0, 256])
    return lbp.flatten()

# Memuat model yang telah disimpan
model_path = 'app/models/svm_model.pkl'
model = joblib.load(model_path)

# Fungsi untuk memprediksi usia dari gambar
def predict_age(image_path):
    # Ekstraksi fitur dari gambar
    features = extract_lbp_features(image_path)
    
    # Melakukan prediksi dengan model
    age_prediction = model.predict([features])
    return age_prediction[0]

# Contoh penggunaan prediksi
image_path = 'test.jpg'  # Ganti dengan path gambar yang ingin diprediksi
predicted_age = predict_age(image_path)
print(f"Usia yang diprediksi: {predicted_age}")
