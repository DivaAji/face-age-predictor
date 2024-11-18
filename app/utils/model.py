import joblib
from app.utils.preprocessing import extract_lbp_features

# Load model SVM
MODEL_PATH = 'app/models/svm_model.pkl'
model = joblib.load(MODEL_PATH)

def predict_age(image_path):
    features = extract_lbp_features(image_path)
    prediction = model.predict([features])
    return prediction[0]
