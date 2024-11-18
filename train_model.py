import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm
import multiprocessing
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

# Terapkan PCA setelah ekstraksi fitur LBP
def reduce_dimensionality(X_train, X_test):
    pca = PCA(n_components=50)  # Mengurangi menjadi 50 komponen
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test

# Fungsi untuk ekstraksi fitur LBP
def extract_lbp_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))  # Resize gambar
    lbp = cv2.calcHist([image], [0], None, [256], [0, 256])  # Hitung histogram LBP
    return lbp.flatten()

# Fungsi untuk memproses satu gambar
def process_image(image_path):
    parts = os.path.basename(image_path).split('_')
    age = int(parts[0])  # Ambil usia dari nama file
    features = extract_lbp_features(image_path)  # Ekstraksi fitur
    return features, age

# Fungsi untuk memuat dataset dengan multiprocessing
def load_dataset(dataset_path):
    X = []
    y = []
    image_paths = [
        os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path) if filename.endswith('.jpg')
    ]
    # Multiprocessing untuk mempercepat ekstraksi fitur
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_image, image_paths), total=len(image_paths)))
    
    # Pisahkan hasil ke X dan y
    for features, age in results:
        X.append(features)
        y.append(age)
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Path ke dataset
    dataset_path = "data/utkface-new"  # Ganti dengan path dataset yang sesuai

    # Memuat dataset
    print("Memuat dataset...")
    X_features, y_labels = load_dataset(dataset_path)
    print("Dataset berhasil dimuat.")

    # Membagi dataset menjadi data latih dan data uji
    print("Membagi dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
    print("Dataset berhasil dibagi.")
    # Setelah load dataset dan pembagian data
    X_train, X_test = reduce_dimensionality(X_train, X_test)
    # Latih model SVM
    print("Melatih model SVM...")
    model = LinearSVC(dual='auto')
  # Kernel linear untuk efisiensi
    model.fit(X_train, y_train)
    print("Model berhasil dilatih.")

    # Simpan model ke file
    model_path = "app/models/svm_model.pkl"
    print(f"Menyimpan model ke {model_path}...")
    joblib.dump(model, model_path)
    print("Model berhasil disimpan.")

    # Evaluasi model
    print("Evaluasi model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {accuracy * 100:.2f}%")
    print("Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred))

    print("Pelatihan selesai!")
