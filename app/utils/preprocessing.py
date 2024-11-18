import cv2
import numpy as np

def extract_lbp_features(image_path):
    # Membaca gambar dalam grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Menghitung histogram LBP
    lbp = cv2.calcHist([image], [0], None, [256], [0, 256])
    return lbp.flatten()
