import cv2

def extract_lbp_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Gambar tidak valid")
    image = cv2.resize(image, (128, 128))
    lbp = cv2.calcHist([image], [0], None, [256], [0, 256])
    return lbp.flatten()
