# top_bottom_classifier.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

class AttireClassifier:
    def __init__(self, top_model_path="top_model.h5", bottom_model_path="bottom_model.h5"):
        """
        top_model: classifies upper body (shirt, t-shirt, kurta, dress)
        bottom_model: classifies lower body (trouser, jeans, saree)
        """
        print("[INFO] Loading top and bottom classifiers...")
        self.top_model = load_model(top_model_path)
        self.bottom_model = load_model(bottom_model_path)

        # Mapping indices to labels (modify according to your training)
        self.top_labels_map = {0: "T-shirt", 1: "Shirt", 2: "Kurta", 3: "Dress"}
        self.bottom_labels_map = {0: "Trouser", 1: "Jeans", 2: "SareeBottom"}

    def preprocess_crop(self, img):
        """Resize and normalize crop for classifier"""
        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict_top(self, top_crop):
        top_input = self.preprocess_crop(top_crop)
        pred = self.top_model.predict(top_input, verbose=0)
        top_idx = np.argmax(pred)
        return self.top_labels_map[top_idx]

    def predict_bottom(self, bottom_crop):
        bottom_input = self.preprocess_crop(bottom_crop)
        pred = self.bottom_model.predict(bottom_input, verbose=0)
        bottom_idx = np.argmax(pred)
        return self.bottom_labels_map[bottom_idx]

    def predict_attire(self, top_crop, bottom_crop):
        """
        Returns:
            top_label, bottom_label
        """
        top_label = self.predict_top(top_crop)
        bottom_label = self.predict_bottom(bottom_crop)
        return top_label, bottom_label

# -----------------------------
# Standalone test
# -----------------------------
if __name__ == "__main__":
    classifier = AttireClassifier()

    # Test with example images
    top_crop = cv2.imread("example_top.jpg")
    bottom_crop = cv2.imread("example_bottom.jpg")

    top_label, bottom_label = classifier.predict_attire(top_crop, bottom_crop)
    print(f"Top: {top_label}, Bottom: {bottom_label}")
