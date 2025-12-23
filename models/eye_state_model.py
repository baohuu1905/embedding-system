import cv2
import numpy as np
import keras
from tensorflow.keras.models import load_model

class EyeStateModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, eye_image):
        eye_image = cv2.resize(eye_image, (20, 10))
        eye_image = eye_image.astype(np.float32)
        eye_image = np.reshape(eye_image, (1, 10, 20, 1))
        eye_image = keras.applications.mobilenet.preprocess_input(eye_image)
        return np.argmax(self.model.predict(eye_image)[0])
