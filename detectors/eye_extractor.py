import cv2
import numpy as np
# from .landmarks import FacialLandmarks

def extract_eye(gray_image, landmarks, start, end):
    eye_points = landmarks[start:end]
    (x, y, w, h) = cv2.boundingRect(np.array([eye_points]))
    return gray_image[y:y + h, x:x + w]
