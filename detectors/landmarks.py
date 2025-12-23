import dlib
from imutils import face_utils

class FacialLandmarks:
    LEFT_EYE_START, LEFT_EYE_END = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    RIGHT_EYE_START, RIGHT_EYE_END = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def __init__(self, predictor_path):
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect(self, gray_image, face_rect):
        shape = self.predictor(gray_image, face_rect)
        return face_utils.shape_to_np(shape)
