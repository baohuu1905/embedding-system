import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = "data/weights.149-0.01.hdf5"
LANDMARK_PATH = "data/shape_predictor_68_face_landmarks.dat"

SCALE = 0.5
ALARM_THRESHOLD = 5
BEEP_FREQ = 2500
BEEP_DURATION = 1000