import cv2
import face_recognition
import dlib

from utils.config import *
from detectors.landmarks import FacialLandmarks
from detectors.eye_extractor import extract_eye
from models.eye_state_model import EyeStateModel
from utils.alarm import trigger_alarm

# Init
cap = cv2.VideoCapture(0)
eye_model = EyeStateModel(MODEL_PATH)
landmark_detector = FacialLandmarks(LANDMARK_PATH)

count_close = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (0, 0), fx=SCALE, fy=SCALE)

    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    l, _, _ = cv2.split(lab)

    faces = face_recognition.face_locations(l, model="hog")

    if faces:
        top, right, bottom, left = faces[0]

        h_ratio = frame.shape[0] / l.shape[0]
        w_ratio = frame.shape[1] / l.shape[1]

        x1, y1, x2, y2 = int(left * w_ratio), int(top * h_ratio), int(right * w_ratio), int(bottom * h_ratio)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect = dlib.rectangle(x1, y1, x2, y2)

        landmarks = landmark_detector.detect(gray, rect)

        left_eye = extract_eye(
            gray,
            landmarks,
            FacialLandmarks.LEFT_EYE_START,
            FacialLandmarks.LEFT_EYE_END
        )

        right_eye = extract_eye(
            gray,
            landmarks,
            FacialLandmarks.RIGHT_EYE_START,
            FacialLandmarks.RIGHT_EYE_END
        )

        left_open = eye_model.predict(left_eye)
        right_open = eye_model.predict(right_eye)

        if left_open and right_open:
            count_close = 0
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            count_close += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if count_close > ALARM_THRESHOLD:
        trigger_alarm(frame, BEEP_FREQ, BEEP_DURATION)

    cv2.imshow("Sleep Detection", cv2.flip(frame, 1))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
