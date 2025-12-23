import cv2
import winsound

def trigger_alarm(frame, freq, duration):
    cv2.putText(
        frame,
        "Sleep detected! Alarm!",
        (100, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        lineType=cv2.LINE_AA
    )
    winsound.Beep(freq, duration)
