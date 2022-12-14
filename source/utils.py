import os
import cv2
from keras_preprocessing.image import img_to_array


def preprocess_face_frame(face_frame):
    # convert to RGB
    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    # preprocess input image for mobilenet
    face_frame_resized = cv2.resize(face_frame, (224, 224))
    face_frame_array = img_to_array(face_frame_resized)
    return face_frame_array


def decode_prediction(pred):
    (no_smoking, smoking) = pred
    smoking_or_not = "Smoking" if smoking > no_smoking else "No Smoking"
    confidence = f"{(max(smoking, no_smoking) * 100):.2f}"
    return smoking_or_not, confidence


def write_bb(smoking_or_not, confidence, box, frame):
    (x, y, w, h) = box
    if smoking_or_not == "No Smoking":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    label = f"{smoking_or_not}: {confidence}%"

    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def load_cascade_detector():
    cascade_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)
    return face_detector

