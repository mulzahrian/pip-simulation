import cv2
import numpy as np

def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (64, 64))   # ✅ sesuai model
    face = face / 255.0
    face = np.reshape(face, (1, 64, 64, 1))  # ✅ HARUS sama
    return face