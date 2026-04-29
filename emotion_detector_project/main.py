import cv2
from detector.model import EmotionModel
from detector.preprocess import preprocess_face
from detector.utils import draw_label

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

emotion_model = EmotionModel('fer2013_big_XCEPTION.54-0.66.hdf5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        processed = preprocess_face(face)
        emotion = emotion_model.predict(processed)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        draw_label(frame, emotion, x, y-10)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
