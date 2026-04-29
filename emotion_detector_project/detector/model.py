import tensorflow as tf
from tensorflow.keras.models import load_model

class EmotionModel:
    def __init__(self, model_path: str):
        self.model = load_model(model_path, compile=False)
        self.labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def predict(self, face_img):
        preds = self.model.predict(face_img)
        return self.labels[preds.argmax()]
