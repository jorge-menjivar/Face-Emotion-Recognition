from enum import Enum
from keras.models import model_from_json
from PIL import Image, ImageOps
import numpy as np


class Emotions(Enum):
    NOEMOTION = 0, ["no emotion detected", "white"]
    ANGER = 1, ["Anger", "orange red"]
    DISGUST = 2, ["Disgust", "forest green"]
    FEAR = 3, ["Fear", "blue violet"]
    HAPPY = 4, ["Happy", "green"]
    SAD = 5, ["Sad", "grey39"]
    SURPRISE = 6, ["Surprise", "turquoise1"]
    NEUTRAL = 7, ["Netral", "white"]

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.ui_component = name
        return member

    def __int__(self):
        return self.value


class Controller:
    def __init__(self, model=None):
        self.sample_size = 2
        self.model = model
        self.counter = 0
        self.frame = (0, 0, 0, 0)
        self.emotion = Emotions.NOEMOTION
        self.model = self.loadModels()
        self.image_width, self.image_height = 48, 48
        self.emotion_label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

    def loadModels(self):
        json_file = open('./models/CNN.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("./models/CNN.h5")
        return loaded_model

    def setSample(self, size):
        self.sample_size = size
        self.counter = 0

    def update(self, inspecting_picture):
        gray_image = ImageOps.grayscale(inspecting_picture)
        gray_image = gray_image.resize((self.image_width, self.image_height))
        the_array = np.asarray(gray_image)
        the_array = the_array.reshape(1, self.image_width, self.image_height, 1) / 255
        y_pred = self.model.predict(the_array)
        #emotion_predict = self.emotion_label_to_text[np.argmax(y_pred)]
        emotion_predict = np.argmax(y_pred) + 1
        self.emotion = Emotions(emotion_predict)
        # TODO, do the model prediction and outcome inside here, inspecting picture are in the form of pillow
