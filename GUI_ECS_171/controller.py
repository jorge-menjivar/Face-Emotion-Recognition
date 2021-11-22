from enum import Enum


class Emotions(Enum):
    NOEMOTION = ["no emotion detected", "white"]
    ANGER = ["Anger", "orange red"]
    DISGUST = ["Disgust", "forest green"]
    FEAR = ["Fear", "blue violet"]
    HAPPY = ["Happy", "green"]
    SAD = ["Sad", "grey39"]
    SURPRISE = ["Surprise", "turquoise1"]
    NEUTRAL = ["Netral", "white"]


class Controller:
    def __init__(self, model=None):
        self.sample_size = 2
        self.model = model
        self.counter = 0
        self.frame = (0, 0, 0, 0)
        self.emotion = Emotions.NOEMOTION

    def setSample(self, size):
        self.sample_size = size
        self.counter = 0

    def update(self, inspecting_picture):
        self.emotion = Emotions.HAPPY
        # TODO, do the model prediction and outcome inside here, inspecting picture are in the form of pillow
