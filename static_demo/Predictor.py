from threading import Thread
import cv2 as cv
import numpy as np
from keras.models import model_from_json
from PIL import ImageColor
from queue import Queue


class Predictor(Thread):

    def __init__(self, queue, daemon=False):
        Thread.__init__(self, daemon=daemon)
        self.queue: Queue = queue
        self.preds = []

        json_file = open('./models/CNN2.json', 'r')
        model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(model_json)
        # load weights into new model
        self.model.load_weights("./models/CNN2.h5")
        print("Loaded model from disk")

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            image, face = self.queue.get()
            try:
                if image is None:
                    break

                pred = self._predict(image, face)
                self.preds.append(pred)
            finally:
                self.queue.task_done()

    def _predict(self, image: cv.Mat, face: tuple):

        emotion = ""
        emotion_color = ()
        face_frame = []

        (x, y, w, h) = face
        b = y + h
        r = x + w

        if w >= 48 and h >= 48:

            try:
                face_image = image[y:b, x:r]

                input_image = cv.resize(face_image, (48, 48))

                input_image = input_image.reshape(
                    (1, 48, 48, 1)) / 255  # type: ignore

                prediction = np.argmax(self.model.predict(input_image), axis=1)

                emotion_label_to_text = {
                    0: 'anger',
                    1: 'disgust',
                    2: 'fear',
                    3: 'happiness',
                    4: 'sadness',
                    5: 'surprise',
                    6: 'neutral'
                }

                colors = {
                    0: "#CD5B45",
                    1: "#228B22",
                    2: "#474747",
                    3: "#FFC125",
                    4: "#6495ED",
                    5: "#B03060",
                    6: "#ffffff"
                }

                emotion = emotion_label_to_text.get(prediction[0])

                color = colors.get(prediction[0])
                r, g, b = ImageColor.getcolor(color, "RGB")  # type: ignore
                emotion_color = b, g, r

                face_frame = face

            except cv.error:
                print("ERROR")

        return emotion, emotion_color, face_frame
