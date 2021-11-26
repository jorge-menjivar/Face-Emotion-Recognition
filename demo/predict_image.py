from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import model_from_json
import numpy as np


def run(image: Image):
    input_image = img_to_array(image)
    input_image = input_image.reshape((1, 48, 48, 1))/255

    json_file = open('CNN2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("CNN2.h5")
    print("Loaded model from disk")

    prediction = np.argmax(loaded_model.predict(input_image), axis=1)

    emotion_label_to_text = {
        0: 'anger',
        1: 'disgust',
        2: 'fear',
        3: 'happiness',
        4: 'sadness',
        5: 'surprise',
        6: 'neutral'
    }

    return {"emotion": emotion_label_to_text.get(prediction[0])}
