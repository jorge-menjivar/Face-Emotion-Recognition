#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 4/7/21 12:33 AM
#@Author: Yiyang Huo
#@File  : Utilize.py

import cv2
from PIL import Image
import numpy as np
from keras.models import Sequential, load_model


IMAGE_SIZE = 256
font = cv2.FONT_HERSHEY_SIMPLEX


# fontScale of presented message
fontScale = 4

# Blue color in BGR of presented message
color = (255, 0, 0)

# Line thickness of 2 px of presented message
thickness = 3

# Using cv2.putText() method


# program entrance
if __name__ == "__main__":
    # define a video capture object
    vid = cv2.VideoCapture(0)

    # load the model
    model = load_model("model.h5")


    # a forever loop until quit signal issued, present the image frame by frame
    while (True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read() # This line read the video from webcam
        (h, w) = frame.shape[:2] # Get the hight and width of the window

        # Lines below first turn frame into Pillow Image class, and then modify the image in the same way in train and test
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(frame)
        pil_im = pil_im.convert('L')
        pil_im = pil_im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
        sample_to_predict = np.array(pil_im).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

        # Use the model to predict the image in webcam, get the result vector
        predictions = model.predict(np.array(sample_to_predict))

        # [1,0] means that it is cat so if the result vector nears [1,0], we accept it as a cat
        if predictions[0][0] >0.9 and predictions[0][0] < 1.1 and predictions[0][1] < 0.1 and predictions[0][1] > -0.1:

            # add a message into the window if cat is spotted
            frame = cv2.putText(frame, 'There is a cat', (w//16, h//2), font,
                                fontScale, color, thickness, cv2.LINE_AA)
        # Present the frame into window
        cv2.imshow('frame', frame)

        # print prediction vector, use to monitor
        print(predictions)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()