import asyncio
import io
import cv2 as cv
import PySimpleGUIQt as sg
from keras.models import model_from_json
import numpy as np
from PIL import ImageColor

sample_frequency = 5

json_file = open('./models/CNN2.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("./models/CNN2.h5")
print("Loaded model from disk")

# define the window layout

sg.theme('Dark2')

# Sample Frequency
freq_widget = sg.Frame(
    title="Sample Frequency",
    layout=[[
        sg.Radio(
            '2 frames',
            group_id='sample_size',
            enable_events=True,
            key='sample per 2',
            font='_ 12',
        ),
        sg.Radio(
            '5 frames',
            group_id='sample_size',
            enable_events=True,
            default=True,
            key='sample per 5',
            font='_ 12',
        ),
        sg.Radio(
            '10 frames',
            group_id='sample_size',
            enable_events=True,
            key='sample per 10',
            font='_ 12',
        ),
    ]],
    font='_ 14',
)

# Full Layout
layout = [
    [freq_widget],
    [sg.Image(filename='', key='CAM')],  # Camera Widget
]

# create the window and show it without the plot
window = sg.Window('Emotion Recognition', layout)
face_cascade = cv.CascadeClassifier('face_default.xml')

emotions = []
emotion_colors = []
face_frames = []


async def predict(image: cv.Mat, faces: list[tuple]):

    emotions.clear()
    emotion_colors.clear()
    face_frames.clear()

    for face in faces:
        (x, y, w, h) = face
        b = y + h
        r = x + w

        if w >= 48 and h >= 48:

            try:
                face_image = image[y:b, x:r]

                input_image = cv.resize(face_image, (48, 48))

                input_image = input_image.reshape(
                    (1, 48, 48, 1)) / 255  # type: ignore

                prediction = np.argmax(model.predict(input_image), axis=1)

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

                emotions.append(emotion_label_to_text.get(prediction[0]))

                color = colors.get(prediction[0])
                r, g, b = ImageColor.getcolor(color, "RGB")  # type: ignore
                emotion_colors.append((b, g, r))

                face_frames.append(face)

            except cv.error:
                print("ERROR")

    return


buf = io.BytesIO()

cap = cv.VideoCapture(0)  # Setup the OpenCV capture device (webcam)
print(sg.theme_list())

# ---------------------------------- Rendering -------------------------------

frame_count = 1
while True:

    event, values = window.Read(timeout=20, timeout_key='timeout')
    if event in (sg.WIN_CLOSED, 'Quit'):
        break
    elif event == "sample per 2":
        sample_frequency = 2
    elif event == "sample per 5":
        sample_frequency = 5
    elif event == "sample per 10":
        sample_frequency = 10

    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if frame_count % sample_frequency == 0:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6)

        asyncio.run(predict(gray, faces))

    for emotion, color, face_frame in zip(emotions, emotion_colors,
                                          face_frames):
        (x, y, w, h) = face_frame
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv.putText(
            frame,
            emotion,
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            .6,
            color,
            1,
            cv.LINE_AA,
        )

    # Update camera window
    imgbytes = cv.imencode('.png', frame)[1].tobytes()
    window['CAM'].update(data=imgbytes)  # type: ignore

    if frame_count == 60:
        frame_count = 0

    frame_count += 1

window.close()
cap.release()
