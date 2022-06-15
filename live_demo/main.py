import asyncio
import io
import cv2 as cv
import PySimpleGUIQt as sg
from keras.models import model_from_json
import numpy as np

sample_frequency = 5

json_file = open('./models/CNN.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("./models/CNN.h5")
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
    font='_ 16',
)

# Outputing Prediction
output_widget = sg.Frame(
    title='Emotion:',
    layout=[[sg.Text(
        "",
        key="RES",
        justification="center",
        font='_ 16',
    )]],
    border_width=10,
    font='_ 16',
)

# Full Layout
layout = [
    [
        sg.Image(filename='', key='CAM'),  # Camera Widget
        sg.Frame(
            title='',
            layout=[[freq_widget], [output_widget]],
            border_width=0,
        ),
    ],
]

# create the window and show it without the plot
window = sg.Window('Emotion Recognition', layout).Finalize()
face_cascade = cv.CascadeClassifier('face_default.xml')


async def predict(image: cv.Mat, face: tuple, _window: sg.Window):
    (x, y, w, h) = face

    emotion = ""
    text_color = "white"

    if w >= 48 and h >= 48:
        image = image[y:y + h, x:x + w]

        input_image = cv.resize(image, (48, 48))

        input_image = input_image.reshape((1, 48, 48, 1)) / 255  # type: ignore

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
            6: "white"
        }

        emotion = emotion_label_to_text.get(prediction[0])
        text_color = colors.get(prediction[0])

    _window['RES'].update(value=emotion, text_color=text_color)  # type: ignore


buf = io.BytesIO()

cap = cv.VideoCapture(0)  # Setup the OpenCV capture device (webcam)
print(sg.theme_list())

# ---------------------------------- Rendering -------------------------------

face_frame = -1, -1, -1, -1
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

        for face in faces:
            face_frame = face
            asyncio.run(predict(gray, face, window))

    (x, y, w, h) = face_frame
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Update camera window
    imgbytes = cv.imencode('.png', frame)[1].tobytes()
    window['CAM'].update(data=imgbytes)  # type: ignore

    if frame_count == 60:
        frame_count = 0

    frame_count += 1

window.close()
cap.release()
