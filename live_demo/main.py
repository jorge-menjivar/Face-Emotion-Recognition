import io
from queue import Queue
import cv2 as cv
import PySimpleGUIQt as sg

from Predictor import Predictor

sample_frequency = 5

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

window = sg.Window(
    'Emotion Recognition',
    layout,
    resizable=True,
    finalize=True,
)

face_cascade = cv.CascadeClassifier('face_default.xml')

buf = io.BytesIO()

cap = cv.VideoCapture(0)  # Setup the OpenCV capture device (webcam)

queue = Queue()

threads: list[Predictor] = []
t1 = Predictor(queue, daemon=True)
t2 = Predictor(queue, daemon=True)
t3 = Predictor(queue, daemon=True)
t4 = Predictor(queue, daemon=True)
t1.start()
t2.start()
t3.start()
t4.start()

threads.append(t1)
threads.append(t2)
threads.append(t3)
threads.append(t4)

# ---------------------------------- Rendering -------------------------------

CAM_WINDOW_W = 700
frame_count = 1

emotions = []
emotion_colors = []
face_frames = []

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
        im_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(im_gray, 1.1, 6)

        for face in faces:
            queue.put((im_gray, face))

        emotions = []
        emotion_colors = []
        face_frames = []

        for t in threads:
            preds = t.preds

            for pred in preds:
                emotion = pred[0]
                emotion_color = pred[1]
                face_frame = pred[2]

                emotions.append(emotion)
                emotion_colors.append(emotion_color)
                face_frames.append(face_frame)

            t.preds.clear()

    # Displaying the predictions of the predictor thread
    for emotion, color, face_frame in zip(emotions, emotion_colors,
                                          face_frames):
        (x, y, w, h) = face_frame
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv.putText(
            frame,
            emotion,
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            1,
            cv.LINE_AA,
        )

    # Update camera window
    h, w, _ = frame.shape

    diff = ((CAM_WINDOW_W - 20) - w) / w

    new_w = CAM_WINDOW_W
    new_h = round(h + (h * diff))

    frame = cv.resize(frame, (new_w, new_h))  # type: ignore
    imgbytes = cv.imencode('.png', frame)[1].tobytes()
    window['CAM'].update(data=imgbytes)  # type: ignore

    if frame_count == 60:
        frame_count = 0

    frame_count += 1

window.close()
queue.put((None, None))
cap.release()
