import base64
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, File, UploadFile
from keras.preprocessing.image import img_to_array
from PIL import Image
from queue import Queue
from Predictor import Predictor
import uvicorn
import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('face_default.xml')
queue = Queue()
predictor_thread = Predictor(queue)
predictor_thread.start()

app = FastAPI(root_path="/")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static/templates")

origins = ["http://localhost:8855", "http://localhost:8855/upload_file", "*"]


@app.get("/", response_class=HTMLResponse)
async def display_home(request: Request):
    return templates.TemplateResponse('home.html', {'request': request})


@app.post("/upload_file")
async def upload_file(user_image: UploadFile = File(...)):
    image = Image.open(user_image.file)

    np_image = img_to_array(image)
    np_image = np.array(np_image, dtype='uint8')
    color = cv.cvtColor(np_image, cv.IMREAD_ANYCOLOR)
    gray = cv.cvtColor(np_image, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 6)

    for face in faces:
        queue.put((gray, face))

    emotions = []
    emotion_colors = []
    face_frames = []

    while queue.qsize() > 0:
        try:
            emotion = predictor_thread.emotion
            emotion_color = predictor_thread.emotion_color
            face_frame = predictor_thread.face_frame

            emotions.append(emotion)
            emotion_colors.append(emotion_color)
            face_frames.append(face_frame)

            (x, y, w, h) = face_frame
            cv.rectangle(np_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv.putText(
                np_image,
                emotion,
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                1,
                cv.LINE_AA,
            )

        except:
            pass

    image_bytes = cv.imencode('.png', color)[1]
    encoded_image_string = base64.b64encode(image_bytes)

    print(f"emotions: {emotions}")

    return {"image": encoded_image_string, "emotions": emotions}


if __name__ == "__main__":
    uvicorn.run(
        app,  # type: ignore
        host="localhost",
        port=8855,
    )
