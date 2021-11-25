from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, Form, File, UploadFile
import uvicorn
from PIL import Image

from demo import predict_image

app = FastAPI(root_path="/")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static/templates")

origins = [
    "http://localhost:8855",
    "http://localhost:8855/upload_file",
    "*"
]


@app.get("/", response_class=HTMLResponse)
async def display_home(request: Request):
    return templates.TemplateResponse('home.html', {'request': request})


@app.post("/upload_file")
async def upload_file(user_image: UploadFile = File(...)):
    image = Image.open(user_image.file).convert('L')
    image = image.resize((48, 48))
    emotion = predict_image.run(image)
    return emotion


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="localhost",
        port=8855,
    )
