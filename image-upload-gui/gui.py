#references: https://www.blog.pythonlibrary.org/2021/02/16/creating-an-image-viewer-with-pysimplegui/
from html.parser import HTMLParser
import PySimpleGUIWeb as sg
import io
import os

from PIL import Image
file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]



def main():
    sg.theme('LightBlue')   # Add a touch of color
    layout = [
        [sg.Text('ECS 171 Machine Learning Project - Group 8')],
        [
            sg.Text("Entire image filename:"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.Button("Load Image"),
        ],
        [sg.Image(key="-IMAGE-")],
    ]
    window = sg.Window("Image Viewer", layout)
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Load Image":
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(values["-FILE-"])
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())
    window.close()
if __name__ == "__main__":
    main()
