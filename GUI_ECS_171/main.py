import sys
from PIL import Image
import io
from sys import exit as exit
import controller


import cv2, PySimpleGUIQt as sg

# define the window layout


frame_up = sg.Frame('Parameter_tunning', [
    [
        sg.Radio('sample per 2', group_id='sample_size', default= True, enable_events=True, key='sample per 2'),
        sg.Radio('sample per 5', group_id='sample_size', enable_events=True, key='sample per 5'),
        sg.Radio('sample per 10', group_id='sample_size', enable_events=True, key='sample per 10'),
    ]], border_width=10, font='Helvetica 15 bold')
frame_middle = sg.Frame('captured face', [[
    sg.Image(filename='', key='_IMAGE_FACE_'),
]], border_width=10, font='Helvetica 15 bold')

frame_down = sg.Frame('result', [[
    sg.Text("no emotion detected", key="EMOTION", size=(300, 200), justification="center", auto_size_text=True, font='Helvetica 29 bold')
]], border_width=10, font='Helvetica 15 bold')






layout1 = [
[
        sg.Image(filename='', key='_IMAGE_'),
        sg.Frame(title='', layout=[[frame_up], [frame_middle], [frame_down]])
    ],
    [
        sg.Column(layout=[[sg.RButton('Exit', size=(10, 1), font='Helvetica 14'),
        sg.RButton('About', size=(10, 1), font='Helvetica 14')]], element_justification='center')

    ]
]

layout2 = [[sg.Input(key='_FILEBROWSE_', enable_events=True, visible=False)],
            [sg.FileBrowse(target='_FILEBROWSE_')],
            [sg.OK()]]

tabgrp = [sg.TabGroup([
            [
                sg.Tab('Personal Details', layout1, border_width=20,
                                tooltip='Image Stream', element_justification='center'),
                sg.Tab('Education', layout2, tooltip='Image Upload')]
            ], border_width=5)]



layout = [
    [sg.Text('ECS171 G8 GUI Demo by Yiyang Huo', size=(40, 1), justification='center', font='Helvetica 20')],
    tabgrp
]

# create the window and show it without the plot
window = sg.Window('Demo Application', layout).Finalize()
window.Maximize()
face_cascade = cv2.CascadeClassifier('face_default.xml')

nosignal = Image.open('no_signal.jpeg')
nosignal = nosignal.resize((300, 300))
buf = io.BytesIO()
nosignal.save(buf, format='JPEG')
nosignal_byte = buf.getvalue()

appcontroller = controller.Controller()


# ---===--- Event LOOP Read and display frames, operate the GUI --- #
cap = cv2.VideoCapture(0)                               # Setup the OpenCV capture device (webcam)
while True:

    button, values = window.Read(timeout=20, timeout_key='timeout')
    if button == 'Exit' or values == None:
        sys.exit(0)
    elif button == 'About':
        sg.PopupNoWait('Made with PySimpleGUI',
                       'www.PySimpleGUI.org',
                       'Check out how the video keeps playing behind this window.',
                       'I finally figured out how to display frames from a webcam.',
                       'ENJOY!  Go make something really cool with this... please!',
                       keep_on_top=True)
    elif button == "sample per 2":
        sg.PopupNoWait('sample the frame per 2 frames',
                       keep_on_top=False)
        appcontroller.setSample(2)
    elif button == "sample per 5":
        sg.PopupNoWait('sample the frame per 5 frames',
                       keep_on_top=False)
        appcontroller.setSample(5)
    elif button == "sample per 10":
        sg.PopupNoWait('sample the frame per 10 frames',
                       keep_on_top=False)
        appcontroller.setSample(10)


    ret, frame = cap.read()

    if appcontroller.counter % appcontroller.sample_size == 0:
        appcontroller.counter = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6)
        if len(faces) > 0:
            appcontroller.frame = faces[0]
            (x, y, w, h) = appcontroller.frame
            if w > 100 and h > 100:
                w = int(w * 1.1)
                h = int(h * 1.1)
                inspect_face = frame[y: y + h, x: x + w]
                imgbytes = cv2.imencode('.png', inspect_face)[1].tobytes()
                imageFile = Image.open(io.BytesIO(imgbytes))
                imageFile = imageFile.resize((int(h / w * 300), 300))
                buf = io.BytesIO()
                imageFile.save(buf, format='JPEG')
                byte_im = buf.getvalue()
                window.FindElement('_IMAGE_FACE_').Update(data=byte_im)

                appcontroller.update(imageFile)

                window.FindElement('EMOTION').Update(value=appcontroller.emotion.ui_component[0], text_color=appcontroller.emotion.ui_component[1])
        else:
            window.FindElement('_IMAGE_FACE_').Update(data=nosignal_byte)

    (x, y, w, h) = appcontroller.frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    appcontroller.counter += 1

    # Read image from capture device (camera)
    imgbytes=cv2.imencode('.png', frame)[1].tobytes()     # Convert the image to PNG Bytes
    window.FindElement('_IMAGE_').Update(data=imgbytes)   # Change the Image Element to show the new image