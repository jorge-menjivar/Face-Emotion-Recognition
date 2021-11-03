#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 11/2/21 5:20 PM
#@Author: Yiyang Huo
#@File  : main.py

import cv2, PySimpleGUIWeb as sg

# define the window layout
layout = [[sg.Image(filename='', key='_IMAGE_')]]

# create the window and show it without the plot
window = sg.Window('Demo Application - OpenCV Integration', layout, location=(800,400))

# ---===--- Event LOOP Read and display frames, operate the GUI --- #
cap = cv2.VideoCapture(0)                               # Setup the OpenCV capture device (webcam)
while True:
    event, values = window.Read(timeout=20, timeout_key='timeout')
    if event is None:
        break
    ret, frame = cap.read()                               # Read image from capture device (camera)
    imgbytes=cv2.imencode('.png', frame)[1].tobytes()     # Convert the image to PNG Bytes
    window.FindElement('_IMAGE_').Update(data=imgbytes)   # Change the Image Element to show the new image