import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#@param: an OpenCV image object
#@return: an OpenCV image containing only the face
def detect_face(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	for (x,y,w,h) in faces:
		return img[y:y+h, x:x+w]
cv2.imwrite("test.jpg",detect_face(cv2.imread("SamuelLJackson.jpg")))
