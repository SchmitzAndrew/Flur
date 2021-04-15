import cv2

from img import Img

#Cascades
faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt.xml")
#eyeCascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")

faceImg = Img("images/schmitz.jpg", faceCascade)

faces = faceImg.detect_face()
img = faceImg.draw_rectangle(faces)
faceImg.show_img(img)
