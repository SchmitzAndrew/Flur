import cv2

from imgDetection import Img
from vidDetection import Vid

#Cascades
faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt.xml")
eyeCascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")

faceImg = Img("images/lcd_soundsystem.jpg", faceCascade)

# faces = faceImg.detect_face()
# img = faceImg.draw_rectangle(faces)
# faceImg.show_img(img)

faceVid = Vid(faceCascade, eyeCascade)
faceVid.capture_video()

