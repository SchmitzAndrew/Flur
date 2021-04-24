import cv2


from imgDetection import Img
from vidDetection import Vid
from blur import blur

#Cascades
faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt.xml")
eyeCascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")

faceImg = Img("images/daft_punk_unmasked.jpg", faceCascade)
faces = faceImg.detect_face()
img = faceImg.draw_rectangle(faces)
blur("images/daft_punk_unmasked.jpg", faces, 5)
faceImg.show_img(img)

#faceVid = Vid(faceCascade, eyeCascade)
#faceVid.capture_video()

