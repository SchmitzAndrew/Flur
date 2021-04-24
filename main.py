import cv2

#Modules
from imgDetection import Img
from vidDetection import Vid
from blur import blur
from cat import Cat

#Cascades
faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt.xml")
eyeCascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")
catCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalcatface.xml")

# detecting faces and blurring
input_image = "images/daft_punk_unmasked.jpg"
faceImg = Img(input_image, faceCascade)
faces = faceImg.detect_face()
img = faceImg.draw_rectangle(faces)
blur(input_image, faces, 5)
faceImg.show_img(img)

#cat detection
input_image = "images/black_cat.jfif"
catImg = Cat(input_image, catCascade)
faces = catImg.detect_face()
img = catImg.draw_rectangle(faces)
blur(input_image, faces, 10)
catImg.show_img(img)

#image detection
#faceVid = Vid(faceCascade, eyeCascade)
#faceVid.capture_video()

