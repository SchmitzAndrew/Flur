import cv2

# Modules
from imgDetection import Img
from vidDetection import Vid
from blur import blur
from cat import Cat
from clown import Clown

# Cascades
faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt.xml")
eyeCascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier("cascades/haarcascade_smile.xml")
catCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalcatface.xml")


# detecting faces and blurring
def facial_blur():
    input_image = "images/lcd_soundsystem.jpg"
    faceImg = Img(input_image, faceCascade)
    faces = faceImg.detect_face()
    img = faceImg.draw_rectangle(faces)
    blur(input_image, faces, 5)
    faceImg.show_img(img)


# cat detection
def cat_detection():
    input_image = "images/black_cat.jfif"
    catImg = Cat(input_image, catCascade)
    faces = catImg.detect_face()
    img = catImg.draw_rectangle(faces)
    blur(input_image, faces, 10)
    catImg.show_img(img)


# video detection
def video_detection():
    faceVid = Vid(faceCascade, eyeCascade)
    faceVid.capture_video()


# clown transformation
def clown_transformation():
    input_image = "images/carrey2.jpg"
    clownImg = Clown(input_image, faceCascade, eyeCascade, smileCascade)
    img = clownImg.detect_features()
    clownImg.show_img(img)


# Prompts and choice:

print("Flur- Facial blurring and PIL transformations")
print("1- Facial Blur")
print("2- Cat Blur")
print("3- Video Detection")
print("4- Clown Transformation")

selection = int(input("Enter the method to call:"))

if selection == 1:
    facial_blur()
elif selection == 2:
    cat_detection()
elif selection == 3:
    video_detection()
elif selection ==4:
    clown_transformation()
