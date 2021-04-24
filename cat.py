import cv2

from imgDetection import Img


class Cat(Img):
    def __init__(self, img_path, catCascade):

        self.img_path = img_path
        self.catCascade = catCascade

    def detect_face(self):
        img = cv2.imread(self.img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.catCascade.detectMultiScale(
            gray,  # selects grayscale image
            scaleFactor=1.03,  # determines distance to and from camera
            minNeighbors=2,  # objects detected near the current one
            minSize=(15, 15),  # size of the window
            maxSize=(1000, 1000)  # max window size
            # DEFAULT: 1.1, 5, 25, 1000
        )
        return faces
