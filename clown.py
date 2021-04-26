import cv2
from PIL import Image

from imgDetection import Img


class Clown(Img):
    def __init__(self, img_path, faceCascade, eyeCascade, smileCascade):
        self.img_path = img_path
        self.faceCascade = faceCascade
        self.eyeCascade = eyeCascade
        self.smileCascade = smileCascade

    def detect_features(self):
        img = cv2.imread(self.img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the features

        faces = self.faceCascade.detectMultiScale(
            gray,  # selects grayscale image
            scaleFactor=1.02,  # determines distance to and from camera
            minNeighbors=1,  # objects detected near the current one
            minSize=(50, 50),  # size of the window
            maxSize=(200, 200)  # max window size
            # DEFAULT: 1.1, 5, 25, 1000
        )

        eyes = self.eyeCascade.detectMultiScale(
            gray,  # selects grayscale image
            scaleFactor=1.146,  # determines distance to and from camera
            minNeighbors=0,  # objects detected near the current one
            minSize=(45, 45),  # size of the window
            maxSize=(50, 50)  # max window size
            # DEFAULT: 1.1, 5, 25, 1000
        )

        smile = self.smileCascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # determines distance to and from camera
            minNeighbors=0,  # objects detected near the current one
            minSize=(25, 25),  # size of the window
            maxSize=(50, 50)  # max window size
            # DEFAULT: 1.1, 5, 25, 1000
        )
        self.draw_rectangle(faces, img,  255, 0, 0)
        self.draw_rectangle(eyes, img, 0, 255, 0)
        self.draw_rectangle(smile, img, 0, 0, 255)

        final_img = self.place_parts(faces, eyes, smile)
        return final_img

    # Draw the rectangle around each face
    def draw_rectangle(self, part, img_path, r, g, b):
        img = cv2.imread(self.img_path)

        for (x, y, w, h) in part:
            cv2.rectangle(img, (x, y), (x + w, y + h), (r, g, b), 2)
        self.show_img(img)

    def place_parts(self, faces, eyes, smile):
        original_img = Image.open(self.img_path)
        nose = Image.open("images/clown_nose.png")
        eye = Image.open("images/clown_eye.png")
        smile = Image.open("images/clown_smile.png")
        for x, y, w, h in faces:
            original_img.paste(nose, ((x + w) / 2), ((y + h) / 2))
        for x, y, w, h in eyes:
            original_img.paste(eyes, ((x + w) / 2), ((y + h) / 2))

        return original_img


