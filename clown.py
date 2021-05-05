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
            scaleFactor=1.35,  # determines distance to and from camera
            minNeighbors=0,  # objects detected near the current one
            minSize=(50, 50),  # size of the window
            maxSize=(200, 200)  # max window size
            # DEFAULT: 1.1, 5, 25, 1000
        )

        eyes = self.eyeCascade.detectMultiScale(
            gray,  # selects grayscale image
            scaleFactor=1.1,  # determines distance to and from camera
            minNeighbors=0,  # objects detected near the current one
            minSize=(45, 45),  # size of the window
            maxSize=(50, 50)  # max window size
            # DEFAULT: 1.1, 5, 25, 1000
        )

        smile = self.smileCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # determines distance to and from camera
            minNeighbors=0,  # objects detected near the current one
            minSize=(150, 150),  # size of the window
            maxSize=(250, 250)  # max window size
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
        original_img = Image.open(self.img_path) #convert to png
        print(original_img.mode)

        nose = Image.open("images/clown_nose.jpg")
        print(nose.mode)
        eye = Image.open("images/clown_eye.jpg")
        smile = Image.open("images/clown_smile.jpg")
        for x, y, w, h in faces:
            coordinates = (x, y, x + w, y - h)
            original_img.paste(nose, coordinates)
        for x, y, w, h in eyes:
            coordinates = (x, y, x + w, y - h)
            original_img.paste(eyes, coordinates)
        original_img.save('out/clown_img.jpg')

        return original_img


