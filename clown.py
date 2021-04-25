import cv2

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
            scaleFactor=1.03,  # determines distance to and from camera
            minNeighbors=10,  # objects detected near the current one
            minSize=(50, 50),  # size of the window
            maxSize=(500, 500)  # max window size
            # DEFAULT: 1.1, 5, 25, 1000
        )

        eyes = self.eyeCascade.detectMultiScale(
            gray,  # selects grayscale image
            scaleFactor=1.01,  # determines distance to and from camera
            minNeighbors=1,  # objects detected near the current one
            minSize=(10, 10),  # size of the window
            maxSize=(50, 50)  # max window size
            # DEFAULT: 1.1, 5, 25, 1000
        )

        smile = self.smileCascade.detectMultiScale(
            gray,
            scaleFactor=1.01,  # determines distance to and from camera
            minNeighbors=1,  # objects detected near the current one
            minSize=(10, 10),  # size of the window
            maxSize=(50, 50)  # max window size
            # DEFAULT: 1.1, 5, 25, 1000
        )
        # Draw the rectangle around each face

        def draw_rectangle(self, part, frame, r, g, b):
            for (x, y, w, h) in part:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (r, g, b), 2)


        self.draw_rectangle(faces, img, 255, 0, 0)

        self.draw_rectangle(eyes, img, 0, 255, 0)

        self.draw_rectangle(smile, img, 0, 0, 255)

