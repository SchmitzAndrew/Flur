import cv2


class Vid:
    def __init__(self, faceCascade, eyeCascade):
        self.faceCascade = faceCascade
        self.eyeCascade = eyeCascade

    def capture_video(self):
        #cap = cv2.VideoCapture(-1)  # To capture video from webcam

        cap = cv2.VideoCapture('images/octopus.mp4') # To use a video file as input
        while True:
            # Read the frame
            _, frame = cap.read()
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect the faces
            faces = self.faceCascade.detectMultiScale(
                gray,  # selects grayscale image
                scaleFactor=1.03,  # determines distance to and from camera
                minNeighbors=10,  # objects detected near the current one
                minSize=(25, 25),  # size of the window
                maxSize=(1000, 1000)  # max window size
                # DEFAULT: 1.1, 5, 25, 1000
            )
            # Detect the eyes
            eyes = self.eyeCascade.detectMultiScale(
                gray,  # selects grayscale image
                scaleFactor=1.03,  # determines distance to and from camera
                minNeighbors=10,  # objects detected near the current one
                minSize=(25, 25),  # size of the window
                maxSize=(1000, 1000)  # max window size
                # DEFAULT: 1.1, 5, 25, 1000
            )
            # Draw the rectangle around each face

            self.draw_rectangle(faces, frame)

            self.draw_rectangle(faces, frame)
            # Display
            cv2.imshow('img', frame)
            # Stop if escape key is pressed
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        # Release the VideoCapture object
        cap.release()

    def draw_rectangle(self, part, img):
        for (x, y, w, h) in part:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
