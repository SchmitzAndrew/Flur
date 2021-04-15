import cv2

class Img:
    def __init__(self, img_path, faceCascade):
        self.img_path = img_path
        self.faceCascade = faceCascade

    def detect_face(self):
        img = cv2.imread(self.img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,  # selects grayscale image
            scaleFactor=1.05,  # determines distance to and from camera
            minNeighbors=10,  # objects detected near the current one
            minSize=(25, 25),  # size of the window
            maxSize=(1000, 1000)  # max window size
            # DEFAULT: 1.1, 5, 25, 1000
        )
        return faces

    def draw_rectangle(self, faces):
        img = cv2.imread(self.img_path)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (52, 207, 204), 2)
        return img

    def show_img(self, img):
        cv2.imshow("Faces found", img)
        cv2.waitKey(0)
        print("Done")
