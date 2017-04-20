import cv2
import imutils

face_cascade = cv2.CascadeClassifier("../data/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

img = cv2.imread("VuAnh/va3.JPG", 1)
img = imutils.resize(img, width=600)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    # flags=cv2.CV_HAAR_SCALE_IMAGE
)
print("Detected %s face" % len(faces))
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
