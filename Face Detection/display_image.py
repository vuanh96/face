import cv2
from imutils import paths


imagePaths = paths.list_images("../images")

for imagePath in imagePaths:
    img = cv2.imread(imagePath, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', img)
    cv2.waitKey(0)