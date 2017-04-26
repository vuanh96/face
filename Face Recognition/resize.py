import cv2
import imutils
import os

for root, dirs, files in os.walk("images"):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        for file in os.listdir(dir_path):
            file = os.path.join(dir_path, file)
            img = cv2.imread(file, 1)
            img = imutils.resize(img, width=500)
            cv2.imwrite(file, img)