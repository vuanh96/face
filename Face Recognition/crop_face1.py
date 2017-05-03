import cv2
import os
import imutils
from create_csv import create_csv


def readFileNames(file_name):
    try:
        inFile = open(file_name)
    except:
        raise IOError('There is no file named path_to_created_csv_file.csv in current directory.')

    picPath = []
    picIndex = []

    for line in inFile.readlines():
        if line != '':
            fields = line.rstrip().split(';')
            picPath.append(fields[0])
            picIndex.append(int(fields[1]))

    return picPath, picIndex


def crop(images_dir, images_crop_dir):
    # create csv_file: list_crop.txt from dir images
    create_csv(images_dir, "list_crop.txt")

    # read file image list need crop
    images, indexes = readFileNames("list_crop.txt")

    # make directory "at" storage images crop
    if not os.path.exists(images_crop_dir):
        os.makedirs(images_crop_dir)
    for i in images:
        dir_name = images_crop_dir + '/' + i.rstrip().split("/")[1]
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    # load file cascade
    face_cascade = cv2.CascadeClassifier("../data/haarcascades/haarcascade_frontalface_alt.xml")
    for image in images:
        src = cv2.imread(image, 1)
        src = imutils.resize(src, width=600)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # detect face
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        if len(faces) != 1:
            print("Error Image {}: detect {} faces".format(image, len(faces)))
            continue
        # crop image and save to dir at
        for (x, y, w, h) in faces:
            face_resize = cv2.resize(src[y:y + h, x:x + w], (200, 200))
            cv2.imwrite(image.replace(images_dir, images_crop_dir), face_resize)

    print("Crop finished")

crop("images1", "at10")
