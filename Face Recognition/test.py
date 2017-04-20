import cv2
import os
from create_csv import create_csv


def readFileNames(file_name):
    try:
        inFile = open(file_name)
    except:
        raise IOError('There is no file named path_to_created_csv_file.csv in current directory.')

    for line in inFile.readlines():
        if line != '':
            fields = line.rstrip().split(';')
            yield fields[0], fields[1]

# create csv_file: list_crop.txt from dir images
create_csv("images", "list_crop.txt")

# read file image list need crop
images, indexs = readFileNames("list_crop.txt")
for image in images:
    print(image)