import os
import dlib
import glob

import math
from skimage import io


def Euclidean(d1, d2):
    T = 0
    for i in range(len(d1)):
        T += (d1[i] - d2[i])**2
    return math.sqrt(T)

# "You can download a trained facial shape predictor and recognition model from:\n"
# "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
# "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")


predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = "Quang"


# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

win = dlib.image_window()

# Now process all the images
for f in glob.glob(os.path.join(faces_folder_path, "*.JPG")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # Now process each face we found.
    face_descriptors = []
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)
        # Draw the face landmarks on the screen so we can see what face is currently being processed.

        win.clear_overlay()
        win.add_overlay(d)
        win.add_overlay(shape)

        # Compute the 128D vector that describes the face in img identified by
        # shape.  In general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people.  He we just print
        # the vector to the screen.

        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(face_descriptor)
        # print(face_descriptor)

        # It should also be noted that you can also call this function like this:

        #  face_descriptor = facerec.compute_face_descriptor(img, shape, 100)

        # The version of the call without the 100 gets 99.13% accuracy on LFW
        # while the version with 100 gets 99.38%.  However, the 100 makes the
        # call 100x slower to execute, so choose whatever version you like.  To
        # explain a little, the 3rd argument tells the code how many times to
        # jitter/resample the image.  When you set it to 100 it executes the
        # face descriptor extraction 100 times on slightly modified versions of
        # the face and returns the average result.  You could also pick a more
        # middle value, such as 10, which is only 10x slower but still gets an
        # LFW accuracy of 99.3%.

        dlib.hit_enter_to_continue()

    for i in range(len(face_descriptors)-1):
        print("Person ", i)
        for j in range(i+1, len(face_descriptors)):
            print(Euclidean(face_descriptors[i], face_descriptors[j]))


