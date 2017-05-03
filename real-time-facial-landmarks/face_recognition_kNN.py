import time

import cv2
import dlib
import imutils
import numpy as np
import os

from imutils.video import VideoStream
from skimage import io
from sklearn.neighbors import KNeighborsClassifier

train_data_dir = "at10"
file_data = "data.npy"
file_classes = "classes.npy"

predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


def save_features():
    print("[INFO:] compute descriptor.......")

    face_descriptors = []
    classes = []
    for rootdir, dirnames, filenames in os.walk(train_data_dir):
        for subdirname in sorted(dirnames):
            subject_path = os.path.join(rootdir, subdirname)
            for filename in os.listdir(subject_path):
                file_path = os.path.join(subject_path, filename)
                img = io.imread(file_path)
                dets = detector(img, 1)
                for det in dets:
                    shape = sp(img, det)
                    face_descriptor = facerec.compute_face_descriptor(img, shape)
                    face_descriptors.append(face_descriptor)
                    classes.append(subdirname)
    np.save(file_data, face_descriptors)
    np.save(file_classes, classes)


def predict(image_path):
    data = np.load(file_data)
    classes = np.load(file_classes)

    print("[INFO] training classifier...")
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(data, classes)

    print("[INFO] predicting image ...")
    frame = cv2.imread(image_path, 1)
    if len(frame[0]) > 1000:
        frame = imutils.resize(frame, width=1000)

    # detect faces in the grayscale frame
    dets = detector(frame, 1)

    # loop over the face detections
    print("Number detected:", len(dets))
    for det in dets:
        x, y, z, t = det.left(), det.top(), det.right(), det.bottom()
        cv2.rectangle(frame, (x, y), (z, t), (0, 255, 0), 2)

        shape = sp(frame, det)
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        face_descriptor = np.array([face_descriptor])

        # predict image with classes
        pred = model.predict(face_descriptor)[0]

        cv2.putText(frame, pred.title(), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)


def predict_camera():
    data = np.load(file_data)
    classes = np.load(file_classes)

    print("[INFO] training classifier...")
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(data, classes)

    print("[INFO] camera sensor warming up...")
    vs = VideoStream(0).start()
    time.sleep(2.0)
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        frame = vs.read()

        # detect faces in the grayscale frame
        dets = detector(frame, 0)

        # loop over the face detections
        for det in dets:
            x, y, z, t = det.left(), det.top(), det.right(), det.bottom()
            cv2.rectangle(frame, (x, y), (z, t), (0, 255, 0), 2)

            shape = sp(frame, det)
            face_descriptor = facerec.compute_face_descriptor(frame, shape)
            face_descriptor = np.array([face_descriptor])

            # predict image with classes
            pred = model.predict(face_descriptor)[0]

            cv2.putText(frame, pred.title(), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    start = time.time()

    # save_features()
    # predict_camera()
    # predict("class/class3.jpg")

    image_dir = "group"
    for filename in sorted(os.listdir(image_dir)):
        file_path = os.path.join(image_dir, filename)
        predict(file_path)

    end = time.time()
    print("Time:{}s".format(end - start))
