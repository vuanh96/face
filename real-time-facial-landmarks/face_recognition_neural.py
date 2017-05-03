import time

import cv2
import dlib
import imutils
import keras
import numpy as np
import os

from imutils.video import VideoStream
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

img_width, img_height = 100, 100
batch_size = 16

train_data_dir = "at10"
file_data = "data.npy"
file_classes = "classes.npy"
model_weights_path = "model_weights.h5"

predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"  # resnet-34

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# determine labels
labels = []
for subdir in sorted(os.listdir(train_data_dir)):
    if os.path.isdir(os.path.join(train_data_dir, subdir)):
        labels.append(subdir)

num_label = len(labels)

# build model
print("[INFO] building model ...")
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=128))
model.add(Dropout(0.5))
model.add(Dense(num_label, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])


def save_features():
    print("[INFO] compute descriptor.......")

    face_descriptors = []
    classes = []
    for rootdir, dirnames, filenames in os.walk(train_data_dir):
        for subdirname in sorted(dirnames):
            subject_path = os.path.join(rootdir, subdirname)
            it = 0
            for filename in os.listdir(subject_path):
                file_path = os.path.join(subject_path, filename)
                img = cv2.imread(file_path)
                dets = detector(img, 1)
                for det in dets:
                    shape = sp(img, det)
                    face_descriptor = facerec.compute_face_descriptor(img, shape)
                    face_descriptors.append(face_descriptor)
                    classes.append(subdirname)
                    it += 1
            print(subdirname, " : ", it, " face detected")
    np.save(file_data, face_descriptors)
    np.save(file_classes, classes)


def train_model():
    data = np.load(file_data)
    classes = np.load(file_classes)

    # transform the labels into vectors in the range [0, num_classes]-- this
    # generates a vector for each label where the index of the label
    # is set to `1` and all other entries to `0`
    le = LabelEncoder()
    classes = le.fit_transform(classes)
    classes = np_utils.to_categorical(classes, num_label)

    # partition the data into training and testing splits, using 75%
    # of the data for training and the remaining 25% for testing
    print("[INFO] constructing training/testing split...")
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        data, classes, test_size=0.25, random_state=5)
    (train_data, validation_data, train_labels, validation_labels) = train_test_split(
        trainData, trainLabels, test_size=0.1, random_state=5)
    print("[INFO] training data...")
    model.fit(train_data, train_labels,
              epochs=1500,
              batch_size=batch_size,
              verbose=2,
              validation_data=(validation_data, validation_labels)
              )

    # show the accuracy on the testing set
    print("[INFO] evaluating on testing set...")
    (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=batch_size, verbose=1)
    print("\n[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

    # save weigth for top_model
    model.save_weights(model_weights_path)


def predict_camera():
    print("[INFO] load weights ...")
    model.load_weights(model_weights_path)
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
            preds = model.predict(face_descriptor)[0]

            # loop over the predictions and display rank predictions + probabilities to our terminal
            P = []
            for i in range(num_label):
                label = labels[i]
                prob = preds[i]
                P.append((label, prob))
            P.sort(key=lambda x: x[1], reverse=True)

            cv2.putText(frame, P[0][0], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("Person :")
            for (i, (label, prob)) in enumerate(P[0:5]):
                print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


def predict(image_path):
    print("[INFO] load weights ...")
    model.load_weights(model_weights_path)

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
        preds = model.predict(face_descriptor)[0]

        # loop over the predictions and display rank predictions + probabilities to our terminal
        P = []
        for i in range(num_label):
            label = labels[i]
            prob = preds[i]
            P.append((label, prob))
        P.sort(key=lambda x: x[1], reverse=True)

        cv2.putText(frame, P[0][0], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("Person :")
        for (i, (label, prob)) in enumerate(P[0:5]):
            print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

    # show the frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)


if __name__ == '__main__':
    start = time.time()

    # save_features()
    # train_model()
    predict_camera()
    # predict("class/class2.jpg")

    # image_dir = "group"
    # for filename in sorted(os.listdir(image_dir)):
    #     file_path = os.path.join(image_dir, filename)
    #     predict(file_path)

    end = time.time()
    print("Time:{}s".format(end - start))
