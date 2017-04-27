from time import time

import cv2
import dlib
import keras
import numpy as np
import os
from skimage import io

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications import VGG16, imagenet_utils
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

img_width, img_height = 100, 100
batch_size = 16

train_data_dir = "at"
file_data = "data.npy"
file_classes = "classes.npy"
model_weights_path = "model_weights.h5"

predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# determine labels
labels = []
for rootdir, dirnames, filenames in os.walk(train_data_dir):
    for subdirname in dirnames:
        labels.append(subdirname)

num_label = len(labels)

# build model

model = Sequential()
model.add(Flatten(input_shape=(1, 1, 128)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_label, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])


def save_bottlebeck_features():
    print("[INFO] compute descriptor.......")

    face_descriptors = []
    classes = []
    for rootdir, dirnames, filenames in os.walk(train_data_dir):
        for subdirname in dirnames:
            subject_path = os.path.join(rootdir, subdirname)
            for filename in os.listdir(subject_path):
                file_path = os.path.join(subject_path, filename)
                img = io.imread(file_path)
                dets = detector(img, 1)
                for det in dets:
                    shape = sp(img, det)
                    face_descriptor = facerec.compute_face_descriptor(img, shape)
                    face_descriptors.append([[face_descriptor]])
                    classes.append(subdirname)

    np.save(file_data, face_descriptors)
    np.save(file_classes, classes)


def train_top_model():
    data = np.load(file_data)
    print(data.shape)
    classes = np.load(file_classes)
    print(classes.shape)

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
              epochs=500,
              batch_size=batch_size,
              verbose=2,
              validation_data=(validation_data, validation_labels))

    # show the accuracy on the testing set
    print("[INFO] evaluating on testing set...")
    (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=batch_size, verbose=1)
    print("\n[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

    # save weigth for top_model
    model.save_weights(model_weights_path)


def predict(imagePath):
    model.load_weights(model_weights_path)
    print("load weights success!")

    img = io.imread(imagePath)
    dets = detector(img, 1)
    for det in dets:
        x, y, z, t = det.left(), det.top(), det.right(), det.bottom()
        cv2.rectangle(img, (x, y), (z, t), (0, 255, 0), 2)

        shape = sp(img, det)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptor = np.array([[[face_descriptor]]])

        # predict image with classes
        preds = model.predict(face_descriptor)[0]

        # loop over the predictions and display rank predictions + probabilities to our terminal
        P = []
        for i in range(len(labels)):
            label = labels[i]
            prob = preds[i]
            P.append((label, prob))
        P.sort(key=lambda x: x[1], reverse=True)
        for (i, (label, prob)) in enumerate(P):
            print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

    # load the image via OpenCV, draw the top prediction on the image,
    # and display the image to our screen
    # orig = cv2.imread(imagePath)
    # cv2.putText(orig, "Label: {}, {:.2f}%".format(P[0][0], P[0][1] * 100),
    #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Classification", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    start = time()

    # save_bottlebeck_features()
    # train_top_model()
    predict("group/g13.JPG")

    end = time()
    print("Time:{}s".format(end - start))