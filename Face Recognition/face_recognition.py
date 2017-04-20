import cv2
import numpy
from create_csv import create_csv
import imutils


# read file csv
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
            img = cv2.imread(fields[0], 0)
            img_resize = cv2.resize(img, (100, 100))
            # img_resize = cv2.equalizeHist(img_resize)
            picPath.append(img_resize)
            picIndex.append(int(fields[1]))
    return picPath, picIndex


def training():
    # create csv_file at.txt
    labelsInfo = create_csv("at", "at.txt")

    print("Start training..........")
    images, labels = readFileNames("at.txt")
    model.train(images, numpy.array(labels))
    for i in range(len(labelsInfo)):
        model.setLabelInfo(i, labelsInfo[i])
    model.save(file_yml)
    print("Training Finished")
    return model


face_cascade = cv2.CascadeClassifier("../data/haarcascades/haarcascade_frontalface_default.xml")


def predict(type_predict):
    video_capture = cv2.VideoCapture(0)
    it = 0
    while True:
        if type_predict == "camera":
            ret, frame = video_capture.read()
        else:
            frame = cv2.imread(type_predict, 1)
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        for (x, y, w, h) in faces:
            # resize face detect
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (100, 100), 1.0, cv2.INTER_CUBIC)
            prediction = model.predict(face_resized)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # put names on image
            if prediction[0] == -1:
                box_text = "Unknow"
            else:
                box_text = model.getLabelInfo(prediction[0]) + ":" + str(int(prediction[1]))
            pos_x = x
            pos_y = max(y - 5, 0)
            try:
                cv2.putText(frame, box_text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, w / 150, (0, 0, 255), 2)
            except:
                raise IndexError("Error Font!")

        # Display the resulting frame
        if type_predict == "camera":
            it += 1
            cv2.putText(frame, "Frame :" + str(it), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        cv2.imshow("Face Recognizer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__ == '__main__':
    model = None
    type_recog = 2
    file_yml = ""
    if type_recog == 1:
        model = cv2.face.createEigenFaceRecognizer()
        file_yml = "eigen_face.yml"
    elif type_recog == 2:
        model = cv2.face.createFisherFaceRecognizer()
        file_yml = "fisher_face.yml"
    elif type_recog == 3:
        model = cv2.face.createLBPHFaceRecognizer()
        file_yml = "LBPH_face.yml"

    # model = training()
    model.load(file_yml)

    predict("test_images/g7.JPG")