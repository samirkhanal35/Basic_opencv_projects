import os
import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('har_face.xml')

people = []
DIR = '/home/samir/Documents/basic_opencv_projects/Faces/train'
for i in os.listdir('/home/samir/Documents/basic_opencv_projects/Faces/train'):
    people.append(i)


features = []
labels = []


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

                # cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)


create_train()
print("----------Training Completed-----------")
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer
face_recognizer.train(features, labels)

face_recognizer.save('face_trained_model.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
