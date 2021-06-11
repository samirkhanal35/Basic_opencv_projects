import numpy as np
import cv2 as cv
import os

people = []
for i in os.listdir('/home/samir/Documents/basic_opencv_projects/Faces/train'):
    people.append(i)

haar_cascade = cv.CascadeClassifier('har_face.xml')


# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained_model.yml')

img = cv.imread(
    '/home/samir/Documents/basic_opencv_projects/Faces/train/Ben Afflek/6.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x: x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print('Label: ', people[label], 'with confidence of ', confidence)

    cv.putText(img, str(people[label]), (20, 20),
               cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow("Detected image", img)
cv.waitKey(0)
