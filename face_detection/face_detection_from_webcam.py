import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

haar_cascade = cv.CascadeClassifier('har_face.xml')

while True:
    ret, img = cap.read()
    img = cv.flip(img, 1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=1)
    for (x, y, w, h) in faces_rect:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    cv.imshow("recognized frame", img)
    k = cv.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv.destroyAllWindows()
