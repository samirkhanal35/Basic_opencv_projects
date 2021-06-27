import cv2

cap = cv2.VideoCapture(0)

while True:
    timer = cv2.getTickCount()
    success, img = cap.read()

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img, str(fps),(20,20), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,255),2)

    cv2.imshow('video', img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break



