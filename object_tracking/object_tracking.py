import cv2

cap = cv2.VideoCapture(0)

# tracker = cv2.legacy_TrackerMOSSE.create()
tracker = cv2.legacy_TrackerCSRT.create()
success, img = cap.read()
bbox = cv2.selectROI("Tracking", img,False)
tracker.init(img, bbox)


def drawBox(img, bbox):
    x ,y ,w ,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),
    cv2.rectangle(img, (x,y), ((x+w),(y+h)), (255,0,255),3,1)
    cv2.putText(img, "Tracking", (75, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


while True:
    timer = cv2.getTickCount()
    success, img = cap.read()
    img = cv2.flip(img, 1)

    success, bbox = tracker.update(img)

    if success:
        drawBox(img, bbox)
    else:
        cv2.putText(img, "Lost", (75, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    txt = "fps: "+str(int(fps))
    cv2.putText(img, txt,(75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,255),2)

    cv2.imshow('video', img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break



