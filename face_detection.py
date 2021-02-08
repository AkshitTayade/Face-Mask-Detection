import cv2
import numpy as np 

# loading the haar cascade for facial detection
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# intialize the video variable from cv2
cap = cv2.VideoCapture(0)

# reading the frames from webcam
while(cap.isOpened()):

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # detect face from live webcam feed
    faces = face.detectMultiScale(frame, 1.3, 5)

    for (x,y,w,h) in faces:

        # region of interest
        face_roi = frame[y:y+h+20, x:x+w]

        # drawing rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w, y+h+20), (0,0,255), 2)

        cv2.imshow('re', face_roi)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()