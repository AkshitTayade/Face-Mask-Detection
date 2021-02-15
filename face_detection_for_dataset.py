import cv2
import numpy as np 
import os

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# loading the haar cascade for facial detection
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# intialize the video variable from cv2
cap = cv2.VideoCapture(0)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# creating directories
if os.path.exists('dataset'):
    pass

else:
    os.mkdir('dataset')
    os.mkdir('dataset/Mask')
    os.mkdir('dataset/Not mask')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# first enter the user's name
filename = input("Enter your name: ")
print()
print("Which Label you want to save in? \n 1. Mask \n 2. Not Mask")
choice = input("Enter 1 or 2: ")
print()
print("")

# depending upon user's label
# we want to save the face accordingly
if choice == "1":
    saving_dir = "dataset/Mask"
else:
    saving_dir = "dataset/Not Mask"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

count = 0

# reading the frames from webcam
while(cap.isOpened()):

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face from live webcam feed
    faces = face.detectMultiScale(frame_gray, 1.3, 5)

    for (x,y,w,h) in faces:

        # region of interest
        face_roi = frame_gray[y:y+h+15, x:x+w]

        if cv2.waitKey(1) & 0xFF == ord('s'):

            count += 1

            # Using cv2.imwrite() method 
            # Saving the image 
            img_name = os.path.join(saving_dir, filename + '_' + str(count) + '.jpg')
            cv2.imwrite(img_name, face_roi)

            print("Saved successfully to ", img_name)
            cv2.putText(frame, f"Saved {count}", (80, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
            
        # drawing rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w, y+h+15), (0,255,0), 2)

        #cv2.imshow('re', face_roi)
    cv2.putText(frame, "Press 's' to save", (900, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, "Press 'q' to quit", (900, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()