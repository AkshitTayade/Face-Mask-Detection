import cv2
import numpy as np 
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# loading the haar cascade for facial detection
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# intialize the video variable from cv2
cap = cv2.VideoCapture(0)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# loading the Deep Learning model
model = load_model("fask_mask_detector.h5")
print("Model Loaded Successfully !")

def predict(img):

  prediction = []
  #img = cv2.imread(img_path)
  
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (224,224))
  img = img_to_array(img)
  img = img.reshape(1, 224, 224, 3)

  predicted_class = model.predict_classes(img)
  predicted_class_proba = round(max(model.predict(img)[0])*100, 2)

  # "{'with_mask': 0, 'without_mask': 1}"

  if predicted_class[0] == 1:
    prediction.append('No Mask')

  else:
    prediction.append('Mask')

  prediction.append(predicted_class_proba)

  return(prediction)

# reading the frames from webcam
while(cap.isOpened()):

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # detect face from live webcam feed
    faces = face.detectMultiScale(frame, 1.3, 5)

    for (x,y,w,h) in faces:

        # region of interest
        face_roi = frame[y:y+h+15, x:x+w]
  
        # drawing rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w, y+h+15), (255,255,25), 2)

        prediction = predict(face_roi)

        if prediction[0] == 'No Mask':
            cv2.putText(frame, f'No Mask {prediction[1]}', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h+15), (0,0,255), 2)
        
        else:
            cv2.putText(frame, f'Mask {prediction[1]}', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h+15), (0,255,0), 2)

    cv2.putText(frame, "Press 'q' to quit", (900, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()