from imutils.video import WebcamVideoStream
import cv2 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


class VideoCamera(object):
    
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        self.stream = WebcamVideoStream(src=0).start()
        
        #reading the cascade file downloaded
        self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # laoding our CNN model we trained
        self.model = load_model("fask_mask_detector.h5")

    def predict(self, img): 
        prediction = []
        #img = cv2.imread(img_path)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        img = img_to_array(img)
        img = img.reshape(1, 224, 224, 3)

        predicted_class = self.model.predict_classes(img)
        predicted_class_proba = round(max(self.model.predict(img)[0])*100, 2)

        # "{'with_mask': 0, 'without_mask': 1}"

        if predicted_class[0] == 1:
            prediction.append('No Mask')

        else:
            prediction.append('Mask')

        prediction.append(predicted_class_proba)

        return(prediction)


    def get_frame(self):
        # read the webcam
        image = self.stream.read()

        #detect faces from webcam
        faces = self.detector.detectMultiScale(image, 1.3, 5)

        for (x,y,w,h) in faces:

            # region of interest
            face_roi = image[y:y+h+15, x:x+w]

            # drawing rectangle around the face
            cv2.rectangle(image, (x,y), (x+w, y+h+15), (255,255,25), 2)

            prediction = self.predict(face_roi)

            if prediction[0] == 'No Mask':
                cv2.putText(image, f'No Mask {prediction[1]}', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.rectangle(image, (x,y), (x+w, y+h+15), (0,0,255), 2)
        
            else:
                cv2.putText(image, f'Mask {prediction[1]}', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.rectangle(image, (x,y), (x+w, y+h+15), (0,255,0), 2)

        # converting into encoded form  
        ret, jpeg = cv2.imencode('.jpg', image)
        
        data = []
        
        data.append(jpeg.tobytes())
       
        return(data)
 