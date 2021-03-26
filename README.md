# Face-Mask-Detection

<p>Corona Virus disease has spread to more than 213 countries infecting more than 7 million people and
killing over 403,202 globally, according to data compiled by worldometer (as on June 5, 2020) <br>

To limit Corona Virus spread, social distancing and observing hygiene standards like compulsory
wearing of mask, use of hand gloves, face shield, and use of sanitizer is very important. <br>

Many Organizations are making it compulsory to follow social distancing and wearing of face mask.</p>

<p align='center'>
    <img alt="IMG" src="https://github.com/AkshitTayade/Face-Mask-Detection/blob/main/static/demo.gif?raw=true"/>
</p>

## Project Outliners
1. Identify human Face and Mouth in each frame of input video
2. Identify Person is using Mask or not

3. Create HAAR Cascade object using ‘CascadeClassifier’ function and
‘haarcascade_frontalface_default.xml’. [Object Detection using Haar feature-based cascade
classifiers is an effective object detection method proposed by Paul Viola and Michael Jones
in their paper, &quot;Rapid Object Detection using a Boosted Cascade of Simple Features&quot; in
2001. It is a machine learning based approach where a cascade function is trained from a lot
of positive and negative images. It is then used to detect objects in other images. 

4. Read image using function ‘imread’ (or ‘read’ for video/ camera input) function

5. Detect face using ‘detectMultiScale’ function

6. We Store Data into two categories <br>
a) With Mask <br>
b) Without Mask

7. We Train the CCN model <br>
<p align='center'>
    <img alt="IMG" src="https://github.com/AkshitTayade/Face-Mask-Detection/blob/main/static/Screenshot%202021-03-26%20at%201.08.16%20PM.png?raw=true" width="500" height="350" />
</p>

8. We have tested our model with test images and got correct output for the same.

> Note: Code is written considering single user face identification

## Running the application
#### Method 1: <br>
You can directly run the application on your desktop using this [file](https://github.com/AkshitTayade/Face-Mask-Detection/blob/main/live_detection.py) 
>Run this in your terminal: ***python3 live_detection.py***

#### Method 2: <br>
If you wish to run this application on browser using Flask, run [this](https://github.com/AkshitTayade/Face-Mask-Detection/blob/main/app.py)
>Run this in your terminal: ***python3 app.py***

<hr></hr>

>Dataset can be found here: 

## Technologies used
* Tensorflow, Keras
* OpenCV
* Flask

<hr></hr>

Created by: [Akshit Tayade](https://github.com/AkshitTayade) and [Aayush Maru](https://github.com/aayushmaru18)
