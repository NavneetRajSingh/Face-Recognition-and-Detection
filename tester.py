import cv2

import os
import numpy as np
import faceRecognition

test_img=cv2.imread('TestImage\RandomImage1.jpeg')
faces_detected,gray_img=faceRecognition.faceDetection(test_img)
print("Face Detected:",faces_detected)


faces,faceID=faceRecognition.labels_for_training_data('TrainingImage')
face_recognizer=faceRecognition.train_classifier(faces,faceID)
face_recognizer.write('trainingData.yml')


face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')

name={0:"Messi", 1:"Ronaldo"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    faceRecognition.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>110):
        continue
    faceRecognition.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("face Detection tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows





