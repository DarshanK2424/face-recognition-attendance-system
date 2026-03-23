import cv2
import time
import os
import numpy as np


recogniser=cv2.face.LBPHFaceRecognizer_create()
recogniser.read("face_model.yml")


label_map=np.load("label_map.npy",allow_pickle=True).item()

face_cascade=cv2.CascadeClassifier("../FACE/haarcascade_frontalface_alt.xml")


capture=cv2.VideoCapture(0)
f=open('attendance.csv','a')


marked = set()

try:
    with open("attendance.csv", "r") as f_read:
        for line in f_read:
            entry = line.strip().split(",")[0]
            marked.add(entry)
except FileNotFoundError:
    pass


while True:
    ret,frame=capture.read()

    if ret is None:
        break
    frame=cv2.flip(frame,1)

   

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face=face_cascade.detectMultiScale(gray,1.1,5)

    for (x,y,w,h) in face:
        faces=gray[y:y+h,x:x+w]

        label,confidence=recogniser.predict(faces)

        if confidence<75:
            name=label_map[label]

        else:
            name="unknown"


        if name!="unknown" and name not in marked:
            current_time=time.strftime("%H:%M:%S")

            f.write(name+","+current_time+"\n")

            marked.add(name)

        

        text=f"{name} ({int(confidence)})"

        cv2.putText(frame,text,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


    cv2.imshow("Attendance",frame)

    key=cv2.waitKey(1)

    if key==ord('q'):
        break


capture.release()
cv2.destroyAllWindows()


        




