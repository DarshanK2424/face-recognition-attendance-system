import cv2
import os

face_detection=cv2.CascadeClassifier("../FACE/haarcascade_frontalface_alt.xml")

username=""
path=f"dataset/{username}"

os.makedirs(path,exist_ok=True)


cap=cv2.VideoCapture(0)

count=0
max=50

while True:
    ret,frame=cap.read()

    

    if ret is None:
        break

    frame=cv2.flip(frame,1)

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=face_detection.detectMultiScale(gray,1.1,5)

    for (x,y,w,h) in faces:
        face=gray[y:y+h , x:x+w]

        

        file_path= os.path.join(path,f"{count}.jpg")
        cv2.putText(frame,"Face detected",(x,y-50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
        cv2.imwrite(file_path,face)

        count+=1
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

    
    cv2.imshow("Frame",frame)

    key=cv2.waitKey(1)
    if key==ord('q') or count>=max:
        break

cap.release()
cv2.destroyAllWindows()

        

