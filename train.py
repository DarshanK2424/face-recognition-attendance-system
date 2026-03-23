import cv2
import os
import numpy as  np


datapath="dataset"
face=[]
label=[]
label_map={}
count=0

for person_name in os.listdir(datapath):
    person_path=os.path.join(datapath,person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[count]=person_name

    for file in os.listdir(person_path):
        img_path=os.path.join(person_path,file)

        img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)


        face.append(img)
        label.append(count)

    
    count+=1



face=np.array(face,dtype='object')
label=np.array(label)


face_recogniser=cv2.face.LBPHFaceRecognizer_create()
face_recogniser.train(face,label)


face_recogniser.save("face_model.yml")

np.save("label_map.npy",label_map)

print("Trainig  complete")

