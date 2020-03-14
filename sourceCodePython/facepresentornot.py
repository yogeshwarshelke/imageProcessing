import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# image path
data_path="/home/yogesh/Desktop/database/"
onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]

training_data, labels = [], []

# read image
for i, file in enumerate(onlyfiles):
    image_path=data_path+onlyfiles[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    training_data.append(np.asarray(images, dtype=np.uint8))
    labels.append(i)

# face training model
lables = np.asarray(labels, dtype=np.uint32)
model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(training_data), np.asarray(labels))
print("Model Training Complete")

# haarcascade xml file
face_classifier=cv2.CascadeClassifier('/home/yogesh/PycharmProjects/imageprocessing/haarcascades/haarcascade_frontalface_default.xml')

#face detection
def face_detector(img, size = 0.5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),2)
        roi=img[y:y+h, x:x+w]
        roi=cv2.resize(roi,(200,200))

    return img,roi


# main code
#cap=cv2.VideoCapture("/home/yogesh/PycharmProjects/imageprocessing/totaldhamal.mp4")
cap=cv2.VideoCapture(0)  # webcam
while True:
    ret,frame=cap.read()
    image, face=face_detector(frame)
    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result=model.predict(face)  # face train

        if result[1]<500:
            confidence=int(100*(1-(result[1])/300))      # confidence face
            display_string=str(confidence)+'%confidence it is user'
        cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(250,120,250),2)

        if confidence>75:
            cv2.putText(image,"Unlocked",(150,250),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow("image cropper",image)
        else:
            cv2.putText(image, "Locked", (150, 250), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("image cropper", image)
    except:
        cv2.putText(image, "Face not found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("image cropper", image)
        pass

    if cv2.waitKey(30)==13 :
        break

# exit
cap.release()
cv2.destroyAllWindows()