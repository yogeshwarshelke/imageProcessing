import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("/home/yogesh/Desktop/trainer/trainer.yml")
cascadePath = '/home/yogesh/PycharmProjects/imageprocessing/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

id =0
names = ["mangesh", "yogesh", "harshad", "arjun", "Z", "W"]

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:

    ret, img = cam.read()
    #img = cv2.flip(img, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        #id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        #confidence = int(100 * (1 - (confidence) / 300))
        if (confidence > 70 and confidence < 100):
            id = names[1]
            id2 =os.name[2]
            con = confidence
            #confidence = int(100 * (1 - (confidence[1]) / 300))
            confidence = "  {0}%".format(round(100 - confidence))
            if (id =='yogesh' and con > 70):
                cv2.putText(img,"yogesh", (x + 5, y + 25), font, 1, (0, 0, 255), 1)
                #cv2.imshow("camera", img)
        else:
            id = "unknown"
            confidence = " {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            #cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        cv2.imshow("camera", img)



    k = cv2.waitKey(1) & 0xff  # Press ‘ESC’ for exiting video
    if k == 13:
        break

print('\n[INFO] Exiting Program and cleanup stuff')
cam.release()
cv2.destroyAllWindows()