import cv2,os
import sqlite3

# LBPH Algoritham and Haarcascde xml file
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("/home/yogesh/Desktop/trainer/trainer.yml") # Trining faces from trainer.yml file

cascadPath="/home/yogesh/PycharmProjects/imageprocessing/haarcascades/haarcascade_frontalface_default.xml"
facecascade=cv2.CascadeClassifier(cascadPath)
Path='dataset'

# Execute face found or not
def getProfile(id):
    conn=sqlite3.connect("faceDatabase2.db")
    cmd="select * from Persons where pid="+str(id)
    cursor=conn.execute(cmd)
    profile=None

    for row in cursor:
        profile=row
    conn.close()
    return profile

#
#cam=cv2.VideoCapture("/home/yogesh/PycharmProjects/imageprocessing/chadani1.mp4")
cam=cv2.VideoCapture(0)
font1=cv2.FONT_HERSHEY_SIMPLEX
#font=cv2.cv.InitFont(font1,1,1,0,1,1)

# face known or unknown
while True:
    ret, img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces=facecascade.detectMultiScale(gray,1.2,5)

    for(x,y,w,h) in faces:
        id, conf=recognizer.predict(gray[y:y+h, x:x+w])

        #print("id:",id,"conf:",conf)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        profile=getProfile(id)

        if(conf<80): #cv2.cv.fromarray(img) #(x,y+h+30)
            if(profile!=None):
                cv2.putText(img,str(profile[0]),(x,y+h+30),font1,1,255,2)
                cv2.putText(img,str(profile[1]),(x,y+h+60),font1,1,255,2)
               # cv2.cv.PutText(cv2.cv.fromarray(img), str(profile[3]), (x, y + h + 90),  255)
                #cv2.cv.PutText(cv2.cv.fromarray(img), str(profile[4]), (x, y + h + 120),  255)
        else:
            cv2.putText(img, "Unknown", (x, y + h + 30), font1, 1, 255, 2)

    cv2.imshow("img",img)
    if cv2.waitKey(10) == 13:
        break

# exit
cam.release()
cv2.destroyAllWindows()