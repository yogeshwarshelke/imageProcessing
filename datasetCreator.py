import cv2
import sqlite3
import os.path


# input file -------------------------------------
num=int(input("Enter Num:"))
if num==0:
    cam = cv2.VideoCapture(0)
elif num==1:
    fname=input("Enter File Name:")
    cam=cv2.VideoCapture("/home/yogesh/PycharmProjects/imageprocessing/"+str(fname)+".mp4")
else:
    cam=cv2.imread("/home/yogesh/PycharmProjects/imageprocessing/multiple.jpg")


# Haarcascade .xml file -------------------------------
f = cv2.CascadeClassifier('/home/yogesh/PycharmProjects/imageprocessing/haarcascades/haarcascade_frontalface_default.xml')

#Database Connection sqlite3
conn=sqlite3.connect("/home/yogesh/PycharmProjects/imageprocessing/faceDatabase2.db")
cursor = conn.cursor()

sql='''CREATE TABLE IF NOT EXISTS Persons (pid int primary key,name varchar(20))'''
cursor.execute(sql)

# function  Execute query------------------------
def insertOrUpdate(ID,Name):
    conn = sqlite3.connect("faceDatabase2.db")
    cursor = conn.cursor()
    cmd = "select * from Persons where pid=" + str(ID)
    cursor1 = cursor.execute(cmd)
    isRecordExist=0

    for row in cursor1:
        isRecordExist = 1
    if(isRecordExist == 1):
        #cmd = "update Persons set Name ="+str(Name) +"Where pid"+str(ID)+")"
        sql_update_query = """Update Persons set name= ? where pid = ?"""
        data = (name,ID)
        cursor.execute(sql_update_query, data)
    else:
        cursor.execute("insert into Persons (pid,name) values (?, ?)",(id,name))

    conn.commit()
    sql = "select * from Persons"
    cursor.execute(sql)

    rows = cursor.fetchall()

    for row in rows:
        print(row)

    #conn.commit()
    conn.close()

# Main code -----------------------------------
id = input("Enter ID:")
name = input("Enter the Name:")
insertOrUpdate(id,name)
sampleNum=0
while(True):
    if num==1 or num==0:
        ret,img=cam.read()
    else:
        img=cam

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#img
    faces=f.detectMultiScale(img,1.3,2)
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1
        cv2.imwrite("/home/yogesh/Desktop/database/user."+str(id)+'.'+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow("face",img)

    if cv2.waitKey(1)==ord('q') or cv2.waitKey(1)==13: #enter key
        break;
    #cv2.waitKey(100)


    if sampleNum>50:
        #cam.release()
        #cv2.destroyAllWindows()
        break

if num==0 or num==1:
    cam.release()

cv2.destroyAllWindows()

