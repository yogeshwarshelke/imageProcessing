import cv2

# Haar cascade xml file
face_classifier=cv2.CascadeClassifier('/home/yogesh/PycharmProjects/imageprocessing/haarcascades/haarcascade_frontalface_default.xml')

# face code
def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(img,1.3,5)
    if faces is():
        return None
    for(x,y,w,h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_color = img[y:y + h, x:x + w]
        cropped_faces=img[y:y+h,x:x+w]
        cropped_faces = cv2.resize(cropped_faces, (1366,768))

    return cropped_faces

# main code
# input = 0 or 1
num=int(input("Enter Num:"))

if num==0:
    cap = cv2.VideoCapture(0)
else:
    cap=cv2.VideoCapture("/home/yogesh/PycharmProjects/imageprocessing/chadani1.mp4")

# face Detection
count=0
while True:
    ret,frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        #face=cv2.resize(face_extractor(frame),(100,120))
        #face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        file_name_path="/home/yogesh/Desktop/sample/user"+str(count)+".jpg"
        cv2.imwrite(file_name_path,frame)
        cv2.putText(frame,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('face cropper',frame)
    else:
        #print('Face not Found..')
        pass

    if cv2.waitKey(10)==13 or count==50:  # total not of faces
        break;

    #cv2.waitKey(50)
# Exit
cap.release()
cv2.destroyAllWindows()
print("Collection Sample Images...")